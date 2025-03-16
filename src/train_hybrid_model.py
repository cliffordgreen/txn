import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our custom modules
from src.models.train import TransactionClassifier
from src.data_processing.transaction_graph import TransactionGraphBuilder, create_train_val_test_split
from src.models.hybrid_transaction_model import HybridTransactionModel, HybridTransactionEnsemble
from src.train_transaction_classifier import generate_synthetic_data, plot_training_metrics


class HybridTransactionClassifier:
    """
    Class for training and evaluating a hybrid transaction classifier
    that combines GNN, tabular, and transformer approaches.
    """
    
    def __init__(self, hidden_channels: int = 128, num_layers: int = 3, 
                 dropout: float = 0.4, conv_type: str = 'sage', heads: int = 2,
                 use_jumping_knowledge: bool = True, use_batch_norm: bool = True,
                 use_self_supervision: bool = True, use_tabular_model: bool = True,
                 use_graph_transformers: bool = True, use_ensemble: bool = True,
                 ensemble_size: int = 3, lr: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize the hybrid transaction classifier.
        
        Args:
            hidden_channels: Dimension of hidden node features
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat')
            heads: Number of attention heads for GAT/transformers
            use_jumping_knowledge: Whether to use jumping knowledge
            use_batch_norm: Whether to use batch normalization
            use_self_supervision: Whether to use self-supervised auxiliary tasks
            use_tabular_model: Whether to include tabular MLP model
            use_graph_transformers: Whether to use graph transformer layers
            use_ensemble: Whether to use ensemble of models
            ensemble_size: Number of models in the ensemble
            lr: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.heads = heads
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_batch_norm = use_batch_norm
        self.use_self_supervision = use_self_supervision
        self.use_tabular_model = use_tabular_model
        self.use_graph_transformers = use_graph_transformers
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.graph = None
        self.graph_builder = None
    
    def prepare_data(self, transactions_df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare transaction data by building a graph and extracting raw features.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Raw features tensor for tabular model
        """
        # Initialize graph builder
        self.graph_builder = TransactionGraphBuilder()
        
        # Build graph from transaction data
        self.graph = self.graph_builder.build_graph(transactions_df)
        
        # Add self-loops if needed
        self._add_self_loops()
        
        # Add reverse edges for better message passing
        self._add_reverse_edges()
        
        # Split graph into train/val/test sets
        self.graph = create_train_val_test_split(self.graph)
        
        # Extract raw features for tabular model
        raw_features = self._extract_raw_features(transactions_df)
        
        return raw_features
    
    def _add_self_loops(self):
        """
        Add self-loops to the graph for better information propagation.
        """
        for node_type in self.graph.node_types:
            # Create self-loops: node points to itself
            num_nodes = self.graph[node_type].x.size(0)
            self_indices = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([self_indices, self_indices])
            self.graph[node_type, 'self', node_type].edge_index = edge_index
    
    def _add_reverse_edges(self):
        """
        Add reverse edges to the graph for bidirectional message passing.
        """
        # Transaction -> Merchant becomes Merchant -> Transaction as well
        src_nodes = self.graph['transaction', 'belongs_to', 'merchant'].edge_index[1]
        dst_nodes = self.graph['transaction', 'belongs_to', 'merchant'].edge_index[0]
        self.graph['merchant', 'rev_belongs_to', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Transaction -> Category becomes Category -> Transaction as well
        src_nodes = self.graph['transaction', 'has_category', 'category'].edge_index[1]
        dst_nodes = self.graph['transaction', 'has_category', 'category'].edge_index[0]
        self.graph['category', 'rev_has_category', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
    
    def _extract_raw_features(self, transactions_df: pd.DataFrame) -> torch.Tensor:
        """
        Extract raw features from transaction dataframe for the tabular model.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Raw features tensor
        """
        # Select numerical features
        numerical_cols = [col for col in transactions_df.columns 
                         if transactions_df[col].dtype in ['int64', 'float64'] 
                         and col not in ['merchant_idx', 'category_idx', 'transaction_id']]
        
        # Create copy to avoid modifying original
        df = transactions_df.copy()
        
        # Add engineered features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Log transform of amount
        df['log_amount'] = np.log1p(df['amount'])
        
        # Add merchant statistics if available
        if 'merchant_idx' in df.columns:
            # Group by merchant and compute aggregated statistics
            merchant_stats = df.groupby('merchant_idx')['amount'].agg(['mean', 'std', 'max']).reset_index()
            # Rename columns
            merchant_stats.columns = ['merchant_idx', 'merchant_mean_amount', 'merchant_std_amount', 'merchant_max_amount']
            # Merge with original dataframe
            df = df.merge(merchant_stats, on='merchant_idx', how='left')
        
        # Select all numerical features including engineered ones
        numerical_cols = [col for col in df.columns 
                         if df[col].dtype in ['int64', 'float64'] 
                         and col not in ['merchant_idx', 'category_idx', 'transaction_id']]
        
        # Convert to tensor
        raw_features = torch.FloatTensor(df[numerical_cols].values)
        
        return raw_features
    
    def initialize_model(self, num_categories: int = 400) -> None:
        """
        Initialize the hybrid model and optimizer.
        
        Args:
            num_categories: Number of transaction categories
        """
        # Initialize model
        if self.use_ensemble:
            self.model = HybridTransactionEnsemble(
                num_models=self.ensemble_size,
                hidden_channels=self.hidden_channels,
                num_layers=[2, 3, 4],  # Vary layers across models
                dropout=self.dropout,
                conv_types=['gcn', 'sage', 'gat'],  # Vary conv types
                use_bagging=True
            )
        else:
            self.model = HybridTransactionModel(
                hidden_channels=self.hidden_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
                conv_type=self.conv_type,
                heads=self.heads,
                use_jumping_knowledge=self.use_jumping_knowledge,
                use_batch_norm=self.use_batch_norm,
                use_self_supervision=self.use_self_supervision,
                use_tabular_model=self.use_tabular_model,
                use_graph_transformers=self.use_graph_transformers
            )
            
            # Update classifier to match number of categories
            self.model.classifier = torch.nn.Linear(self.hidden_channels, num_categories)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=True
        )
    
    def train(self, raw_features: torch.Tensor, num_epochs: int = 100, 
              patience: int = 15, auxiliary_weight: float = 0.2) -> dict:
        """
        Train the hybrid model on the transaction graph.
        
        Args:
            raw_features: Raw features tensor for tabular model
            num_epochs: Maximum number of training epochs
            patience: Number of epochs to wait for validation improvement before early stopping
            auxiliary_weight: Weight for auxiliary self-supervised tasks
            
        Returns:
            Dictionary containing training and validation metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized before training")
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.graph = self.graph.to(device)
        
        if raw_features is not None:
            raw_features = raw_features.to(device)
        
        # Extract node features and edge indices
        x_dict = {node_type: self.graph[node_type].x for node_type in self.graph.node_types}
        edge_index_dict = {edge_type: self.graph[edge_type].edge_index 
                          for edge_type in self.graph.edge_types}
        
        # Get train/val masks and labels
        train_mask = self.graph['transaction'].train_mask
        val_mask = self.graph['transaction'].val_mask
        y = self.graph['transaction'].y
        
        # Extract merchant IDs and amounts if using self-supervision
        if self.use_self_supervision and not self.use_ensemble:
            # Extract merchant IDs for merchant prediction task
            # Assuming merchant IDs are stored in the graph or can be derived
            # This is a placeholder - adapt to actual data structure
            merchant_ids = torch.zeros_like(y)
            for i, edge_idx in enumerate(self.graph['transaction', 'belongs_to', 'merchant'].edge_index.t()):
                merchant_ids[edge_idx[0]] = edge_idx[1]
            
            # Extract transaction amounts for amount prediction task
            # Assuming amounts are available in raw_features
            # This is a placeholder - adapt to actual data structure
            if raw_features is not None:
                amounts = raw_features[:, 0].unsqueeze(1)  # Assuming first column is amount
            else:
                amounts = torch.zeros(y.size(0), 1, device=device)
        
        # Initialize metrics
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_ensemble:
                logits = self.model(x_dict, edge_index_dict, raw_features[train_mask] if raw_features is not None else None)
                train_loss = torch.nn.functional.cross_entropy(logits, y[train_mask])
            else:
                outputs = self.model(x_dict, edge_index_dict, raw_features)
                
                # Main classification loss
                main_loss = torch.nn.functional.cross_entropy(outputs['logits'][train_mask], y[train_mask])
                
                # Add auxiliary losses if using self-supervision
                if self.use_self_supervision:
                    # Merchant prediction loss
                    merchant_loss = torch.nn.functional.cross_entropy(
                        outputs['merchant_logits'][train_mask],
                        merchant_ids[train_mask]
                    )
                    
                    # Amount prediction loss (mean squared error)
                    amount_loss = torch.nn.functional.mse_loss(
                        outputs['amount_pred'][train_mask],
                        amounts[train_mask]
                    )
                    
                    # Combined loss with weighting
                    train_loss = main_loss + auxiliary_weight * (merchant_loss + amount_loss)
                else:
                    train_loss = main_loss
            
            # Compute accuracy for training set
            if self.use_ensemble:
                train_acc = self._compute_accuracy(logits, y[train_mask])
            else:
                train_acc = self._compute_accuracy(outputs['logits'][train_mask], y[train_mask])
            
            # Backward pass
            train_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                # Forward pass
                if self.use_ensemble:
                    val_logits = self.model(x_dict, edge_index_dict, 
                                          raw_features[val_mask] if raw_features is not None else None)
                    val_loss = torch.nn.functional.cross_entropy(val_logits, y[val_mask])
                    val_acc = self._compute_accuracy(val_logits, y[val_mask])
                else:
                    val_outputs = self.model(x_dict, edge_index_dict, raw_features)
                    val_loss = torch.nn.functional.cross_entropy(val_outputs['logits'][val_mask], y[val_mask])
                    val_acc = self._compute_accuracy(val_outputs['logits'][val_mask], y[val_mask])
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Update metrics
            metrics['train_loss'].append(train_loss.item())
            metrics['train_acc'].append(train_acc)
            metrics['val_loss'].append(val_loss.item())
            metrics['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss.item():.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss.item():.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f"Loaded best model from epoch {best_model_state['epoch']+1} "
                  f"with validation accuracy {best_model_state['val_acc']:.4f}")
        
        return metrics
    
    def evaluate(self, raw_features: torch.Tensor) -> dict:
        """
        Evaluate the trained model on the test set.
        
        Args:
            raw_features: Raw features tensor for tabular model
            
        Returns:
            Dictionary containing test metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized before evaluation")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get device
        device = next(self.model.parameters()).device
        
        # Extract node features and edge indices
        x_dict = {node_type: self.graph[node_type].x for node_type in self.graph.node_types}
        edge_index_dict = {edge_type: self.graph[edge_type].edge_index 
                          for edge_type in self.graph.edge_types}
        
        # Get test mask and labels
        test_mask = self.graph['transaction'].test_mask
        y = self.graph['transaction'].y
        
        # Forward pass
        with torch.no_grad():
            if self.use_ensemble:
                test_logits = self.model(x_dict, edge_index_dict, 
                                       raw_features[test_mask] if raw_features is not None else None)
                test_loss = torch.nn.functional.cross_entropy(test_logits, y[test_mask])
                test_acc = self._compute_accuracy(test_logits, y[test_mask])
                
                # Get predictions
                y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
            else:
                test_outputs = self.model(x_dict, edge_index_dict, raw_features)
                test_loss = torch.nn.functional.cross_entropy(test_outputs['logits'][test_mask], y[test_mask])
                test_acc = self._compute_accuracy(test_outputs['logits'][test_mask], y[test_mask])
                
                # Get predictions
                y_pred = torch.argmax(test_outputs['logits'][test_mask], dim=1).cpu().numpy()
            
            # Get true labels
            y_true = y[test_mask].cpu().numpy()
        
        # Compute additional metrics
        # Note: For 400 categories, computing the full classification report may be too verbose
        # Consider aggregating categories or computing metrics for top categories only
        
        return {
            'test_loss': test_loss.item(),
            'test_acc': test_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def predict(self, transactions_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new transaction data.
        
        Args:
            transactions_df: DataFrame containing new transaction data
            
        Returns:
            Predicted category indices for each transaction
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Build graph from new transaction data
        if self.graph_builder is None:
            self.graph_builder = TransactionGraphBuilder()
        
        new_graph = self.graph_builder.build_graph(transactions_df)
        
        # Add self-loops
        num_nodes = new_graph['transaction'].x.size(0)
        self_indices = torch.arange(num_nodes, dtype=torch.long)
        edge_index = torch.stack([self_indices, self_indices])
        new_graph['transaction', 'self', 'transaction'].edge_index = edge_index
        
        # Add reverse edges
        src_nodes = new_graph['transaction', 'belongs_to', 'merchant'].edge_index[1]
        dst_nodes = new_graph['transaction', 'belongs_to', 'merchant'].edge_index[0]
        new_graph['merchant', 'rev_belongs_to', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        src_nodes = new_graph['transaction', 'has_category', 'category'].edge_index[1]
        dst_nodes = new_graph['transaction', 'has_category', 'category'].edge_index[0]
        new_graph['category', 'rev_has_category', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Extract raw features
        raw_features = self._extract_raw_features(transactions_df)
        
        # Make predictions
        if self.use_ensemble:
            probs = self.model.predict(new_graph, raw_features)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
        else:
            outputs = self.model(
                {node_type: new_graph[node_type].x for node_type in new_graph.node_types},
                {edge_type: new_graph[edge_type].edge_index for edge_type in new_graph.edge_types},
                raw_features
            )
            predictions = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
        
        return predictions
    
    def save_model(self, output_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'model_config': {
                'hidden_channels': self.hidden_channels,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'conv_type': self.conv_type,
                'heads': self.heads,
                'use_jumping_knowledge': self.use_jumping_knowledge,
                'use_batch_norm': self.use_batch_norm,
                'use_self_supervision': self.use_self_supervision,
                'use_tabular_model': self.use_tabular_model,
                'use_graph_transformers': self.use_graph_transformers,
                'use_ensemble': self.use_ensemble,
                'ensemble_size': self.ensemble_size if self.use_ensemble else None
            }
        }, output_path)
    
    def load_model(self, input_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            input_path: Path to the saved model
        """
        # Load model checkpoint
        checkpoint = torch.load(input_path)
        
        # Initialize model with saved configuration
        config = checkpoint['model_config']
        self.hidden_channels = config['hidden_channels']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.conv_type = config['conv_type']
        self.heads = config['heads']
        self.use_jumping_knowledge = config['use_jumping_knowledge']
        self.use_batch_norm = config['use_batch_norm']
        self.use_self_supervision = config['use_self_supervision']
        self.use_tabular_model = config['use_tabular_model']
        self.use_graph_transformers = config['use_graph_transformers']
        self.use_ensemble = config['use_ensemble']
        self.ensemble_size = config['ensemble_size'] if self.use_ensemble else None
        
        # Initialize model
        if self.graph_builder:
            num_categories = len(self.graph_builder.category_mapping)
        else:
            num_categories = 400  # Default
        
        self.initialize_model(num_categories)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _compute_accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute classification accuracy.
        
        Args:
            logits: Model output logits
            y: Ground truth labels
            
        Returns:
            Classification accuracy
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        return correct / total


def plot_training_comparison(base_metrics, enhanced_metrics, hybrid_metrics):
    """
    Plot comparison of training metrics between base, enhanced, and hybrid models.
    
    Args:
        base_metrics: Training metrics for base GNN model
        enhanced_metrics: Training metrics for enhanced GNN model
        hybrid_metrics: Training metrics for hybrid model
    """
    plt.figure(figsize=(16, 8))
    
    # Plot validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(base_metrics['val_acc'], label='Base GNN')
    plt.plot(enhanced_metrics['val_acc'], label='Enhanced GNN')
    plt.plot(hybrid_metrics['val_acc'], label='Hybrid Model')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(base_metrics['val_loss'], label='Base GNN')
    plt.plot(enhanced_metrics['val_loss'], label='Enhanced GNN')
    plt.plot(hybrid_metrics['val_loss'], label='Hybrid Model')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/model_comparison.png')
    plt.close()


def main():
    """
    Main function for training and evaluating the hybrid transaction classifier.
    """
    print("\n=== Advanced Hybrid Transaction Classification ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate synthetic data
    print("\nGenerating synthetic transaction data...")
    num_transactions = 10000
    num_merchants = 200
    num_categories = 400
    
    transactions_df = generate_synthetic_data(
        num_transactions=num_transactions,
        num_merchants=num_merchants,
        num_categories=num_categories
    )
    
    print(f"Generated {len(transactions_df)} transactions with {num_merchants} merchants and {num_categories} categories")
    print("\nSample transactions:")
    print(transactions_df.head())
    
    # Initialize hybrid classifier
    print("\nInitializing hybrid transaction classifier...")
    classifier = HybridTransactionClassifier(
        hidden_channels=128,
        num_layers=3,
        dropout=0.4,
        conv_type='sage',
        heads=2,
        use_jumping_knowledge=True,
        use_batch_norm=True,
        use_self_supervision=True,
        use_tabular_model=True,
        use_graph_transformers=True,
        use_ensemble=True,
        ensemble_size=3,
        lr=0.001,
        weight_decay=1e-4
    )
    
    # Prepare data
    print("\nBuilding transaction graph and extracting features...")
    raw_features = classifier.prepare_data(transactions_df)
    
    # Get actual number of categories from the graph builder
    num_categories = len(classifier.graph_builder.category_mapping)
    print(f"Actual number of categories in the data: {num_categories}")
    
    # Initialize model with the actual number of categories
    classifier.initialize_model(num_categories)
    print(f"Model initialized with {classifier.hidden_channels} hidden channels and {classifier.num_layers} layers")
    
    # Print model information
    print("\nHybrid model architecture:")
    print("- Enhanced GNN with residual connections and jumping knowledge")
    print("- Tabular MLP for direct feature learning")
    print("- Attention-based feature fusion")
    print("- Self-supervised auxiliary tasks")
    print("- Ensemble of specialized models" if classifier.use_ensemble else "- Single model")
    
    # Train model
    print("\nTraining hybrid model (this may take a while)...")
    metrics = classifier.train(raw_features, num_epochs=100, patience=15)
    
    # Plot training metrics
    plot_training_metrics(metrics)
    print("Training metrics plotted to 'plots/training_metrics.png'")
    
    # Evaluate model
    print("\nEvaluating hybrid model on test set...")
    test_metrics = classifier.evaluate(raw_features)
    print(f"Test Loss: {test_metrics['test_loss']:.4f} | Test Accuracy: {test_metrics['test_acc']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/hybrid_transaction_model.pt'
    classifier.save_model(model_path)
    print(f"\nHybrid model saved to {model_path}")
    
    print("\nAdvantages of the hybrid approach:")
    print("1. Leverages both graph structure and raw features for better predictions")
    print("2. Self-supervised learning helps with limited labeled data")
    print("3. Ensemble combines diverse models for improved robustness")
    print("4. Graph transformers capture global dependencies in the transaction graph")
    print("5. Attention mechanism adaptively balances graph and tabular signals")
    
    print("\n=== Hybrid Transaction Classification Complete ===")


if __name__ == '__main__':
    main()