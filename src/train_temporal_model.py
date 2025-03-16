import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import our custom modules
from src.data_processing.transaction_graph import TransactionGraphBuilder, create_train_val_test_split
from src.models.temporal_transaction_model import (
    AdvancedTemporalTransactionGNN, 
    TemporalTransactionEnsemble,
    TemporalTransactionEncoder
)
from src.train_transaction_classifier import generate_synthetic_data


class UserTransactionSequenceBuilder:
    """
    Class for building user transaction sequences for sequential modeling.
    """
    
    def __init__(self, max_seq_length: int = 20):
        """
        Initialize the sequence builder.
        
        Args:
            max_seq_length: Maximum sequence length
        """
        self.max_seq_length = max_seq_length
    
    def build_sequences(self, transactions_df: pd.DataFrame) -> tuple:
        """
        Build user transaction sequences from transaction data.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Tuple of (sequences, lengths, timestamps, user_ids)
        """
        # Make a copy to avoid modifying the original dataframe
        df = transactions_df.copy()
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add a user ID column if it doesn't exist
        # In real data, there would be a user ID for each transaction
        # For synthetic data, we'll create random user IDs
        if 'user_id' not in df.columns:
            num_users = max(100, len(df) // 20)  # Average 20 transactions per user
            df['user_id'] = np.random.randint(0, num_users, len(df))
        
        # Group transactions by user
        user_groups = df.groupby('user_id')
        
        # Initialize lists for sequences, lengths, and timestamps
        sequences = []
        seq_lengths = []
        seq_timestamps = []
        user_ids = []
        
        # Extract features for sequences
        feature_cols = [col for col in df.columns 
                       if df[col].dtype in ['int64', 'float64'] 
                       and col not in ['user_id', 'transaction_id']]
        
        # Add engineered features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['log_amount'] = np.log1p(df['amount'])
        
        # Update feature columns with engineered features
        feature_cols = [col for col in df.columns 
                       if df[col].dtype in ['int64', 'float64'] 
                       and col not in ['user_id', 'transaction_id']]
        
        # Build sequences for each user
        for user_id, group in user_groups:
            # Get user transactions
            user_transactions = group[feature_cols].values
            user_timestamps = group['timestamp'].values
            user_categories = group['category_id'].values
            
            # Handle users with more than max_seq_length transactions
            if len(user_transactions) > self.max_seq_length:
                # Use sliding window to create multiple sequences
                for i in range(0, len(user_transactions) - self.max_seq_length + 1, self.max_seq_length // 2):
                    end_idx = i + self.max_seq_length
                    seq = user_transactions[i:end_idx]
                    timestamps = user_timestamps[i:end_idx]
                    categories = user_categories[i:end_idx]
                    
                    sequences.append(seq)
                    seq_lengths.append(len(seq))
                    seq_timestamps.append(timestamps)
                    user_ids.append(user_id)
            else:
                # Pad shorter sequences
                padded_seq = np.zeros((self.max_seq_length, len(feature_cols)))
                padded_timestamps = np.zeros(self.max_seq_length)
                
                seq_len = len(user_transactions)
                padded_seq[:seq_len] = user_transactions
                padded_timestamps[:seq_len] = user_timestamps
                
                sequences.append(padded_seq)
                seq_lengths.append(seq_len)
                seq_timestamps.append(padded_timestamps)
                user_ids.append(user_id)
        
        # Convert to tensors
        sequences = torch.FloatTensor(np.array(sequences))
        seq_lengths = torch.LongTensor(seq_lengths)
        seq_timestamps = torch.FloatTensor(np.array(seq_timestamps))
        user_ids = torch.LongTensor(user_ids)
        
        return sequences, seq_lengths, seq_timestamps, user_ids


class TemporalTransactionClassifier:
    """
    Class for training and evaluating a temporal transaction classifier
    that combines graph structure with sequential patterns.
    """
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, output_dim: int = 400,
                 num_rnn_layers: int = 2, num_gnn_layers: int = 3, dropout: float = 0.3,
                 use_attention: bool = True, bidirectional: bool = True, 
                 gnn_type: str = 'gated', use_ensemble: bool = True,
                 ensemble_size: int = 3, lr: float = 0.001, 
                 weight_decay: float = 1e-4, max_seq_length: int = 20):
        """
        Initialize the temporal transaction classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            output_dim: Dimension of output features (num classes)
            num_rnn_layers: Number of RNN layers
            num_gnn_layers: Number of GNN layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            bidirectional: Whether to use bidirectional RNN
            gnn_type: Type of GNN ('gated', 'rgcn', 'gat')
            use_ensemble: Whether to use ensemble of models
            ensemble_size: Number of models in the ensemble
            lr: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_seq_length: Maximum sequence length
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.gnn_type = gnn_type
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.graph = None
        self.graph_builder = None
        self.sequence_builder = UserTransactionSequenceBuilder(max_seq_length=max_seq_length)
    
    def prepare_data(self, transactions_df: pd.DataFrame) -> tuple:
        """
        Prepare transaction data by building a graph and extracting sequences.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Tuple of (graph, sequences, lengths, timestamps, edge_time, node_time)
        """
        # Initialize graph builder
        self.graph_builder = TransactionGraphBuilder()
        
        # Build graph from transaction data
        self.graph = self.graph_builder.build_graph(transactions_df)
        
        # Add self-loops for better information propagation
        for node_type in self.graph.node_types:
            num_nodes = self.graph[node_type].x.size(0)
            self_indices = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([self_indices, self_indices])
            self.graph[f"{node_type}_self_loop"] = edge_index
        
        # Add reverse edges for bidirectional message passing
        src_nodes = self.graph['transaction', 'belongs_to', 'merchant'].edge_index[1]
        dst_nodes = self.graph['transaction', 'belongs_to', 'merchant'].edge_index[0]
        self.graph['merchant', 'rev_belongs_to', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        src_nodes = self.graph['transaction', 'has_category', 'category'].edge_index[1]
        dst_nodes = self.graph['transaction', 'has_category', 'category'].edge_index[0]
        self.graph['category', 'rev_has_category', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Split graph into train/val/test sets
        self.graph = create_train_val_test_split(self.graph)
        
        # Build user transaction sequences
        sequences, seq_lengths, seq_timestamps, user_ids = \
            self.sequence_builder.build_sequences(transactions_df)
        
        # Create edge timestamps
        # In a real-world scenario, these would come from the transaction data
        # For simplicity, we'll use random timestamps for edges
        num_edges = 0
        for edge_type in self.graph.edge_types:
            num_edges += self.graph[edge_type].edge_index.size(1)
        
        edge_time = torch.FloatTensor(
            np.random.uniform(
                transactions_df['timestamp'].min(),
                transactions_df['timestamp'].max(),
                num_edges
            )
        )
        
        # Create node timestamps
        # For transaction nodes, we'll use the actual transaction timestamps
        # For other nodes, we'll use the median of connected transaction timestamps
        node_time = {}
        for node_type in self.graph.node_types:
            num_nodes = self.graph[node_type].x.size(0)
            if node_type == 'transaction':
                # Use actual transaction timestamps
                node_time[node_type] = torch.FloatTensor(transactions_df['timestamp'].values)
            else:
                # Use median of connected transaction timestamps
                node_time[node_type] = torch.zeros(num_nodes)
        
        # Combine node timestamps
        all_node_time = torch.cat([node_time[node_type] for node_type in self.graph.node_types])
        
        return self.graph, sequences, seq_lengths, seq_timestamps, edge_time, all_node_time
    
    def initialize_model(self, input_dim: int, output_dim: int = 400) -> None:
        """
        Initialize the temporal transaction model and optimizer.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features (num classes)
        """
        # Initialize model
        if self.use_ensemble:
            self.model = TemporalTransactionEnsemble(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                num_models=self.ensemble_size,
                dropout=self.dropout
            )
        else:
            self.model = AdvancedTemporalTransactionGNN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=output_dim,
                num_rnn_layers=self.num_rnn_layers,
                num_gnn_layers=self.num_gnn_layers,
                dropout=self.dropout,
                use_attention=self.use_attention,
                bidirectional=self.bidirectional,
                gnn_type=self.gnn_type
            )
        
        # Initialize optimizer with weight decay and gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Initialize learning rate scheduler with cosine annealing and warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart period after each restart
            eta_min=1e-6  # Minimum learning rate
        )
    
    def train(self, sequences: torch.Tensor, seq_lengths: torch.Tensor, 
              seq_timestamps: torch.Tensor, edge_time: torch.Tensor, 
              node_time: torch.Tensor, num_epochs: int = 100, 
              patience: int = 15) -> dict:
        """
        Train the temporal transaction model.
        
        Args:
            sequences: Transaction sequences [batch_size, seq_len, input_dim]
            seq_lengths: Sequence lengths [batch_size]
            seq_timestamps: Sequence timestamps [batch_size, seq_len]
            edge_time: Edge timestamps [num_edges]
            node_time: Node timestamps [num_nodes]
            num_epochs: Maximum number of training epochs
            patience: Number of epochs to wait for validation improvement before early stopping
            
        Returns:
            Dictionary containing training and validation metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized before training")
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.graph = self.graph.to(device)
        sequences = sequences.to(device)
        seq_lengths = seq_lengths.to(device)
        seq_timestamps = seq_timestamps.to(device)
        edge_time = edge_time.to(device)
        node_time = node_time.to(device)
        
        # Extract node features
        x = torch.cat([self.graph[node_type].x for node_type in self.graph.node_types])
        
        # Extract edge indices
        edge_index_list = []
        for edge_type in self.graph.edge_types:
            edge_index_list.append(self.graph[edge_type].edge_index)
        
        # Combine edge indices
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            # Create empty edge index if no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # Get train/val masks and labels
        train_mask = self.graph['transaction'].train_mask
        val_mask = self.graph['transaction'].val_mask
        y = self.graph['transaction'].y
        
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
            logits = self.model(
                x, edge_index, sequences, seq_lengths, 
                seq_timestamps, edge_time, node_time
            )
            
            # Compute loss and accuracy for training set
            train_loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])
            train_acc = self._compute_accuracy(logits[train_mask], y[train_mask])
            
            # Backward pass
            train_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Update learning rate scheduler
            self.scheduler.step(epoch)
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                # Forward pass
                val_logits = self.model(
                    x, edge_index, sequences, seq_lengths, 
                    seq_timestamps, edge_time, node_time
                )
                
                # Compute loss and accuracy for validation set
                val_loss = torch.nn.functional.cross_entropy(val_logits[val_mask], y[val_mask])
                val_acc = self._compute_accuracy(val_logits[val_mask], y[val_mask])
            
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
    
    def evaluate(self, sequences: torch.Tensor, seq_lengths: torch.Tensor, 
                 seq_timestamps: torch.Tensor, edge_time: torch.Tensor, 
                 node_time: torch.Tensor) -> dict:
        """
        Evaluate the trained model on the test set.
        
        Args:
            sequences: Transaction sequences [batch_size, seq_len, input_dim]
            seq_lengths: Sequence lengths [batch_size]
            seq_timestamps: Sequence timestamps [batch_size, seq_len]
            edge_time: Edge timestamps [num_edges]
            node_time: Node timestamps [num_nodes]
            
        Returns:
            Dictionary containing test metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized before evaluation")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get device
        device = next(self.model.parameters()).device
        
        # Extract node features
        x = torch.cat([self.graph[node_type].x for node_type in self.graph.node_types])
        
        # Extract edge indices
        edge_index_list = []
        for edge_type in self.graph.edge_types:
            edge_index_list.append(self.graph[edge_type].edge_index)
        
        # Combine edge indices
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            # Create empty edge index if no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # Get test mask and labels
        test_mask = self.graph['transaction'].test_mask
        y = self.graph['transaction'].y
        
        # Forward pass
        with torch.no_grad():
            test_logits = self.model(
                x, edge_index, sequences, seq_lengths, 
                seq_timestamps, edge_time, node_time
            )
            
            # Compute loss and accuracy for test set
            test_loss = torch.nn.functional.cross_entropy(test_logits[test_mask], y[test_mask])
            test_acc = self._compute_accuracy(test_logits[test_mask], y[test_mask])
            
            # Get predictions
            y_pred = torch.argmax(test_logits[test_mask], dim=1).cpu().numpy()
            y_true = y[test_mask].cpu().numpy()
        
        return {
            'test_loss': test_loss.item(),
            'test_acc': test_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
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
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_rnn_layers': self.num_rnn_layers,
                'num_gnn_layers': self.num_gnn_layers,
                'dropout': self.dropout,
                'use_attention': self.use_attention,
                'bidirectional': self.bidirectional,
                'gnn_type': self.gnn_type,
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
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_rnn_layers = config['num_rnn_layers']
        self.num_gnn_layers = config['num_gnn_layers']
        self.dropout = config['dropout']
        self.use_attention = config['use_attention']
        self.bidirectional = config['bidirectional']
        self.gnn_type = config['gnn_type']
        self.use_ensemble = config['use_ensemble']
        self.ensemble_size = config['ensemble_size'] if self.use_ensemble else None
        
        # Initialize model
        self.initialize_model(self.input_dim, self.output_dim)
        
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


def plot_training_metrics(metrics):
    """
    Plot training and validation metrics.
    
    Args:
        metrics: Dictionary containing training and validation metrics
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/temporal_training_metrics.png')
    plt.close()


def main():
    """
    Main function for training and evaluating the temporal transaction classifier.
    """
    print("\n=== Advanced Temporal Transaction Classification ===")
    
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
    
    # Initialize temporal classifier
    print("\nInitializing temporal transaction classifier...")
    classifier = TemporalTransactionClassifier(
        hidden_dim=128,
        num_rnn_layers=2,
        num_gnn_layers=3,
        dropout=0.3,
        use_attention=True,
        bidirectional=True,
        gnn_type='gated',
        use_ensemble=True,
        ensemble_size=3,
        lr=0.001,
        weight_decay=1e-4,
        max_seq_length=20
    )
    
    # Prepare data
    print("\nBuilding transaction graph and sequences...")
    graph, sequences, seq_lengths, seq_timestamps, edge_time, node_time = \
        classifier.prepare_data(transactions_df)
    
    # Get input dimension from sequences
    input_dim = sequences.size(2)
    
    # Get actual number of categories from the graph builder
    num_categories = len(classifier.graph_builder.category_mapping)
    print(f"Actual number of categories in the data: {num_categories}")
    
    # Initialize model with the actual dimensions
    classifier.initialize_model(input_dim, num_categories)
    print(f"Model initialized with {input_dim} input features, {classifier.hidden_dim} hidden features")
    
    # Print model information
    print("\nTemporal Model Architecture:")
    print("- Advanced Temporal GNN with time-aware graph attention")
    print("- Recurrent neural networks for sequence modeling")
    print("- Transformers for capturing global dependencies")
    print("- Time encoding for temporal features")
    print("- Ensemble approach for improved robustness" if classifier.use_ensemble else "- Single model")
    
    # Train model
    print("\nTraining temporal model (this may take a while)...")
    metrics = classifier.train(
        sequences, seq_lengths, seq_timestamps, edge_time, node_time,
        num_epochs=100, patience=15
    )
    
    # Plot training metrics
    plot_training_metrics(metrics)
    print("Training metrics plotted to 'plots/temporal_training_metrics.png'")
    
    # Evaluate model
    print("\nEvaluating temporal model on test set...")
    test_metrics = classifier.evaluate(
        sequences, seq_lengths, seq_timestamps, edge_time, node_time
    )
    print(f"Test Loss: {test_metrics['test_loss']:.4f} | Test Accuracy: {test_metrics['test_acc']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/temporal_transaction_model.pt'
    classifier.save_model(model_path)
    print(f"\nTemporal model saved to {model_path}")
    
    print("\nAdvantages of the temporal approach:")
    print("1. Captures sequential patterns and temporal dependencies in user transactions")
    print("2. Models transaction timing and inter-transaction relationships")
    print("3. Combines graph structure with temporal dynamics")
    print("4. Uses sophisticated time encoding for better feature representation")
    print("5. Employs attention mechanisms to focus on relevant transactions")
    
    print("\n=== Temporal Transaction Classification Complete ===")


if __name__ == '__main__':
    main()