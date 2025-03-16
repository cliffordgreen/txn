import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Import our custom modules
from src.data_processing.transaction_graph import TransactionGraphBuilder, create_train_val_test_split
from src.models.gnn_model import TransactionGNN


class TransactionClassifier:
    """
    Class for training and evaluating a GNN-based transaction classifier.
    """
    
    def __init__(self, hidden_channels: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, conv_type: str = 'sage',
                 lr: float = 0.001, weight_decay: float = 5e-4):
        """
        Initialize the transaction classifier.
        
        Args:
            hidden_channels: Dimension of hidden node features
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat')
            lr: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.model = None
        self.optimizer = None
        self.graph = None
        self.graph_builder = None
    
    def prepare_data(self, transactions_df: pd.DataFrame) -> HeteroData:
        """
        Prepare transaction data by building a graph and splitting it into train/val/test sets.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            PyTorch Geometric HeteroData object with train/val/test masks
        """
        # Initialize graph builder
        self.graph_builder = TransactionGraphBuilder()
        
        # Build graph from transaction data
        self.graph = self.graph_builder.build_graph(transactions_df)
        
        # Split graph into train/val/test sets
        self.graph = create_train_val_test_split(self.graph)
        
        return self.graph
    
    def initialize_model(self, num_categories: int = 400) -> None:
        """
        Initialize the GNN model and optimizer.
        
        Args:
            num_categories: Number of transaction categories
        """
        # Initialize model
        self.model = TransactionGNN(
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            conv_type=self.conv_type
        )
        
        # Update classifier to match number of categories
        self.model.classifier = torch.nn.Linear(self.hidden_channels, num_categories)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
    def train(self, num_epochs: int = 100, patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the GNN model on the transaction graph.
        
        Args:
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
        
        # Extract node features and edge indices
        x_dict = {node_type: self.graph[node_type].x for node_type in self.model.metadata[0]}
        edge_index_dict = {edge_type: self.graph[edge_type].edge_index 
                          for edge_type in self.model.metadata[1]}
        
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
            logits = self.model(x_dict, edge_index_dict)
            
            # Compute loss and accuracy for training set
            train_loss = F.cross_entropy(logits[train_mask], y[train_mask])
            train_acc = self._compute_accuracy(logits[train_mask], y[train_mask])
            
            # Backward pass
            train_loss.backward()
            self.optimizer.step()
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                # Forward pass
                logits = self.model(x_dict, edge_index_dict)
                
                # Compute loss and accuracy for validation set
                val_loss = F.cross_entropy(logits[val_mask], y[val_mask])
                val_acc = self._compute_accuracy(logits[val_mask], y[val_mask])
            
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
                      f"Val Acc: {val_acc:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the trained model on the test set.
        
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
        x_dict = {node_type: self.graph[node_type].x for node_type in self.model.metadata[0]}
        edge_index_dict = {edge_type: self.graph[edge_type].edge_index 
                          for edge_type in self.model.metadata[1]}
        
        # Get test mask and labels
        test_mask = self.graph['transaction'].test_mask
        y = self.graph['transaction'].y
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(x_dict, edge_index_dict)
            test_loss = F.cross_entropy(logits[test_mask], y[test_mask])
            test_acc = self._compute_accuracy(logits[test_mask], y[test_mask])
        
        # Compute additional metrics if needed
        # (e.g., precision, recall, F1 score)
        
        return {
            'test_loss': test_loss.item(),
            'test_acc': test_acc
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
        
        # Make predictions
        probs = self.model.predict(new_graph)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        
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
            'model_config': {
                'hidden_channels': self.hidden_channels,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'conv_type': self.conv_type
            }
        }, output_path)
    
    def load_model(self, input_path: str, num_categories: int = 400) -> None:
        """
        Load a trained model from disk.
        
        Args:
            input_path: Path to the saved model
            num_categories: Number of transaction categories
        """
        # Load model checkpoint
        checkpoint = torch.load(input_path)
        
        # Initialize model with saved configuration
        self.hidden_channels = checkpoint['model_config']['hidden_channels']
        self.num_layers = checkpoint['model_config']['num_layers']
        self.dropout = checkpoint['model_config']['dropout']
        self.conv_type = checkpoint['model_config']['conv_type']
        
        # Initialize model
        self.initialize_model(num_categories)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
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


def main():
    """
    Main function for training and evaluating the transaction classifier.
    """
    # Load transaction data
    # This is a placeholder - replace with actual data loading code
    # transactions_df = pd.read_csv('path/to/transactions.csv')
    
    # For demonstration, create a synthetic dataset
    np.random.seed(42)
    num_transactions = 1000
    num_merchants = 50
    num_categories = 10  # Using 10 categories for demonstration
    
    transactions_df = pd.DataFrame({
        'transaction_id': range(num_transactions),
        'merchant_id': np.random.randint(0, num_merchants, num_transactions),
        'category_id': np.random.randint(0, num_categories, num_transactions),
        'amount': np.random.uniform(1, 1000, num_transactions),
        'timestamp': np.random.randint(1577836800, 1609459200, num_transactions),  # 2020 timestamps
        'is_online': np.random.choice([0, 1], num_transactions),
        'is_international': np.random.choice([0, 1], num_transactions),
    })
    
    # Initialize classifier
    classifier = TransactionClassifier(
        hidden_channels=64,
        num_layers=2,
        dropout=0.3,
        conv_type='sage',
        lr=0.001,
        weight_decay=5e-4
    )
    
    # Prepare data
    graph = classifier.prepare_data(transactions_df)
    
    # Initialize model with the actual number of categories
    num_categories = len(classifier.graph_builder.category_mapping)
    classifier.initialize_model(num_categories)
    
    # Train model
    metrics = classifier.train(num_epochs=50, patience=10)
    
    # Evaluate model
    test_metrics = classifier.evaluate()
    print(f"Test Loss: {test_metrics['test_loss']:.4f} | Test Acc: {test_metrics['test_acc']:.4f}")
    
    # Save model
    classifier.save_model('models/transaction_gnn.pt')


if __name__ == '__main__':
    main()