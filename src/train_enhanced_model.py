import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Import our custom modules
from src.models.train import TransactionClassifier
from src.data_processing.transaction_graph import TransactionGraphBuilder, create_train_val_test_split
from src.models.enhanced_gnn_model import EnhancedTransactionGNN
from src.train_transaction_classifier import generate_synthetic_data, plot_training_metrics


class EnhancedTransactionClassifier(TransactionClassifier):
    """
    Class for training and evaluating an enhanced GNN-based transaction classifier.
    Extends the base TransactionClassifier with improved model architecture.
    """
    
    def __init__(self, hidden_channels: int = 128, num_layers: int = 3, 
                 dropout: float = 0.4, conv_type: str = 'sage', heads: int = 2,
                 use_jumping_knowledge: bool = True, use_batch_norm: bool = True,
                 lr: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize the enhanced transaction classifier.
        
        Args:
            hidden_channels: Dimension of hidden node features
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat')
            heads: Number of attention heads for GAT
            use_jumping_knowledge: Whether to use jumping knowledge
            use_batch_norm: Whether to use batch normalization
            lr: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        # Initialize parent class
        super().__init__(hidden_channels, num_layers, dropout, conv_type, lr, weight_decay)
        
        # Additional parameters for enhanced model
        self.heads = heads
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_batch_norm = use_batch_norm
    
    def initialize_model(self, num_categories: int = 400) -> None:
        """
        Initialize the enhanced GNN model and optimizer.
        
        Args:
            num_categories: Number of transaction categories
        """
        # Initialize enhanced model
        self.model = EnhancedTransactionGNN(
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            conv_type=self.conv_type,
            heads=self.heads,
            use_jumping_knowledge=self.use_jumping_knowledge,
            use_batch_norm=self.use_batch_norm
        )
        
        # Update classifier to match number of categories
        self.model.classifier = torch.nn.Linear(self.hidden_channels, num_categories)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )


def main():
    """
    Main function for training and evaluating the enhanced transaction classifier.
    """
    print("\n=== Enhanced Transaction Classification using Graph Neural Networks ===")
    
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
    
    # Initialize enhanced classifier with optimal hyperparameters
    print("\nInitializing enhanced transaction classifier...")
    classifier = EnhancedTransactionClassifier(
        hidden_channels=128,
        num_layers=3,
        dropout=0.4,
        conv_type='sage',
        heads=2,
        use_jumping_knowledge=True,
        use_batch_norm=True,
        lr=0.001,
        weight_decay=1e-4
    )
    
    # Prepare data
    print("\nBuilding transaction graph...")
    graph_builder = TransactionGraphBuilder(num_categories=num_categories)
    graph = graph_builder.build_graph(transactions_df)
    
    # Add reverse edges for better message passing
    # Transaction -> Merchant becomes Merchant -> Transaction as well
    # This improves the model's ability to leverage graph structure
    src_nodes = graph['transaction', 'belongs_to', 'merchant'].edge_index[1]
    dst_nodes = graph['transaction', 'belongs_to', 'merchant'].edge_index[0]
    graph['merchant', 'rev_belongs_to', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
    
    # Transaction -> Category becomes Category -> Transaction as well
    src_nodes = graph['transaction', 'has_category', 'category'].edge_index[1]
    dst_nodes = graph['transaction', 'has_category', 'category'].edge_index[0]
    graph['category', 'rev_has_category', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
    
    # Split the graph
    graph = create_train_val_test_split(graph, train_ratio=0.7, val_ratio=0.15)
    
    # Set the graph in classifier
    classifier.graph = graph
    classifier.graph_builder = graph_builder
    
    # Get actual number of categories from the graph builder
    num_categories = len(graph_builder.category_mapping)
    print(f"Actual number of categories in the data: {num_categories}")
    
    # Initialize model with the actual number of categories
    classifier.initialize_model(num_categories)
    print(f"Model initialized with {classifier.model.hidden_channels} hidden channels and {classifier.model.num_layers} layers")
    
    # Train model
    print("\nTraining enhanced model...")
    metrics = classifier.train(num_epochs=100, patience=15)
    
    # Plot training metrics
    plot_training_metrics(metrics)
    print("Training metrics plotted to 'plots/training_metrics.png'")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_metrics = classifier.evaluate()
    print(f"Test Loss: {test_metrics['test_loss']:.4f} | Test Accuracy: {test_metrics['test_acc']:.4f}")
    
    # Get predictions for test set to analyze performance
    classifier.model.eval()
    with torch.no_grad():
        # Extract node features and edge indices
        x_dict = {node_type: graph[node_type].x for node_type in classifier.model.metadata[0]}
        edge_index_dict = {edge_type: graph[edge_type].edge_index 
                          for edge_type in classifier.model.metadata[1] if edge_type in graph}
        
        # Forward pass
        logits = classifier.model(x_dict, edge_index_dict)
        
        # Get test mask and labels
        test_mask = graph['transaction'].test_mask
        y_true = graph['transaction'].y[test_mask].cpu().numpy()
        
        # Get predictions
        probs = torch.softmax(logits[test_mask], dim=1)
        y_pred = torch.argmax(probs, dim=1).cpu().numpy()
    
    # Compare with baseline model
    print("\nModel architecture comparison:")
    print("Enhanced model includes:")
    print("- Residual connections between layers")
    print("- Batch normalization for improved training stability")
    print("- Jumping knowledge for better information flow across layers")
    print("- Bidirectional message passing (reverse edges)")
    
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/enhanced_transaction_gnn.pt'
    classifier.save_model(model_path)
    print(f"\nEnhanced model saved to {model_path}")
    
    print("\n=== Enhanced Transaction Classification Complete ===")


if __name__ == '__main__':
    main()