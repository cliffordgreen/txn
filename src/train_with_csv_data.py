import os
import sys
import pandas as pd
import torch

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_hyper_temporal_model import HyperTemporalTransactionClassifier, plot_training_metrics

def main():
    """
    Main function for training and evaluating the hyper-temporal transaction classifier
    using data from a CSV file.
    """
    print("\n=== Training Hyper-Temporal Transaction Model with CSV Data ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load transaction data from CSV
    print("\nLoading transaction data from CSV...")
    csv_path = 'data/synthetic_transactions.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    transactions_df = pd.read_csv(csv_path)
    print(f"Loaded {len(transactions_df)} transactions")
    print("\nSample transactions with descriptions:")
    print(transactions_df[['transaction_id', 'merchant_id', 'category_id', 'amount', 'transaction_description']].head())
    
    # Initialize a simplified classifier for quick testing
    print("\nInitializing simplified classifier for demonstration purposes...")
    classifier = HyperTemporalTransactionClassifier(
        hidden_dim=64,    # Small size for faster training
        num_heads=2,      # Small size for faster training 
        num_layers=1,     # Minimum layers for faster training
        dropout=0.1,
        use_hyperbolic=False,  # Disable hyperbolic for simpler model
        use_neural_ode=False,  # Disable neural ODE for simpler model
        use_ensemble=False,    # Disable ensemble for faster training
        max_seq_length=10,     # Shorter sequences for faster training
        lr=1e-3,
        weight_decay=0
    )
    
    # Prepare data
    print("\nBuilding transaction graph and sequences...")
    (graph, sequences, seq_lengths, seq_timestamps, 
     edge_time, node_time, time_features, t0, t1) = classifier.prepare_data(transactions_df)
    
    # Extract transaction descriptions for processing
    transaction_descriptions = transactions_df['transaction_description'].tolist()
    
    # Get dimensions from our data structures
    seq_dim = sequences.size(2)  # Sequence feature dimension
    graph_dim = classifier.graph['transaction'].x.size(1)  # Graph feature dimension
    num_categories = len(classifier.graph_builder.category_mapping)  # Number of categories
    
    print(f"Sequence feature dimension: {seq_dim}")
    print(f"Graph feature dimension: {graph_dim}")
    print(f"Number of categories: {num_categories}")
    
    # Create separate models for graph and sequence features
    # We will unify both in a future version but for this quick test, we'll just use the sequence dimension
    # for both to make training work
    classifier.initialize_model(seq_dim)
    print(f"Model initialized with {seq_dim} input features, {classifier.hidden_dim} hidden features")
    
    # Train model with small dataset and epochs to showcase functionality
    print("\nTraining hyper-temporal model with advanced text processing...")
    print("Note: This will download pre-trained transformer models and may take some time for first run")
    
    # Use subset of data for quicker execution
    subset_size = min(1000, len(transaction_descriptions))
    transaction_descriptions_subset = transaction_descriptions[:subset_size]
    
    metrics = classifier.train(
        sequences, seq_lengths, seq_timestamps, edge_time, node_time, time_features, t0, t1,
        transaction_descriptions=transaction_descriptions_subset,
        num_epochs=3,  # Very small number of epochs for demonstration
        patience=2,    # Reduced for testing
        warmup_pct=0.1
    )
    
    # Plot training metrics
    plot_training_metrics(metrics)
    print("Training metrics plotted to 'plots/hyper_temporal_training_metrics.png'")
    
    # Evaluate model
    print("\nEvaluating hyper-temporal model on test set...")
    test_metrics = classifier.evaluate(
        sequences, seq_lengths, seq_timestamps, edge_time, node_time, time_features, t0, t1,
        transaction_descriptions=transaction_descriptions_subset  # Use same subset for evaluation
    )
    print(f"Test Loss: {test_metrics['test_loss']:.4f} | Test Accuracy: {test_metrics['test_acc']:.4f}")
    print(f"Top-3 Accuracy: {test_metrics['test_top3_acc']:.4f} | Top-5 Accuracy: {test_metrics['test_top5_acc']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/hyper_temporal_transaction_model_from_csv.pt'
    classifier.save_model(model_path)
    print(f"\nHyper-temporal model with text processing saved to {model_path}")
    
    print("\n=== Training Complete ===")

if __name__ == '__main__':
    main()