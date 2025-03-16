import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Import our custom modules
from src.models.train import TransactionClassifier
from src.data_processing.transaction_graph import TransactionGraphBuilder


def generate_synthetic_data(num_transactions=10000, num_merchants=200, num_categories=400):
    """
    Generate synthetic transaction data for demonstration purposes.
    
    Args:
        num_transactions: Number of transactions to generate
        num_merchants: Number of unique merchants
        num_categories: Number of transaction categories
        
    Returns:
        DataFrame containing synthetic transaction data
    """
    np.random.seed(42)
    
    # Generate transaction IDs
    transaction_ids = range(num_transactions)
    
    # Generate merchant IDs with realistic distribution (some merchants appear more frequently)
    merchant_popularity = np.random.exponential(scale=0.1, size=num_merchants)
    merchant_popularity = merchant_popularity / merchant_popularity.sum()
    merchant_ids = np.random.choice(
        range(num_merchants), 
        size=num_transactions, 
        p=merchant_popularity
    )
    
    # Generate category IDs with realistic distribution
    # Some categories are more common than others
    category_popularity = np.random.exponential(scale=0.05, size=num_categories)
    category_popularity = category_popularity / category_popularity.sum()
    category_ids = np.random.choice(
        range(num_categories), 
        size=num_transactions, 
        p=category_popularity
    )
    
    # Generate transaction amounts with realistic distribution
    amounts = np.random.exponential(scale=50, size=num_transactions)
    
    # Generate timestamps over a one-year period
    start_timestamp = 1577836800  # Jan 1, 2020
    end_timestamp = 1609459200    # Dec 31, 2020
    timestamps = np.random.randint(start_timestamp, end_timestamp, num_transactions)
    
    # Generate binary features
    is_online = np.random.choice([0, 1], num_transactions, p=[0.7, 0.3])
    is_international = np.random.choice([0, 1], num_transactions, p=[0.9, 0.1])
    
    # Create merchant-category relationships
    # Each merchant tends to have transactions in specific categories
    merchant_category_affinity = {}
    for m in range(num_merchants):
        # Each merchant has high affinity for 1-5 categories
        num_preferred_categories = np.random.randint(1, 6)
        preferred_categories = np.random.choice(range(num_categories), num_preferred_categories, replace=False)
        merchant_category_affinity[m] = preferred_categories
    
    # Adjust some categories based on merchant affinity (80% of the time)
    for i in range(num_transactions):
        merchant = merchant_ids[i]
        if np.random.random() < 0.8 and len(merchant_category_affinity[merchant]) > 0:
            category_ids[i] = np.random.choice(merchant_category_affinity[merchant])
    
    # Create DataFrame
    transactions_df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'merchant_id': merchant_ids,
        'category_id': category_ids,
        'amount': amounts,
        'timestamp': timestamps,
        'is_online': is_online,
        'is_international': is_international,
    })
    
    # Add some derived features
    transactions_df['day_of_week'] = pd.to_datetime(transactions_df['timestamp'], unit='s').dt.dayofweek
    transactions_df['hour_of_day'] = pd.to_datetime(transactions_df['timestamp'], unit='s').dt.hour
    
    return transactions_df


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
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_metrics.png')
    plt.close()


def main():
    """
    Main function for training and evaluating the transaction classifier.
    """
    print("\n=== Transaction Classification using Graph Neural Networks ===")
    
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
    
    # Initialize classifier with optimal hyperparameters
    print("\nInitializing transaction classifier...")
    classifier = TransactionClassifier(
        hidden_channels=128,  # Increased from 64 for more expressive power
        num_layers=3,         # Increased from 2 to capture more complex patterns
        dropout=0.4,          # Increased from 0.3 to prevent overfitting
        conv_type='sage',     # GraphSAGE performs well on heterogeneous graphs
        lr=0.001,
        weight_decay=1e-4     # Reduced from 5e-4 for less regularization
    )
    
    # Prepare data
    print("\nBuilding transaction graph...")
    graph = classifier.prepare_data(transactions_df)
    
    # Get actual number of categories from the graph builder
    num_categories = len(classifier.graph_builder.category_mapping)
    print(f"Actual number of categories in the data: {num_categories}")
    
    # Initialize model with the actual number of categories
    classifier.initialize_model(num_categories)
    print(f"Model initialized with {classifier.model.hidden_channels} hidden channels and {classifier.model.num_layers} layers")
    
    # Train model
    print("\nTraining model...")
    metrics = classifier.train(num_epochs=100, patience=15)
    
    # Plot training metrics
    plot_training_metrics(metrics)
    print("Training metrics plotted to 'plots/training_metrics.png'")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_metrics = classifier.evaluate()
    print(f"Test Loss: {test_metrics['test_loss']:.4f} | Test Accuracy: {test_metrics['test_acc']:.4f}")
    
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/transaction_gnn.pt'
    classifier.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n=== Transaction Classification Complete ===")


if __name__ == '__main__':
    main()