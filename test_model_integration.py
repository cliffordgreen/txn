import os
import sys
import pandas as pd
import numpy as np
import torch

# Add the src directory to the path for imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)
print(f"Added module path: {module_path}")

# Import our project modules
from src.generate_synthetic_data import generate_synthetic_transaction_data
from src.train_with_feedback_data import TransactionFeedbackClassifier

# Generate a small amount of data for testing
print("Generating synthetic data...")
df = generate_synthetic_transaction_data(
    num_transactions=500,
    num_merchants=50,
    num_categories=100,
    output_dir='data',
    num_files=1,
    save_csv=True
)

print("\nData generation complete. Sample data:")
print(df.head(3))
print(f"\nData shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Initialize the classifier
print("\nInitializing model...")
classifier = TransactionFeedbackClassifier(
    hidden_dim=128,
    category_dim=100,  # Will be overridden by actual count in data
    tax_type_dim=10,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
    use_hyperbolic=True,
    use_neural_ode=False,  # Disable for faster testing
    max_seq_length=5,
    lr=0.001,
    weight_decay=1e-5,
    multi_task=True,
    use_text=False  # Disable for simplicity
)

try:
    # Prepare data
    print("\nPreparing transaction data...")
    (transaction_features, seq_features, timestamps, 
     user_features, is_new_user, transaction_descriptions, company_features, t0, t1) = classifier.prepare_data(df)
    
    # Print feature shapes
    print("\nFeature shapes:")
    print(f"Transaction features: {transaction_features.shape}")
    print(f"Sequence features: {seq_features.shape}")
    print(f"Timestamps: {timestamps.shape}")
    if user_features is not None:
        print(f"User features: {user_features.shape}")
    if company_features is not None:
        print(f"Company features: {company_features.shape}")
    
    # Initialize model
    input_dim = transaction_features.size(1)
    company_input_dim = company_features.size(1) if company_features is not None else None
    
    print(f"\nInitializing model with input_dim={input_dim}, company_input_dim={company_input_dim}")
    classifier.initialize_model(input_dim, company_input_dim=company_input_dim)
    
    # Print model info
    num_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Do a single training step to verify everything works
    print("\nPerforming a single training step...")
    train_metrics = classifier.train_step(
        transaction_features, seq_features, timestamps,
        user_features, is_new_user, transaction_descriptions,
        company_features, t0, t1
    )
    
    print("\nTraining step completed successfully!")
    print(f"Training metrics: {train_metrics}")
    
    print("\nIntegration test completed successfully!")
except Exception as e:
    print(f"\nError during model integration test: {str(e)}")
    import traceback
    traceback.print_exc()