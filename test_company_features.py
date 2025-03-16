import pandas as pd
import numpy as np
import torch
from src.train_with_feedback_data import TransactionFeedbackClassifier, load_transaction_feedback_data

# Create synthetic data with company features
num_transactions = 100
num_users = 5
num_categories = 10 
num_tax_types = 5

# Create synthetic transaction data with company features
df = pd.DataFrame({
    'user_id': np.random.randint(0, num_users, num_transactions),
    'txn_id': [f"txn_{i}" for i in range(num_transactions)],
    'is_new_user': np.random.choice([0, 1], num_transactions, p=[0.9, 0.1]),
    'presented_category_id': np.random.randint(0, num_categories, num_transactions),
    'presented_tax_account_type': np.random.randint(0, num_tax_types, num_transactions),
    'conf_score': np.random.uniform(0.5, 1.0, num_transactions),
    'amount': np.random.uniform(10.0, 1000.0, num_transactions),
    'merchant_id': np.random.randint(0, 20, num_transactions),
    'company_type': np.random.choice(['LLC', 'Corporation', 'Sole Proprietorship'], num_transactions),
    'company_size': np.random.choice(['Small', 'Medium', 'Large'], num_transactions)
})

print("Created test dataset with the following company feature columns:")
print(f"- company_type: {df['company_type'].value_counts().to_dict()}")
print(f"- company_size: {df['company_size'].value_counts().to_dict()}")

# Initialize classifier
classifier = TransactionFeedbackClassifier(
    hidden_dim=128,
    category_dim=num_categories,
    tax_type_dim=num_tax_types,
    num_heads=2,
    num_layers=1,
    dropout=0.1,
    use_hyperbolic=True,
    use_neural_ode=False,
    use_ensemble=False,
    multi_task=True,
    use_text=False
)

# Prepare data
print("\nPreparing data with company features...")
data_tuple = classifier.prepare_data(df)
transaction_features, seq_features, timestamps, user_features, is_new_user, transaction_descriptions, company_features, t0, t1 = data_tuple

# Print company features if available
if company_features is not None:
    print(f"Company features detected with shape: {company_features.shape}")
    print(f"Sample company feature tensor: {company_features[0][:5]}...")
else:
    print("No company features detected in the graph")

# Initialize model
input_dim = transaction_features.size(1)
company_input_dim = company_features.size(1) if company_features is not None else None
classifier.initialize_model(input_dim, graph_input_dim=input_dim, company_input_dim=company_features.size(1) if company_features is not None else None)
print(f"\nModel initialized with {input_dim} input features")

# Print model information
num_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
print(f"Model has {num_params:,} trainable parameters")

# Try a forward pass
print("\nTesting model forward pass with company features...")
# Run the model in eval mode with a small batch
classifier.model.eval()

# Create a specialized dim_alignment layer for the test
if company_features is not None:
    if not hasattr(classifier.model, 'dim_alignment') or 'company' not in classifier.model.dim_alignment:
        company_align = torch.nn.Linear(company_features.size(1), input_dim).to(transaction_features.device)
        classifier.model.dim_alignment['company'] = company_align

with torch.no_grad():
    output = classifier.model(
        transaction_features[:5], 
        seq_features[:5], 
        transaction_features[:5],
        timestamps[:5], 
        t0, t1, 
        None,  # No descriptions
        user_features=user_features[:5] if user_features is not None else None,
        is_new_user=is_new_user[:5] if is_new_user is not None else None,
        company_features=company_features[:5] if company_features is not None else None
    )
    
    if isinstance(output, tuple):
        category_logits, tax_type_logits = output
        print(f"Multi-task output received")
        print(f"Category logits shape: {category_logits.shape}")
        print(f"Tax type logits shape: {tax_type_logits.shape}")
    else:
        print(f"Single-task output shape: {output.shape}")

print("Test completed successfully\!")
