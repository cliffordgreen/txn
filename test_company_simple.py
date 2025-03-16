import pandas as pd
import numpy as np
import torch
from src.models.hyper_temporal_model import HyperTemporalTransactionModel, CompanyAwareContextLayer

print("Testing company features integration:")

# Create synthetic company features
batch_size = 5
company_input_dim = 6
hidden_dim = 64

# Create model with company feature support
model = HyperTemporalTransactionModel(
    input_dim=32,
    hidden_dim=hidden_dim,
    output_dim=10,
    num_heads=2,
    num_layers=1,
    dropout=0.1,
    use_hyperbolic=False,
    use_neural_ode=False,
    multi_task=True,
    company_input_dim=company_input_dim
)

# Test company projection
company_features = torch.rand(batch_size, company_input_dim)
print(f"Input company features shape: {company_features.shape}")

# Initialize dim_alignment
if 'company' not in model.dim_alignment:
    model.dim_alignment['company'] = torch.nn.Linear(company_input_dim, 32)

# Test company projection directly
company_h = model.company_projection(company_features)
print(f"Projected company features shape: {company_h.shape}")

# Test company context layer directly
context_layer = CompanyAwareContextLayer(hidden_dim=hidden_dim)
transaction_features = torch.rand(batch_size, 3, hidden_dim)  # batch, seq_len, hidden_dim
company_h = company_h.unsqueeze(1)
print(f"Transaction features shape: {transaction_features.shape}")
print(f"Company features shape for context: {company_h.shape}")

# Apply company context layer
enriched_features = context_layer(transaction_features, company_h)
print(f"Enriched transaction features shape: {enriched_features.shape}")

print("All tests passed successfully\!")
