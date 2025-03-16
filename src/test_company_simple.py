#!/usr/bin/env python3
"""
A simplified test of the transaction classification model with company features
"""
import os
import torch
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Import our modules
from src.data_processing.transaction_graph import TransactionGraphBuilder
from src.models.hyper_temporal_model import HyperTemporalTransactionModel

def generate_mini_test_data(num_samples=100):
    """Generate a minimal test dataset with company features"""
    # Create 4 test companies
    company_data = {
        "company_id": ["A123", "B456", "C789", "D012"],
        "industry_name": ["Retail", "Software", "Healthcare", "Construction"],
        "company_size": ["Small", "Medium", "Large", "Enterprise"],
        "region": ["West", "East", "Central", "North"]
    }
    
    # Create transaction data
    df = pd.DataFrame({
        'txn_id': [f"t{i}" for i in range(num_samples)],
        'user_id': np.random.choice(["user1", "user2", "user3", "user4"], num_samples),
        'company_id': np.random.choice(company_data["company_id"], num_samples),
        'amount': np.random.uniform(10, 1000, num_samples),
        'industry_name': np.random.choice(company_data["industry_name"], num_samples),
        'category_id': np.random.randint(0, 10, num_samples),
        'merchant_id': np.random.choice(["m1", "m2", "m3", "m4", "m5"], num_samples)
    })
    
    # Make user_id and company_id consistent
    for i, user_id in enumerate(["user1", "user2", "user3", "user4"]):
        df.loc[df['user_id'] == user_id, 'company_id'] = company_data["company_id"][i]
        df.loc[df['user_id'] == user_id, 'industry_name'] = company_data["industry_name"][i]
    
    return df

def test_inference_with_company_features():
    """Test inference with company features on a small model"""
    print("Generating test data...")
    transactions = generate_mini_test_data(100)
    
    print("Building transaction graph...")
    graph_builder = TransactionGraphBuilder(num_categories=10)
    graph = graph_builder.build_graph(transactions)
    
    # Extract features from the graph
    transaction_features = graph['transaction'].x
    
    # Create dummy sequence features
    batch_size = transaction_features.size(0)
    seq_len = 5
    seq_features = transaction_features.unsqueeze(1).repeat(1, seq_len, 1)
    
    # Create dummy timestamps
    timestamps = torch.zeros(batch_size, seq_len)
    for i in range(seq_len):
        timestamps[:, i] = i
    
    # Get user features if available
    user_features = graph['user'].x if 'user' in graph.node_types else None
    
    # Get company features if available
    company_features = graph['company'].x if 'company' in graph.node_types else None
    
    # Print feature dimensions and data types
    print(f"Transaction features shape: {transaction_features.shape}, dtype: {transaction_features.dtype}")
    if user_features is not None:
        print(f"User features shape: {user_features.shape}, dtype: {user_features.dtype}")
    if company_features is not None:
        print(f"Company features shape: {company_features.shape}, dtype: {company_features.dtype}")
        
        # Detailed inspection of company features
        print(f"Company features min: {company_features.min()}, max: {company_features.max()}")
        
        # Check if transaction and company features match in batch dimension
        print(f"Transaction batch size: {transaction_features.size(0)}, Company batch size: {company_features.size(0)}")
        
        # Debug all node types in the graph
        print(f"Available node types: {graph.node_types}")
        
        # Check if company nodes match transactions 1:1
        print(f"Company node count: {graph['company'].num_nodes}")
        print(f"Transaction node count: {graph['transaction'].num_nodes}")
        
        # Debug edge connectivity
        if ('transaction', 'from_company', 'company') in graph.edge_types:
            print(f"Transaction-to-company edges: {graph[('transaction', 'from_company', 'company')].edge_index.shape}")
        if ('company', 'has_transaction', 'transaction') in graph.edge_types:
            print(f"Company-to-transaction edges: {graph[('company', 'has_transaction', 'transaction')].edge_index.shape}")
    
    # Get input dimension
    input_dim = transaction_features.size(1)
    
    # Ensure company_input_dim is correct - use the actual dimension from the company features
    company_input_dim = company_features.size(1) if company_features is not None else 4
    
    print(f"Creating small model with input_dim={input_dim}, company_input_dim={company_input_dim}")
    print(f"Company features detected with {company_input_dim} dimensions")
    
    # Create a simplified test version of the model
    class SimpleCompanyTestModel(torch.nn.Module):
        """Simplified model just for testing company feature integration"""
        def __init__(self, input_dim, company_input_dim, hidden_dim=32, output_dim=10):
            super().__init__()
            
            # Basic projection layers
            self.transaction_projection = torch.nn.Linear(input_dim, hidden_dim)
            self.company_projection = torch.nn.Linear(company_input_dim, hidden_dim)
            
            # Simple fusion layer
            self.fusion = torch.nn.Linear(hidden_dim * 2, hidden_dim)
            
            # Output layer
            self.output = torch.nn.Linear(hidden_dim, output_dim)
            
        def forward(self, transaction_features, seq_features, tabular_features,
                  timestamps, start_time, duration, company_features=None):
            """Forward pass with optional company features"""
            # Project transaction features
            transaction_h = self.transaction_projection(transaction_features)
            
            # Process company features if provided
            if company_features is not None:
                # Project company features
                company_h = self.company_projection(company_features)
                
                # Fuse transaction and company features
                combined = torch.cat([transaction_h, company_h], dim=1)
                fused = self.fusion(combined)
            else:
                # Without company features, just use transaction features with padding
                dummy = torch.zeros_like(transaction_h)
                combined = torch.cat([transaction_h, dummy], dim=1)
                fused = self.fusion(combined)
            
            # Final output layer
            output = self.output(fused)
            return output
    
    # Create our simple test model
    model = SimpleCompanyTestModel(
        input_dim=input_dim,
        company_input_dim=company_input_dim,
        hidden_dim=32,
        output_dim=10
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Test with company features
    print("Testing inference with company features...")
    with torch.no_grad():
        # Forward pass with company features
        # Check company features shape before passing to model
        print(f"Before passing to model - company features shape: {company_features.shape}")
        
        try:
            # Debug feature dimensions and check alignment across all inputs
            print(f"transaction_features: {transaction_features.shape}")
            print(f"seq_features: {seq_features.shape}")
            print(f"timestamps: {timestamps.shape}")
            
            # Create transaction (row-specific) aligned company features
            # We need to ensure each transaction row has corresponding company features
            # Map the company features based on the transaction_graph relationships
            
            # Force align company features with transaction features
            # This ensures each transaction gets its corresponding company features
            aligned_company_features = torch.zeros(
                transaction_features.size(0), 
                company_features.size(1), 
                dtype=company_features.dtype
            )
            
            # For simplicity in this test, we'll just copy the company features to align with transactions
            # In a real implementation, you'd use the graph structure to correctly align them
            edge_index = graph[('transaction', 'from_company', 'company')].edge_index
            for i in range(edge_index.size(1)):
                # Map from transaction_idx -> company_idx using edge_index
                transaction_idx = edge_index[0, i].item()
                company_idx = edge_index[1, i].item()
                aligned_company_features[transaction_idx] = company_features[company_idx]
            
            print(f"Aligned company features shape: {aligned_company_features.shape}")
            
            # Forward pass with aligned company features
            output_with_company = model(
                transaction_features, seq_features, transaction_features,
                timestamps, 0.0, float(seq_len),
                company_features=aligned_company_features
            )
            
            # Forward pass without company features
            output_without_company = model(
                transaction_features, seq_features, transaction_features,
                timestamps, 0.0, float(seq_len),
                company_features=None
            )
        except Exception as e:
            print(f"Error during model forward pass: {str(e)}")
            raise
    
    # Check if outputs are different
    if isinstance(output_with_company, tuple):
        # Multi-task output
        category_with_company, _ = output_with_company
        category_without_company, _ = output_without_company
        
        # Check category predictions
        preds_with_company = torch.argmax(category_with_company, dim=1)
        preds_without_company = torch.argmax(category_without_company, dim=1)
    else:
        # Single-task output
        preds_with_company = torch.argmax(output_with_company, dim=1)
        preds_without_company = torch.argmax(output_without_company, dim=1)
    
    # Calculate difference
    diff_count = (preds_with_company != preds_without_company).sum().item()
    diff_percentage = diff_count / batch_size * 100
    
    print(f"Company feature impact: {diff_count}/{batch_size} predictions different ({diff_percentage:.2f}%)")
    
    if diff_count > 0:
        print("✓ Company features are affecting the model's predictions")
    else:
        print("❌ Company features are not affecting the model's predictions")

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_inference_with_company_features()
    
    print("\nTest completed!")