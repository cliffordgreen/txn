#!/usr/bin/env python3
"""
Simple test script for company features integration - with a minimal setup to debug issues
"""
import os
import torch
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Import our modules
from src.data_processing.transaction_graph import TransactionGraphBuilder
from src.models.hyper_temporal_model import CompanyAwareContextLayer, MultiModalFusion

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

def test_graph_builder_with_company():
    """Test that the graph builder correctly processes company features"""
    # Create test data
    transactions = generate_mini_test_data(100)
    
    # Create graph builder
    graph_builder = TransactionGraphBuilder(num_categories=10)
    
    # Build graph and extract features
    graph = graph_builder.build_graph(transactions)
    
    # Check if company features were created
    if 'company' in graph.node_types:
        print(f"✓ Company nodes added to graph: {len(graph['company'].x)} nodes with {graph['company'].x.size(1)} features")
    else:
        print("❌ No company nodes in graph")
    
    # Check if company edges were created
    if ('transaction', 'from_company', 'company') in graph.edge_types:
        edge_index = graph['transaction', 'from_company', 'company'].edge_index
        print(f"✓ Transaction-company edges created: {edge_index.size(1)} edges")
    else:
        print("❌ No transaction-company edges in graph")
    
    return graph

def test_company_features_in_model():
    """Test that company features can be integrated into the model"""
    # Create fake features 
    batch_size = 32
    seq_len = 5
    hidden_dim = 64
    
    # Create test company features that match the expected dimension
    company_features = torch.randn(batch_size, hidden_dim)  # Company features must match hidden_dim
    
    # Create test transaction features
    transaction_features = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create company-aware context layer
    company_layer = CompanyAwareContextLayer(hidden_dim=hidden_dim)
    
    # Create multi-modal fusion
    fusion = MultiModalFusion(input_dim=hidden_dim, hidden_dim=hidden_dim)
    
    # Test company context layer
    try:
        output = company_layer(transaction_features, company_features)
        print(f"✓ CompanyAwareContextLayer works: input {transaction_features.shape} -> output {output.shape}")
    except Exception as e:
        print(f"❌ CompanyAwareContextLayer failed: {e}")
    
    # Test fusion layer
    try:
        graph_features = torch.randn(batch_size, hidden_dim)
        seq_features = torch.randn(batch_size, hidden_dim)
        tabular_features = torch.randn(batch_size, hidden_dim)
        
        output = fusion(graph_features, seq_features, tabular_features, company_features)
        print(f"✓ MultiModalFusion works with company features: output {output.shape}")
    except Exception as e:
        print(f"❌ MultiModalFusion failed: {e}")

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Testing company features in graph building...")
    graph = test_graph_builder_with_company()
    
    print("\nTesting company features in model components...")
    test_company_features_in_model()
    
    print("\nAll tests complete!")