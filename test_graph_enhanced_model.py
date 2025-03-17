import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
print(f"Added module path: {current_dir}")

# Import our modules
from src.data_processing.transaction_graph import build_transaction_relationship_graph, extract_graph_features
try:
    from src.models.hybrid_transaction_model import EnhancedHybridTransactionModel
    has_enhanced_model = True
except ImportError:
    print("EnhancedHybridTransactionModel not available")
    has_enhanced_model = False

# Create synthetic data
def create_synthetic_data(num_transactions=500, num_companies=10, num_merchants=30, 
                          num_categories=50, seed=42):
    """Create synthetic transaction data for testing"""
    
    np.random.seed(seed)
    
    # Create merchant data
    merchants = [f"merchant_{i}" for i in range(num_merchants)]
    merchant_industries = np.random.randint(0, 10, size=num_merchants)
    merchant_to_industry = {m: i for m, i in zip(merchants, merchant_industries)}
    
    # Create company data
    companies = [f"company_{i}" for i in range(num_companies)]
    company_sizes = np.random.choice(['small', 'medium', 'large'], size=num_companies)
    company_types = np.random.choice(['LLC', 'Corp', 'SP'], size=num_companies)
    
    # Create timestamps covering 30 days with higher frequency during business hours
    start_time = pd.Timestamp('2023-01-01')
    end_time = pd.Timestamp('2023-01-31')
    
    # Business hours have higher frequency (9 AM - 5 PM on weekdays)
    def generate_business_weighted_timestamp():
        # Generate random date in range
        days = np.random.randint(0, 31)
        # Business days (weekdays) more likely
        is_weekend = np.random.random() < 0.3
        if is_weekend:
            # Weekend (Saturday/Sunday)
            day_of_week = np.random.choice([5, 6])  # 5=Sat, 6=Sun
        else:
            # Weekday
            day_of_week = np.random.randint(0, 5)  # 0=Mon, 4=Fri
            
        # Calculate actual date
        date = start_time + pd.Timedelta(days=days)
        # Adjust to ensure correct day of week
        date = date + pd.Timedelta(days=(day_of_week - date.dayofweek) % 7)
        
        # Business hours more likely on weekdays
        if not is_weekend and np.random.random() < 0.7:
            # Business hours (9 AM - 5 PM)
            hour = np.random.randint(9, 17)
        else:
            # Other hours
            hour = np.random.randint(0, 24)
            
        # Random minute and second
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        return date.replace(hour=hour, minute=minute, second=second)
    
    # Generate timestamps
    timestamps = [generate_business_weighted_timestamp() for _ in range(num_transactions)]
    timestamps.sort()  # Sort chronologically
    
    # Create transactions
    transactions = []
    
    for i in range(num_transactions):
        # Assign a company and merchant
        company_id = np.random.randint(0, num_companies)
        merchant_id = np.random.randint(0, num_merchants)
        
        # Get merchant name and industry
        merchant_name = merchants[merchant_id]
        industry_code = merchant_to_industry[merchant_name]
        
        # Get company details
        company_name = companies[company_id]
        company_size = company_sizes[company_id]
        company_type = company_types[company_id]
        
        # Generate amount (different distributions by industry)
        base_amount = np.random.lognormal(mean=4, sigma=1)  # Centered around ~$50
        industry_factor = 0.5 + industry_code / 10  # Industries have different avg amounts
        amount = base_amount * industry_factor
        
        # Introduce recurring patterns for some merchants (e.g., subscriptions)
        if merchant_id % 5 == 0:  # Every 5th merchant has recurring charges
            # Make amount more consistent
            amount = round(amount, -1)  # Round to nearest 10
        
        # Create category mapping - related to industry
        category_id = (industry_code * 5 + np.random.randint(0, 5)) % num_categories
        
        # Create transaction record
        transaction = {
            'timestamp': timestamps[i],
            'company_id': company_id,
            'company_name': company_name,
            'company_size': company_size,
            'company_type': company_type,
            'merchant_id': merchant_id,
            'merchant_name': merchant_name,
            'industry_code': industry_code,
            'amount': amount,
            'category_id': category_id,
            'category_name': f"category_{category_id}",
            'tax_type_id': np.random.randint(0, 5),  # 5 tax types
            'txn_id': f"txn_{i}"
        }
        
        transactions.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    return df

def test_graph_enhanced_model():
    """Test the graph-enhanced model with synthetic data"""
    
    if not has_enhanced_model:
        print("EnhancedHybridTransactionModel is not available. Skipping test.")
        return
    
    print("=== Testing Graph-Enhanced Transaction Model ===")
    
    # Create synthetic data
    print("Creating synthetic transaction data...")
    df = create_synthetic_data()
    print(f"Created {len(df)} synthetic transactions")
    print(f"Data columns: {df.columns.tolist()}")
    
    # Build transaction graph
    print("\nBuilding transaction relationship graph...")
    edge_index, edge_attr, edge_type = build_transaction_relationship_graph(df)
    
    # Count the different types of relationships
    if len(edge_type) > 0:
        unique_types, type_counts = torch.unique(edge_type, return_counts=True)
        print("\nRelationship types in the graph:")
        for t, count in zip(unique_types, type_counts):
            rel_type_name = ['company', 'merchant', 'industry', 'price', 'temporal'][t]
            print(f"  {rel_type_name}: {count.item()} edges")
    else:
        print("No edges found in the graph!")
        return
    
    # Extract node features
    print("\nExtracting node features...")
    node_features = extract_graph_features(df)
    print(f"Node features shape: {node_features.shape}")
    
    # Initialize model
    print("\nInitializing EnhancedHybridTransactionModel...")
    # We'll project the inputs to match these dimensions later
    hidden_dim = 64
    model = EnhancedHybridTransactionModel(
        input_dim=hidden_dim,  # Set to hidden_dim since we'll project the inputs
        hidden_dim=hidden_dim,  # Smaller for testing
        output_dim=df['category_id'].nunique(),
        num_heads=4,
        num_graph_layers=2,
        num_temporal_layers=2,
        dropout=0.1,
        use_hyperbolic=True,
        use_neural_ode=False,  # Faster for testing
        use_text=False,        # No text descriptions in synthetic data
        multi_task=True,
        tax_type_dim=df['tax_type_id'].nunique(),
        company_input_dim=6,   # Set to match company features dimensions from output
        num_relations=5,       # company, merchant, industry, price, temporal
        graph_weight=0.6,
        temporal_weight=0.4,
        use_dynamic_weighting=True
    )
    
    # Prepare data from DataFrame
    print("\nPreparing data...")
    data = model.prepare_data_from_dataframe(df)
    print("Data prepared successfully.")
    
    # Check data shapes
    print("\nPrepared data shapes:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # First initialize the input projections before the test
    print("\nInitializing input projections...")
    
    # Add a projection layer to adapt node features if needed
    input_dim = data['x'].size(1)  # Original input dimension
    hidden_dim = 64  # The model's hidden dimension
    
    # Create one-time projections
    node_projection = nn.Linear(input_dim, hidden_dim).to(data['x'].device)
    seq_projection = nn.Linear(input_dim, hidden_dim).to(data['seq_features'].device)
    
    # Project the inputs to have the right dimensions
    projected_x = node_projection(data['x'])
    projected_seq_features = seq_projection(data['seq_features'].reshape(-1, data['seq_features'].size(2)))
    projected_seq_features = projected_seq_features.reshape(data['seq_features'].size(0), data['seq_features'].size(1), hidden_dim)
    projected_tabular_features = seq_projection(data['tabular_features'])
    
    # Make sure company_ids match batch size - use only the first batch_size elements
    batch_size = data['batch_size']
    company_ids_subset = data['company_ids'][:batch_size] if data['company_ids'].size(0) > batch_size else data['company_ids']
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        output = model(
            x=projected_x,
            edge_index=data['edge_index'],
            edge_type=data['edge_type'],
            edge_attr=data['edge_attr'],
            seq_features=projected_seq_features,
            timestamps=data['timestamps'],
            tabular_features=projected_tabular_features,
            t0=data['t0'],
            t1=data['t1'],
            company_features=data['company_features'],
            company_ids=company_ids_subset,
            batch_size=data['batch_size'],
            seq_len=data['seq_len']
        )
        
        print("Forward pass successful!")
        
        if isinstance(output, tuple):
            category_logits, tax_type_logits = output
            print(f"  Category logits shape: {category_logits.shape}")
            print(f"  Tax type logits shape: {tax_type_logits.shape}")
        else:
            print(f"  Output logits shape: {output.shape}")
        
        # Test embedding extraction
        print("\nExtracting node embeddings...")
        embeddings = model.extract_embeddings(data)
        if embeddings is not None:
            print(f"Extracted embeddings shape: {embeddings.shape}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_enhanced_model()
    
    if success:
        print("\n🎉 EnhancedHybridTransactionModel is working properly!")
        print("""
Additional notes for using the EnhancedHybridTransactionModel:

1. Input Dimensions:
   - Ensure input_dim matches the dimension of projected features
   - If using raw features with different dimensions, add projection layers

2. Company Features:
   - The model automatically adapts company features with different dimensions
   - You can specify company_input_dim to match your data

3. Company-based Grouping:
   - Provide company_ids for temporal grouping to improve pattern detection
   - Make sure company_ids match the batch size

4. GPU Optimization:
   - For CUDA graph acceleration, use the built-in CUDA support
   - For p3.2xlarge instances with V100 GPUs, ensure tensors are on the correct device

5. Integration with XGBoost:
   - Use the extract_embeddings() method to get node embeddings
   - These embeddings can be used as features for XGBoost

For further assistance, refer to the documentation or contact support.
""")
    else:
        print("\n❌ Test failed. Check the error messages above.")