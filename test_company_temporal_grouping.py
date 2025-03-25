import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

# Add UninitializedParameter to safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([UninitializedParameter])

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hybrid_transaction_model import EnhancedHybridTransactionModel

def test_company_ids_formatting():
    """Test proper company_ids formatting for DynamicContextualTemporal layer."""
    
    print("Creating test DataFrame with company_ids...")
    # Create a test DataFrame with company_ids
    df = pd.DataFrame({
        'amount': np.random.rand(100) * 1000,
        'company_id': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=100),
        'user_id': np.random.choice(['U1', 'U2', 'U3'], size=100),
        'merchant_id': np.random.choice(['M1', 'M2', 'M3'], size=100),
        'category_id': np.random.choice(['CAT1', 'CAT2', 'CAT3'], size=100),
        'transaction_type': np.random.choice(['DEBIT', 'CREDIT'], size=100),
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
    })
    
    print(f"Created DataFrame with {len(df)} rows and {df['company_id'].nunique()} unique companies")
    
    # Create a model with company-based temporal grouping
    print("Creating model...")
    model = EnhancedHybridTransactionModel(
        input_dim=64,
        hidden_dim=128,
        output_dim=3,  # 3 categories
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_graph_features=True,
        use_text_processor=False,
        graph_model_type='enhanced',
        temporal_model_type='hyper',
    )
    
    # Prepare data from DataFrame
    print("Preparing data from DataFrame...")
    batch_size = 32
    seq_len = 5
    data = model.prepare_data_from_dataframe(df, batch_size=batch_size, seq_len=seq_len)
    
    # Verify company_ids shape
    if 'company_ids' in data and data['company_ids'] is not None:
        company_ids = data['company_ids']
        print(f"✅ company_ids found with shape: {company_ids.shape}")
        
        # Check dimensions
        if len(company_ids.shape) == 2:
            print(f"✅ company_ids has correct [batch_size, seq_len] format: {company_ids.shape}")
            assert company_ids.shape[0] == min(batch_size, len(df) // seq_len + (1 if len(df) % seq_len > 0 else 0))
            assert company_ids.shape[1] == seq_len
        else:
            print(f"❌ company_ids has unexpected shape: {company_ids.shape}")
            
        # Check unique values
        unique_ids = torch.unique(company_ids)
        print(f"✅ Found {len(unique_ids)} unique company_ids")
        
        # Check distribution
        print("Company ID distribution:")
        for company_id in unique_ids:
            count = (company_ids == company_id).sum().item()
            print(f"  Company ID {company_id.item()}: {count} occurrences")
            
    else:
        print("❌ company_ids not found in prepared data")

if __name__ == "__main__":
    print("Testing company_ids formatting for DynamicContextualTemporal layer...")
    test_company_ids_formatting()
    print("Test completed.")