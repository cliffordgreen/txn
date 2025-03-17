import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import random


def build_transaction_relationship_graph(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a comprehensive transaction graph with multiple relationship types:
    - Company relationships (transactions from same company)
    - Merchant relationships (transactions with same merchant)
    - Industry relationships (transactions in same industry)
    - Price similarity relationships (transactions with similar amounts)
    - Temporal relationships (sequential transactions from same company)
    
    Args:
        df: DataFrame containing transaction data with columns:
            - company_id: Company identifier
            - merchant_id: Merchant identifier
            - industry_code: (Optional) Industry code
            - amount: (Optional) Transaction amount
            - timestamp: Transaction timestamp
            
    Returns:
        edge_index: Tensor of shape [2, num_edges] containing edge indices
        edge_attr: Tensor of shape [num_edges, 1] containing edge weights
        edge_type: Tensor of shape [num_edges] containing edge type identifiers
            0: company relationship
            1: merchant relationship
            2: industry relationship
            3: price relationship
            4: temporal relationship
    """
    edges = []
    edge_attrs = []
    edge_types = []  # To track the type of each relationship
    
    # 1. Create edges between transactions from the same company
    if 'company_id' in df.columns:
        company_groups = df.groupby('company_id')
        for company_id, group in company_groups:
            txn_indices = group.index.tolist()
            
            # If too many transactions, sample pairs to avoid creating too many edges
            if len(txn_indices) > 50:
                # Select random pairs with a cap
                num_pairs = min(1000, len(txn_indices) * 5)
                random.seed(42)  # For reproducibility
                pairs = [(txn_indices[i], txn_indices[j]) 
                        for i in range(len(txn_indices)) 
                        for j in range(i+1, len(txn_indices))]
                
                if len(pairs) > num_pairs:
                    pairs = random.sample(pairs, num_pairs)
                
                for src, dst in pairs:
                    edges.append((src, dst))
                    
                    # Add edge weight based on time difference if timestamp exists
                    if 'timestamp' in group.columns:
                        try:
                            time_diff = abs((group.loc[src, 'timestamp'] - group.loc[dst, 'timestamp']).total_seconds())
                            similarity = 1.0 / (1.0 + time_diff/86400)  # Normalize by day
                        except:
                            similarity = 0.7  # Default if timestamp comparison fails
                    else:
                        similarity = 0.7  # Default company relationship strength
                        
                    edge_attrs.append([similarity])
                    edge_types.append(0)  # 0 = company relationship
            else:
                # If few transactions, create edges between all pairs
                for i in range(len(txn_indices)):
                    for j in range(i+1, len(txn_indices)):
                        src, dst = txn_indices[i], txn_indices[j]
                        edges.append((src, dst))
                        
                        # Add edge weight based on time difference if timestamp exists
                        if 'timestamp' in group.columns:
                            try:
                                time_diff = abs((group.loc[src, 'timestamp'] - group.loc[dst, 'timestamp']).total_seconds())
                                similarity = 1.0 / (1.0 + time_diff/86400)  # Normalize by day
                            except:
                                similarity = 0.7  # Default if timestamp comparison fails
                        else:
                            similarity = 0.7  # Default company relationship strength
                            
                        edge_attrs.append([similarity])
                        edge_types.append(0)  # 0 = company relationship
    
    # 2. Create edges between transactions with the same merchant
    if 'merchant_id' in df.columns:
        merchant_groups = df.groupby('merchant_id')
        for merchant_id, group in merchant_groups:
            txn_indices = group.index.tolist()
            
            # If too many transactions with same merchant, sample
            if len(txn_indices) > 50:
                # Select random pairs with a cap
                num_pairs = min(1000, len(txn_indices) * 5)
                random.seed(43)  # Different seed from company relationships
                pairs = [(txn_indices[i], txn_indices[j]) 
                        for i in range(len(txn_indices)) 
                        for j in range(i+1, len(txn_indices))]
                
                if len(pairs) > num_pairs:
                    pairs = random.sample(pairs, num_pairs)
                
                for src, dst in pairs:
                    edges.append((src, dst))
                    edge_attrs.append([0.85])  # Merchant relationships are strong
                    edge_types.append(1)  # 1 = merchant relationship
            else:
                # If few transactions, create edges between all pairs
                for i in range(len(txn_indices)):
                    for j in range(i+1, len(txn_indices)):
                        edges.append((txn_indices[i], txn_indices[j]))
                        edge_attrs.append([0.85])  # Merchant relationships are strong
                        edge_types.append(1)  # 1 = merchant relationship
    
    # 3. Create edges between transactions from the same industry
    if 'industry_code' in df.columns:
        industry_groups = df.groupby('industry_code')
        for industry_code, group in industry_groups:
            txn_indices = group.index.tolist()
            
            # For large industries, limit connections
            if len(txn_indices) > 100:
                # Sample connections randomly - fewer than company/merchant
                num_pairs = min(500, len(txn_indices) * 2)
                random.seed(44)  # Different seed
                pairs = [(txn_indices[i], txn_indices[j]) 
                        for i in range(len(txn_indices)) 
                        for j in range(i+1, len(txn_indices))]
                
                if len(pairs) > num_pairs:
                    pairs = random.sample(pairs, num_pairs)
                
                for src, dst in pairs:
                    edges.append((src, dst))
                    edge_attrs.append([0.6])  # Industry connections have lower weight
                    edge_types.append(2)  # 2 = industry relationship
            elif len(txn_indices) > 10:
                # Medium-sized industry groups
                for i in range(len(txn_indices)):
                    # Connect to a subset of other transactions
                    for j in range(i+1, min(i+20, len(txn_indices))):
                        edges.append((txn_indices[i], txn_indices[j]))
                        edge_attrs.append([0.6])  # Industry connections have lower weight
                        edge_types.append(2)  # 2 = industry relationship
            else:
                # Small industry groups - connect all
                for i in range(len(txn_indices)):
                    for j in range(i+1, len(txn_indices)):
                        edges.append((txn_indices[i], txn_indices[j]))
                        edge_attrs.append([0.6])  # Industry connections have lower weight
                        edge_types.append(2)  # 2 = industry relationship
    
    # 4. Create edges based on price similarity
    if 'amount' in df.columns:
        try:
            # Get the amount values
            amounts = df['amount'].values
            
            # Normalize the amounts for binning
            min_amount = np.min(amounts)
            max_amount = np.max(amounts)
            if max_amount > min_amount:  # Avoid division by zero
                normalized_amounts = (amounts - min_amount) / (max_amount - min_amount)
                
                # Group transactions into amount bins (e.g., 10 bins)
                num_bins = 10
                amount_bins = np.digitize(normalized_amounts, bins=np.linspace(0, 1, num_bins))
                
                # Create edges between transactions in the same price bin
                for bin_idx in range(1, num_bins + 1):
                    bin_indices = np.where(amount_bins == bin_idx)[0]
                    
                    # Skip empty bins
                    if len(bin_indices) <= 1:
                        continue
                    
                    # If too many transactions in same bin, sample
                    if len(bin_indices) > 50:
                        # Sample a limited number of pairs
                        num_pairs = min(300, len(bin_indices) * 2)
                        random.seed(45)  # Different seed
                        
                        # Create all possible pairs
                        pairs = [(i, j) for i in range(len(bin_indices)) for j in range(i+1, len(bin_indices))]
                        
                        # Sample if too many
                        if len(pairs) > num_pairs:
                            pairs = random.sample(pairs, num_pairs)
                        
                        for i, j in pairs:
                            idx1, idx2 = bin_indices[i], bin_indices[j]
                            edges.append((idx1, idx2))
                            
                            # Weight based on actual amount similarity
                            amount_sim = 1.0 - abs(normalized_amounts[idx1] - normalized_amounts[idx2])
                            edge_attrs.append([amount_sim * 0.7])  # Amount edges have 0.7 max weight
                            edge_types.append(3)  # 3 = price relationship
                    else:
                        for i in range(len(bin_indices)):
                            for j in range(i+1, len(bin_indices)):
                                idx1, idx2 = bin_indices[i], bin_indices[j]
                                edges.append((idx1, idx2))
                                
                                # Weight based on actual amount similarity
                                amount_sim = 1.0 - abs(normalized_amounts[idx1] - normalized_amounts[idx2])
                                edge_attrs.append([amount_sim * 0.7])  # Amount edges have 0.7 max weight
                                edge_types.append(3)  # 3 = price relationship
        except Exception as e:
            print(f"Error creating price similarity edges: {str(e)}")
    
    # 5. Create sequential transaction edges (temporal)
    # Sort transactions by timestamp if available
    if 'timestamp' in df.columns and 'company_id' in df.columns:
        try:
            sorted_df = df.sort_values('timestamp')
            sorted_indices = sorted_df.index.tolist()
            
            for i in range(len(sorted_indices) - 1):
                curr_idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]
                
                # Only connect sequential transactions within the same company
                if sorted_df.loc[curr_idx, 'company_id'] == sorted_df.loc[next_idx, 'company_id']:
                    edges.append((curr_idx, next_idx))
                    
                    # Weight based on time difference
                    time_diff = (sorted_df.loc[next_idx, 'timestamp'] - sorted_df.loc[curr_idx, 'timestamp']).total_seconds()
                    
                    # Stronger weight for transactions closer in time (max 0.95, min 0.1)
                    weight = max(0.1, min(0.95, 86400 / (time_diff + 3600)))  # 86400 = seconds in a day
                    edge_attrs.append([weight])
                    edge_types.append(4)  # 4 = temporal relationship
        except Exception as e:
            print(f"Error creating temporal edges: {str(e)}")
    
    # Convert to tensors if we have any edges
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_attr, edge_type
    else:
        # Return empty tensors if no edges
        return (torch.empty((2, 0), dtype=torch.long), 
                torch.empty((0, 1), dtype=torch.float), 
                torch.empty(0, dtype=torch.long))


def extract_graph_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Extract graph node features from transaction data.
    
    Args:
        df: DataFrame containing transaction data
        
    Returns:
        node_features: Tensor of node features [num_nodes, num_features]
    """
    features = []
    
    # 1. Amount features (if available)
    if 'amount' in df.columns:
        # Raw amount (normalized)
        amounts = df['amount'].values
        normalized_amounts = (amounts - np.mean(amounts)) / (np.std(amounts) + 1e-8)
        features.append(torch.tensor(normalized_amounts, dtype=torch.float).unsqueeze(1))
        
        # Log amount (helps with skewed distributions)
        log_amounts = np.log1p(np.abs(amounts))
        normalized_log_amounts = (log_amounts - np.mean(log_amounts)) / (np.std(log_amounts) + 1e-8)
        features.append(torch.tensor(normalized_log_amounts, dtype=torch.float).unsqueeze(1))
    
    # 2. Temporal features (if available)
    if 'timestamp' in df.columns:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                timestamps = pd.to_datetime(df['timestamp'])
            else:
                timestamps = df['timestamp']
            
            # Hour of day (cyclical encoding)
            hours = timestamps.dt.hour.values
            hour_sin = np.sin(2 * np.pi * hours / 24)
            hour_cos = np.cos(2 * np.pi * hours / 24)
            features.append(torch.tensor(hour_sin, dtype=torch.float).unsqueeze(1))
            features.append(torch.tensor(hour_cos, dtype=torch.float).unsqueeze(1))
            
            # Day of week (cyclical encoding)
            days = timestamps.dt.dayofweek.values
            day_sin = np.sin(2 * np.pi * days / 7)
            day_cos = np.cos(2 * np.pi * days / 7)
            features.append(torch.tensor(day_sin, dtype=torch.float).unsqueeze(1))
            features.append(torch.tensor(day_cos, dtype=torch.float).unsqueeze(1))
            
            # Month (cyclical encoding)
            months = timestamps.dt.month.values
            month_sin = np.sin(2 * np.pi * months / 12)
            month_cos = np.cos(2 * np.pi * months / 12)
            features.append(torch.tensor(month_sin, dtype=torch.float).unsqueeze(1))
            features.append(torch.tensor(month_cos, dtype=torch.float).unsqueeze(1))
            
            # Is weekend
            is_weekend = (days >= 5).astype(float)
            features.append(torch.tensor(is_weekend, dtype=torch.float).unsqueeze(1))
        except Exception as e:
            print(f"Error extracting temporal features: {str(e)}")
    
    # 3. Categorical features (one-hot encoded)
    categorical_cols = ['merchant_id', 'company_id', 'industry_code']
    for col in categorical_cols:
        if col in df.columns:
            try:
                # Get unique values and create mapping
                unique_values = df[col].unique()
                value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
                
                # Convert to indices
                indices = df[col].map(value_to_idx).values
                
                # One-hot encode (for smaller cardinality features)
                if len(unique_values) <= 100:
                    one_hot = np.eye(len(unique_values))[indices]
                    features.append(torch.tensor(one_hot, dtype=torch.float))
                else:
                    # For high cardinality features, use the index directly
                    # and an embedding will be learned during training
                    indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(1)
                    features.append(indices_tensor.float())
            except Exception as e:
                print(f"Error extracting {col} features: {str(e)}")
    
    # Concatenate all features
    if features:
        return torch.cat(features, dim=1)
    else:
        # Return empty tensor if no features
        return torch.empty((len(df), 0), dtype=torch.float)