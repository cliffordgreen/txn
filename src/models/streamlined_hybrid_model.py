import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Import our custom models
from src.models.hyper_temporal_model import HyperTemporalTransactionModel
from src.models.streamlined_graph_model import GraphEnhancedTemporalModel
from src.data_processing.transaction_graph import build_transaction_relationship_graph

class EnhancedHybridTransactionModel(nn.Module):
    """
    Enhanced hybrid model that integrates:
    1. Graph-based relationships (merchant, company, industry, and price)
    2. Temporal patterns with company-based grouping
    3. Hyperbolic encoding for hierarchical relationships
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 400,
                 num_heads: int = 8, num_graph_layers: int = 2, num_temporal_layers: int = 2, 
                 dropout: float = 0.2, use_hyperbolic: bool = True, use_neural_ode: bool = False,
                 use_text: bool = False, multi_task: bool = True, tax_type_dim: int = 20,
                 company_input_dim: Optional[int] = None, num_relations: int = 5,
                 graph_weight: float = 0.6, temporal_weight: float = 0.4,
                 use_dynamic_weighting: bool = True):
        """Initialize the enhanced hybrid transaction model."""
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multi_task = multi_task
        self.tax_type_dim = tax_type_dim
        self.graph_weight = graph_weight
        self.temporal_weight = temporal_weight
        self.use_dynamic_weighting = use_dynamic_weighting
        
        # Normalize weights to sum to 1
        total_weight = graph_weight + temporal_weight
        self.graph_weight = graph_weight / total_weight
        self.temporal_weight = temporal_weight / total_weight
        
        # Graph-enhanced model
        self.graph_model = GraphEnhancedTemporalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_graph_layers=num_graph_layers,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout,
            use_hyperbolic=use_hyperbolic,
            use_neural_ode=use_neural_ode,
            multi_task=multi_task,
            tax_type_output_dim=tax_type_dim,
            company_input_dim=company_input_dim,
            num_relations=num_relations
        )
        
        # Temporal model
        self.temporal_model = HyperTemporalTransactionModel(
            input_dim=hidden_dim,  # Match the projected dimension
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout,
            use_hyperbolic=use_hyperbolic,
            use_neural_ode=use_neural_ode,
            use_text_processor=use_text,
            graph_input_dim=hidden_dim,
            company_input_dim=company_input_dim,
            tax_type_output_dim=tax_type_dim,
            multi_task=multi_task
        )
        
        # Dynamic weighting module (if used)
        if use_dynamic_weighting:
            self.weight_module = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),  # 2 weights: graph and temporal
                nn.Softmax(dim=1)
            )
            
            # Feature extractors for weight calculation
            self.graph_feature_extractor = nn.Linear(output_dim, hidden_dim)
            self.temporal_feature_extractor = nn.Linear(output_dim, hidden_dim)
        
        # Input projection for adapting dimensions
        self.input_projection = nn.LazyLinear(hidden_dim)
        self.tabular_projection = nn.LazyLinear(hidden_dim)
        
        # Output projections
        self.category_output = nn.Linear(output_dim, output_dim)
        if multi_task:
            self.tax_type_output = nn.Linear(tax_type_dim, tax_type_dim)
        
    def forward(self, x, edge_index, edge_type, edge_attr, seq_features, 
                timestamps, tabular_features, t0, t1, descriptions=None,
                company_features=None, company_ids=None, batch_size=None, seq_len=None):
        """Forward pass through the enhanced hybrid model."""
        # Get batch size and sequence length if not provided
        if batch_size is None and seq_features is not None:
            batch_size = seq_features.shape[0]
        if seq_len is None and seq_features is not None:
            seq_len = seq_features.shape[1]
        
        # Ensure input dimensions match the model's expectations
        if x.shape[1] != self.hidden_dim:
            x = self.input_projection(x)
        
        # Forward pass through graph model
        graph_output = self.graph_model(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            seq_features=seq_features,
            timestamps=timestamps,
            company_features=company_features,
            company_ids=company_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            t0=t0,
            t1=t1
        )
        
        # Project tabular features to match expected dimensions
        projected_tabular_features = self.tabular_projection(tabular_features)
        
        # Forward pass through temporal model
        temporal_output = self.temporal_model(
            graph_features=projected_tabular_features,
            seq_features=seq_features,
            tabular_features=projected_tabular_features,
            timestamps=timestamps,
            t0=t0,
            t1=t1,
            descriptions=descriptions,
            company_features=company_features,
            company_ids=company_ids
        )
        
        # Calculate weights if using dynamic weighting
        if self.use_dynamic_weighting:
            # Extract features from outputs for weight calculation
            if self.multi_task:
                graph_cat, _ = graph_output
                temporal_cat, _ = temporal_output
            else:
                graph_cat = graph_output
                temporal_cat = temporal_output
                
            graph_features = self.graph_feature_extractor(graph_cat)
            temporal_features = self.temporal_feature_extractor(temporal_cat)
            
            # Calculate weights
            combined_features = torch.cat([graph_features, temporal_features], dim=1)
            weights = self.weight_module(combined_features)
            
            graph_weight = weights[:, 0].unsqueeze(1)
            temporal_weight = weights[:, 1].unsqueeze(1)
        else:
            # Use fixed weights
            graph_weight = self.graph_weight
            temporal_weight = self.temporal_weight
        
        # Combine the outputs based on weights
        if self.multi_task:
            # Unpack the outputs
            graph_category, graph_tax = graph_output
            temporal_category, temporal_tax = temporal_output
            
            # Weight and combine the outputs
            combined_category = graph_weight * graph_category + temporal_weight * temporal_category
            combined_tax = graph_weight * graph_tax + temporal_weight * temporal_tax
            
            # Apply output projections
            category_logits = self.category_output(combined_category)
            tax_type_logits = self.tax_type_output(combined_tax)
            
            return category_logits, tax_type_logits
        else:
            # Single task - combine directly
            combined_output = graph_weight * graph_output + temporal_weight * temporal_output
            category_logits = self.category_output(combined_output)
            
            return category_logits
            
    def extract_embeddings(self, data):
        """Extract embeddings from the graph model."""
        self.eval()
        with torch.no_grad():
            # Extract embeddings from graph model
            if hasattr(self.graph_model, 'extract_embeddings'):
                embeddings = self.graph_model.extract_embeddings(
                    x=data['x'],
                    edge_index=data['edge_index'],
                    edge_type=data['edge_type'],
                    edge_attr=data['edge_attr']
                )
                return embeddings
            else:
                return None
    
    def prepare_data_from_dataframe(self, df):
        """Prepare data for the model from a DataFrame."""
        # Build transaction relationship graph
        edge_index, edge_attr, edge_type = build_transaction_relationship_graph(df)
        
        # Extract features from the transaction data
        features = []
        
        # Numerical Features
        if 'amount' in df.columns:
            amount = df['amount'].values
            amount_normalized = (amount - np.mean(amount)) / (np.std(amount) + 1e-8)
            features.append(amount_normalized)
            log_amount = np.log1p(np.abs(amount)) * np.sign(amount)
            features.append(log_amount)
            
        # Entity ID features
        for id_col in ['user_id', 'merchant_id', 'company_id', 'industry_code', 'region_id', 
                      'language_id', 'account_type_id', 'scheduleC_id']:
            if id_col in df.columns:
                ids = pd.factorize(df[id_col])[0]
                ids_norm = ids / max(1, ids.max())
                features.append(ids_norm)
                
                # Extract company IDs for temporal grouping
                if id_col == 'company_id':
                    company_ids_tensor = torch.tensor(ids, dtype=torch.long)
                
        # Boolean Features
        for bool_col in ['is_new_user', 'is_before_cutoff_date', 'qbo_accountant_attached_current_flag',
                         'qbo_accountant_attached_ever', 'qblive_attach_flag']:
            if bool_col in df.columns:
                features.append(df[bool_col].values.astype(float))
                
        # Categorical Features (One-Hot Encoded)
        for cat_col in ['transaction_type', 'qbo_current_product', 'qbo_signup_type_desc',
                       'company_model_bucket_name', 'industry_name']:
            if cat_col in df.columns:
                dummies = pd.get_dummies(df[cat_col])
                for col in dummies.columns:
                    features.append(dummies[col].values)
        
        # Timestamp Features
        for ts_col in ['books_create_timestamp', 'generated_timestamp', 'update_timestamp']:
            if ts_col in df.columns:
                try:
                    timestamps = pd.to_datetime(df[ts_col])
                    
                    # Month as cyclical feature
                    month = timestamps.dt.month.values.astype(float)
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    features.append(month_sin)
                    features.append(month_cos)
                    
                    # Day of week as cyclical feature
                    day_of_week = timestamps.dt.dayofweek.values.astype(float)
                    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
                    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
                    features.append(dow_sin)
                    features.append(dow_cos)
                except Exception:
                    pass
        
        # Combine features into a matrix
        feature_matrix = np.column_stack(features) if features else np.zeros((len(df), 1))
        
        # Ensure we have enough features for the model's input projection
        min_features = self.input_dim if hasattr(self, 'input_dim') else 128
        if feature_matrix.shape[1] < min_features:
            padded_matrix = np.zeros((len(df), min_features))
            padded_matrix[:, :feature_matrix.shape[1]] = feature_matrix
            feature_matrix = padded_matrix
            
        node_features = torch.tensor(feature_matrix, dtype=torch.float)
        
        # Create sequence features
        batch_size = min(len(df), 128)  # Use a reasonable batch size
        seq_len = 5  # Fixed sequence length for simplicity
        
        # Create sequence features from node features
        seq_features = torch.zeros((batch_size, seq_len, node_features.shape[1]))
        for i in range(batch_size):
            # For each batch item, get a sequence of transactions
            start_idx = i
            for j in range(seq_len):
                idx = start_idx + j if start_idx + j < len(df) else start_idx  # Prevent index out of bounds
                seq_features[i, j] = node_features[idx]
        
        # Create timestamps
        timestamps_tensor = None
        if 'generated_timestamp' in df.columns or 'timestamp' in df.columns:
            try:
                # Get timestamp column
                ts_col = 'generated_timestamp' if 'generated_timestamp' in df.columns else 'timestamp'
                timestamps = pd.to_datetime(df[ts_col], errors='coerce')
                
                if not timestamps.isna().all():
                    # Fill missing values and convert to seconds
                    median_ts = timestamps.median()
                    timestamps = timestamps.fillna(median_ts)
                    timestamps_int = timestamps.astype('int64') // 10**9
                    
                    # Normalize timestamps
                    min_ts = timestamps_int.min()
                    timestamps_norm = timestamps_int - min_ts
                    
                    # Create tensor with the right shape
                    timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
                    
                    # If we have company_id, use it to create meaningful temporal sequences
                    if 'company_id' in df.columns:
                        company_groups = df.groupby('company_id')
                        companies = list(company_groups.groups.keys())
                        
                        for i in range(batch_size):
                            company_idx = i % len(companies)
                            company = companies[company_idx]
                            
                            # Get this company's transactions
                            company_df = company_groups.get_group(company)
                            company_ts = timestamps_norm[company_df.index].sort_values()
                            
                            if len(company_ts) > 0:
                                # Get available timestamps
                                available_ts = min(len(company_ts), seq_len)
                                timestamps_tensor[i, :available_ts] = torch.tensor(
                                    company_ts.iloc[:available_ts].values, dtype=torch.float32
                                )
                    else:
                        # No company grouping - just use chronological order
                        sorted_ts = np.sort(timestamps_norm.values)
                        for i in range(batch_size):
                            start_idx = i * seq_len
                            end_idx = min(start_idx + seq_len, len(sorted_ts))
                            if end_idx > start_idx:
                                actual_len = end_idx - start_idx
                                timestamps_tensor[i, :actual_len] = torch.tensor(
                                    sorted_ts[start_idx:end_idx], dtype=torch.float32
                                )
            except Exception:
                # Fallback to synthetic timestamps
                timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
                for i in range(batch_size):
                    timestamps_tensor[i] = torch.tensor([float(j) for j in range(seq_len)], dtype=torch.float32)
        else:
            # No timestamp column - create synthetic timestamps
            timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
            for i in range(batch_size):
                timestamps_tensor[i] = torch.tensor([float(j) for j in range(seq_len)], dtype=torch.float32)
        
        # Package data
        data = {
            'x': node_features,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_attr': edge_attr,
            'seq_features': seq_features,
            'timestamps': timestamps_tensor,
            'tabular_features': node_features[:batch_size].clone(),
            't0': 0.0,
            't1': 1.0,
            'company_features': None,  # Add company features if available
            'company_ids': locals().get('company_ids_tensor', None),
            'batch_size': batch_size,
            'seq_len': seq_len
        }
        
        return data