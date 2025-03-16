import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

# Import our custom modules
from src.data_processing.transaction_graph import TransactionGraphBuilder, create_train_val_test_split
from src.models.hyper_temporal_model import (
    HyperTemporalTransactionModel,
    HyperTemporalEnsemble,
    HyperbolicTransactionEncoder,
    MultiModalFusion,
    DynamicContextualTemporal
)
from src.train_transaction_classifier import generate_synthetic_data
import random


def generate_synthetic_transaction_descriptions(transaction_df):
    """
    Generate synthetic transaction descriptions for testing text processing capabilities.
    
    Args:
        transaction_df: DataFrame containing transaction data
        
    Returns:
        DataFrame with added transaction_description column
    """
    # Create a copy of the dataframe
    df = transaction_df.copy()
    
    # Sample merchant names
    retail_merchants = ["Walmart", "Target", "Amazon", "Best Buy", "Costco", "Home Depot", "Macy's",
                       "Kroger", "Safeway", "Publix", "Whole Foods", "IKEA", "Walgreens", "CVS", 
                       "Apple Store", "GameStop", "Lowes", "Trader Joe's", "Starbucks", "McDonalds",
                       "Subway", "Taco Bell", "Chipotle", "Pizza Hut", "Domino's", "KFC", "Wendy's"]
    
    tech_merchants = ["Apple", "Microsoft", "Google", "Samsung", "Dell", "HP", "Adobe", "Netflix",
                      "Spotify", "Hulu", "Dropbox", "Slack", "Zoom", "AWS", "GitHub", "Steam",
                      "PlayStation", "Xbox Live", "Nintendo"]
    
    finance_merchants = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One",
                         "Fidelity", "Vanguard", "PayPal", "Venmo", "Square", "Robinhood",
                         "Ameritrade", "E*Trade", "Visa", "Mastercard", "American Express"]
    
    # Transaction types
    purchase_prefixes = ["Purchase at", "Payment to", "Charge from", "Debit Card Purchase at", 
                        "Point of Sale Purchase at", "Online Payment to", "Subscription to",
                        "Recurring Payment to", "Automatic Payment to", "Mobile Payment to"]
    
    withdrawal_prefixes = ["ATM Withdrawal at", "Cash Withdrawal from", "Branch Withdrawal at"]
    
    deposit_prefixes = ["Deposit at", "Direct Deposit from", "Mobile Deposit", "Transfer from", 
                        "ACH Deposit from", "Electronic Deposit from"]
    
    subscription_words = ["Monthly", "Annual", "Premium", "Subscription", "Membership", "Service"]
    
    transaction_suffixes = ["Thank you", "Online", "Mobile", "In store", "Authorized on",
                           "#REF123456", "Pending", "Completed", "Confirmed", "Processed"]
    
    descriptions = []
    
    for _, row in df.iterrows():
        merchant_id = row['merchant_id']
        amount = row['amount']
        is_online = row.get('is_online', random.random() > 0.5)
        
        # Assign merchant name based on merchant_id
        merchant_name = ""
        if merchant_id < len(retail_merchants) / 2:
            merchant_name = retail_merchants[merchant_id % len(retail_merchants)]
        elif merchant_id < len(retail_merchants):
            merchant_name = tech_merchants[merchant_id % len(tech_merchants)]
        else:
            merchant_name = finance_merchants[merchant_id % len(finance_merchants)]
        
        # Create description based on amount and merchant
        if amount < 20:
            prefix = random.choice(purchase_prefixes)
            suffix = random.choice(transaction_suffixes) if random.random() > 0.7 else ""
            online_text = "Online" if is_online else ""
            description = f"{prefix} {merchant_name} {online_text} ${amount:.2f} {suffix}".strip()
        elif amount > 1000:
            if random.random() > 0.7:
                prefix = random.choice(deposit_prefixes)
                suffix = random.choice(transaction_suffixes) if random.random() > 0.7 else ""
                description = f"{prefix} {merchant_name} ${amount:.2f} {suffix}".strip()
            else:
                prefix = random.choice(purchase_prefixes)
                suffix = random.choice(transaction_suffixes) if random.random() > 0.7 else ""
                online_text = "Online" if is_online else ""
                description = f"{prefix} {merchant_name} {online_text} ${amount:.2f} {suffix}".strip()
        else:
            if random.random() > 0.8:
                prefix = random.choice(withdrawal_prefixes)
                description = f"{prefix} {merchant_name} ${amount:.2f}".strip()
            elif random.random() > 0.7:
                sub_word = random.choice(subscription_words)
                prefix = random.choice(purchase_prefixes)
                description = f"{prefix} {merchant_name} {sub_word} ${amount:.2f}".strip()
            else:
                prefix = random.choice(purchase_prefixes)
                suffix = random.choice(transaction_suffixes) if random.random() > 0.7 else ""
                online_text = "Online" if is_online else ""
                description = f"{prefix} {merchant_name} {online_text} ${amount:.2f} {suffix}".strip()
        
        descriptions.append(description)
    
    # Add descriptions to dataframe
    df['transaction_description'] = descriptions
    
    return df


class AdvancedTransactionSequenceBuilder:
    """
    Advanced sequence builder that constructs multi-level temporal transaction sequences 
    with sophisticated temporal features and user behavioral patterns.
    """
    
    def __init__(self, max_seq_length: int = 30, use_augmentation: bool = True,
                seasonal_encoding: bool = True, time_bucketing: bool = True):
        """
        Initialize the advanced sequence builder.
        
        Args:
            max_seq_length: Maximum sequence length
            use_augmentation: Whether to use data augmentation
            seasonal_encoding: Whether to use seasonal encoding
            time_bucketing: Whether to use time bucketing
        """
        self.max_seq_length = max_seq_length
        self.use_augmentation = use_augmentation
        self.seasonal_encoding = seasonal_encoding
        self.time_bucketing = time_bucketing
    
    def build_sequences(self, transactions_df: pd.DataFrame) -> tuple:
        """
        Build advanced transaction sequences from transaction data.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Tuple of (sequences, lengths, timestamps, user_ids, time_features)
        """
        # Make a copy to avoid modifying the original dataframe
        df = transactions_df.copy()
        
        # Convert timestamps to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add a user ID column if it doesn't exist
        if 'user_id' not in df.columns:
            num_users = max(100, len(df) // 30)  # Average 30 transactions per user
            df['user_id'] = np.random.randint(0, num_users, len(df))
        
        # Add advanced temporal features
        if self.seasonal_encoding:
            # Add cyclical time encodings
            df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
            
            # Extract temporal components
            df['day_of_month'] = df['datetime'].dt.day / 31
            df['week_of_year'] = df['datetime'].dt.isocalendar().week / 53
            df['quarter'] = df['datetime'].dt.quarter / 4
        
        # Add time since last transaction (by user)
        df = df.sort_values(['user_id', 'timestamp'])
        df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
        df['time_since_last_txn'] = (df['timestamp'] - df['prev_timestamp']).fillna(0)
        
        # Log transform time difference and normalize
        df['log_time_diff'] = np.log1p(df['time_since_last_txn'] / 3600)  # Log-scaled hours
        
        # Add time bucketing features if enabled
        if self.time_bucketing:
            # Create time buckets (morning, afternoon, evening, night)
            hour = df['datetime'].dt.hour
            df['time_bucket'] = pd.cut(
                hour, 
                bins=[0, 6, 12, 18, 24], 
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
            # One-hot encode time buckets
            time_bucket_dummies = pd.get_dummies(df['time_bucket'], prefix='time_bucket')
            df = pd.concat([df, time_bucket_dummies], axis=1)
            
            # Create day type (weekday vs weekend)
            df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
            
            # Create month seasonality
            df['is_holiday_season'] = df['datetime'].dt.month.isin([11, 12, 1])
        
        # Group transactions by user
        user_groups = df.groupby('user_id')
        
        # Initialize lists for sequences, lengths, and timestamps
        sequences = []
        seq_lengths = []
        seq_timestamps = []
        seq_time_features = []
        user_ids = []
        
        # Extract features for sequences (numerical and engineered)
        feature_cols = [col for col in df.columns 
                       if df[col].dtype in ['int64', 'float64'] 
                       and col not in ['user_id', 'transaction_id', 'timestamp', 'prev_timestamp']]
        
        # Add log-transformed amount
        df['log_amount'] = np.log1p(df['amount'])
        feature_cols.append('log_amount')
        
        # Build sequences for each user
        for user_id, group in user_groups:
            # Get user transactions
            user_transactions = group[feature_cols].values
            user_timestamps = group['timestamp'].values
            
            # Handle users with more than max_seq_length transactions
            if len(user_transactions) > self.max_seq_length:
                if self.use_augmentation:
                    # Use sliding window with variable stride to create multiple sequences
                    for stride_factor in [1, 2, 3]:  # Different strides for diversity
                        stride = max(1, self.max_seq_length // stride_factor)
                        for i in range(0, len(user_transactions) - self.max_seq_length + 1, stride):
                            end_idx = i + self.max_seq_length
                            seq = user_transactions[i:end_idx]
                            timestamps = user_timestamps[i:end_idx]
                            
                            # Extract time features
                            time_features = self._extract_time_features(timestamps)
                            
                            sequences.append(seq)
                            seq_lengths.append(len(seq))
                            seq_timestamps.append(timestamps)
                            seq_time_features.append(time_features)
                            user_ids.append(user_id)
                else:
                    # Use most recent transactions if no augmentation
                    seq = user_transactions[-self.max_seq_length:]
                    timestamps = user_timestamps[-self.max_seq_length:]
                    
                    # Extract time features
                    time_features = self._extract_time_features(timestamps)
                    
                    sequences.append(seq)
                    seq_lengths.append(len(seq))
                    seq_timestamps.append(timestamps)
                    seq_time_features.append(time_features)
                    user_ids.append(user_id)
            else:
                # Pad shorter sequences
                padded_seq = np.zeros((self.max_seq_length, len(feature_cols)))
                padded_timestamps = np.zeros(self.max_seq_length)
                
                seq_len = len(user_transactions)
                padded_seq[:seq_len] = user_transactions
                padded_timestamps[:seq_len] = user_timestamps
                
                # Extract time features
                time_features = self._extract_time_features(padded_timestamps[:seq_len])
                
                sequences.append(padded_seq)
                seq_lengths.append(seq_len)
                seq_timestamps.append(padded_timestamps)
                seq_time_features.append(time_features)
                user_ids.append(user_id)
        
        # Convert to tensors
        sequences = torch.FloatTensor(np.array(sequences))
        seq_lengths = torch.LongTensor(seq_lengths)
        seq_timestamps = torch.FloatTensor(np.array(seq_timestamps))
        
        # Ensure time features have consistent dimensions before conversion
        # Find the maximum dimension
        max_dim = max(tf.shape[0] if hasattr(tf, 'shape') else 0 for tf in seq_time_features)
        
        # Pad all time features to the same dimensions
        padded_time_features = []
        for tf in seq_time_features:
            if hasattr(tf, 'shape'):
                # Create zero tensor of max dimension
                padded = np.zeros((max_dim, 8))
                # Copy existing values
                padded[:tf.shape[0], :] = tf
                padded_time_features.append(padded)
            else:
                # Handle empty case
                padded_time_features.append(np.zeros((max_dim, 8)))
        
        seq_time_features = torch.FloatTensor(np.array(padded_time_features))
        user_ids = torch.LongTensor(user_ids)
        
        return sequences, seq_lengths, seq_timestamps, user_ids, seq_time_features
    
    def _extract_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Extract advanced time features from timestamps.
        
        Args:
            timestamps: Array of timestamps
            
        Returns:
            Array of time features
        """
        if len(timestamps) == 0:
            return np.zeros((0, 8))
        
        # Convert to datetime
        datetimes = pd.to_datetime(timestamps, unit='s')
        
        # Extract time features
        hours = datetimes.hour / 24.0
        days = datetimes.dayofweek / 7.0
        months = datetimes.month / 12.0
        years = (datetimes.year - 2020) / 10.0  # Normalize to decades since 2020
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hours)
        hour_cos = np.cos(2 * np.pi * hours)
        day_sin = np.sin(2 * np.pi * days)
        day_cos = np.cos(2 * np.pi * days)
        
        # Combine all features
        time_features = np.column_stack([
            hour_sin, hour_cos, day_sin, day_cos,
            months, years,
            hours, days  # Add raw values as well
        ])
        
        return time_features


class HyperTemporalTransactionClassifier:
    """
    Class for training and evaluating a hyper-temporal transaction classifier
    that combines hyperbolic geometry, multi-modal fusion, and dynamic temporal modeling.
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 256, output_dim: int = 400,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.2,
                 use_hyperbolic: bool = True, use_neural_ode: bool = True,
                 use_ensemble: bool = True, ensemble_size: int = 5,
                 max_seq_length: int = 30, lr: float = 2e-4, 
                 weight_decay: float = 1e-5):
        """
        Initialize the hyper-temporal transaction classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            output_dim: Dimension of output features (num classes)
            num_heads: Number of attention heads
            num_layers: Number of model layers
            dropout: Dropout probability
            use_hyperbolic: Whether to use hyperbolic encoding
            use_neural_ode: Whether to use neural ODE layers
            use_ensemble: Whether to use ensemble of models
            ensemble_size: Number of models in the ensemble
            max_seq_length: Maximum sequence length
            lr: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_hyperbolic = use_hyperbolic
        self.use_neural_ode = use_neural_ode
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        self.max_seq_length = max_seq_length
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.graph = None
        self.graph_builder = None
        self.sequence_builder = AdvancedTransactionSequenceBuilder(
            max_seq_length=max_seq_length,
            use_augmentation=True,
            seasonal_encoding=True,
            time_bucketing=True
        )
    
    def prepare_data(self, transactions_df: pd.DataFrame) -> tuple:
        """
        Prepare transaction data by building a graph and extracting sequences.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Tuple of (graph, sequences, lengths, timestamps, edge_time, node_time, time_features)
        """
        # Initialize graph builder
        self.graph_builder = TransactionGraphBuilder()
        
        # Build graph from transaction data
        self.graph = self.graph_builder.build_graph(transactions_df)
        
        # Add self-loops for better information propagation
        for node_type in self.graph.node_types:
            num_nodes = self.graph[node_type].x.size(0)
            self_indices = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([self_indices, self_indices])
            self.graph[f"{node_type}_self_loop"] = edge_index
        
        # Add reverse edges for bidirectional message passing
        src_nodes = self.graph['transaction', 'belongs_to', 'merchant'].edge_index[1]
        dst_nodes = self.graph['transaction', 'belongs_to', 'merchant'].edge_index[0]
        self.graph['merchant', 'rev_belongs_to', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        src_nodes = self.graph['transaction', 'has_category', 'category'].edge_index[1]
        dst_nodes = self.graph['transaction', 'has_category', 'category'].edge_index[0]
        self.graph['category', 'rev_has_category', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Add additional edge types for temporal relationships
        # Transactions from the same user are connected with time-aware edges
        if 'user_id' not in transactions_df.columns:
            # Create user IDs if they don't exist
            num_users = max(100, len(transactions_df) // 30)
            transactions_df = transactions_df.copy()
            transactions_df['user_id'] = np.random.randint(0, num_users, len(transactions_df))
        
        # Create sequential edges between transactions from the same user
        user_groups = transactions_df.groupby('user_id')
        sequential_src = []
        sequential_dst = []
        
        for user_id, group in user_groups:
            # Sort by timestamp
            sorted_group = group.sort_values('timestamp')
            # Get transaction indices
            if len(sorted_group) > 1:
                # Connect each transaction to its next one
                for i in range(len(sorted_group) - 1):
                    sequential_src.append(sorted_group.iloc[i].name)
                    sequential_dst.append(sorted_group.iloc[i + 1].name)
        
        if sequential_src:
            sequential_edge_index = torch.tensor([sequential_src, sequential_dst], dtype=torch.long)
            self.graph['transaction', 'next', 'transaction'].edge_index = sequential_edge_index
        
        # Split graph into train/val/test sets
        self.graph = create_train_val_test_split(self.graph)
        
        # Build user transaction sequences
        sequences, seq_lengths, seq_timestamps, user_ids, time_features = \
            self.sequence_builder.build_sequences(transactions_df)
        
        # Create edge timestamps
        # In a real-world scenario, these would come from the transaction data
        # For simplicity, we'll use real timestamps for edges when available
        edge_timestamps = {}
        edge_time_values = []
        
        # For edges with timestamps, use the actual timestamps
        for edge_type in self.graph.edge_types:
            num_edges = self.graph[edge_type].edge_index.size(1)
            if edge_type[0] == 'transaction' and edge_type[2] == 'transaction':
                # For transaction-to-transaction edges, use sequential timestamps
                src_idx = self.graph[edge_type].edge_index[0]
                dst_idx = self.graph[edge_type].edge_index[1]
                
                if 'timestamp' in transactions_df.columns:
                    src_timestamps = torch.tensor(transactions_df.iloc[src_idx.numpy()]['timestamp'].values)
                    dst_timestamps = torch.tensor(transactions_df.iloc[dst_idx.numpy()]['timestamp'].values)
                    edge_time = dst_timestamps
                else:
                    # If no timestamps, use random values
                    edge_time = torch.randint(0, 1000000, (num_edges,)).float()
                
                edge_timestamps[edge_type] = edge_time
                edge_time_values.append(edge_time)
            else:
                # For other edges, use random timestamps
                if 'timestamp' in transactions_df.columns:
                    # Use realistic timestamps from the transaction range
                    edge_time = torch.tensor(
                        np.random.uniform(
                            transactions_df['timestamp'].min(),
                            transactions_df['timestamp'].max(),
                            num_edges
                        )
                    ).float()
                else:
                    edge_time = torch.randint(0, 1000000, (num_edges,)).float()
                
                edge_timestamps[edge_type] = edge_time
                edge_time_values.append(edge_time)
        
        # Combine all edge timestamps
        all_edge_time = torch.cat(edge_time_values) if edge_time_values else torch.tensor([])
        
        # Create node timestamps
        # For transaction nodes, we'll use the actual transaction timestamps
        # For other nodes, we'll use the median of connected transaction timestamps
        node_timestamps = {}
        for node_type in self.graph.node_types:
            num_nodes = self.graph[node_type].x.size(0)
            if node_type == 'transaction' and 'timestamp' in transactions_df.columns:
                # Use actual transaction timestamps
                node_timestamps[node_type] = torch.tensor(transactions_df['timestamp'].values).float()
            else:
                # Use median of connected transaction timestamps or random values
                node_timestamps[node_type] = torch.randint(0, 1000000, (num_nodes,)).float()
        
        # Combine all node timestamps
        all_node_time = torch.cat([node_timestamps[node_type] for node_type in self.graph.node_types])
        
        # Get min and max timestamps for ODE integration
        if 'timestamp' in transactions_df.columns:
            t0 = float(transactions_df['timestamp'].min())
            t1 = float(transactions_df['timestamp'].max())
        else:
            t0 = 0.0
            t1 = 1000000.0
        
        return (
            self.graph, sequences, seq_lengths, seq_timestamps, 
            all_edge_time, all_node_time, time_features, t0, t1
        )
    
    def initialize_model(self, input_dim: int) -> None:
        """
        Initialize the hyper-temporal transaction model and optimizer.
        
        Args:
            input_dim: Dimension of input features
        """
        # Initialize model
        if self.use_ensemble:
            self.model = HyperTemporalEnsemble(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_models=self.ensemble_size,
                dropout=self.dropout
            )
        else:
            self.model = HyperTemporalTransactionModel(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_hyperbolic=self.use_hyperbolic,
                use_neural_ode=self.use_neural_ode
            )
        
        # Initialize optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def train(self, sequences: torch.Tensor, seq_lengths: torch.Tensor,
              seq_timestamps: torch.Tensor, edge_time: torch.Tensor,
              node_time: torch.Tensor, time_features: torch.Tensor,
              t0: float, t1: float, transaction_descriptions: List[str] = None,
              num_epochs: int = 100, patience: int = 15, warmup_pct: float = 0.1) -> dict:
        """
        Train the hyper-temporal transaction model.
        
        Args:
            sequences: Transaction sequences [batch_size, seq_len, input_dim]
            seq_lengths: Sequence lengths [batch_size]
            seq_timestamps: Sequence timestamps [batch_size, seq_len]
            edge_time: Edge timestamps [num_edges]
            node_time: Node timestamps [num_nodes]
            time_features: Sequence time features [batch_size, seq_len, time_dim]
            t0: Start time for ODE integration
            t1: End time for ODE integration
            transaction_descriptions: List of transaction descriptions
            num_epochs: Maximum number of training epochs
            patience: Number of epochs to wait for validation improvement before early stopping
            warmup_pct: Percentage of training for learning rate warmup
            
        Returns:
            Dictionary containing training and validation metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized before training")
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.graph = self.graph.to(device)
        sequences = sequences.to(device)
        seq_lengths = seq_lengths.to(device)
        seq_timestamps = seq_timestamps.to(device)
        edge_time = edge_time.to(device)
        node_time = node_time.to(device)
        time_features = time_features.to(device)
        
        # Use only transaction node features 
        # Project graph features to same dimension as sequence features for compatibility
        transaction_features = self.graph['transaction'].x
        
        # Get the first sequence's feature dimension as reference
        seq_feat_dim = sequences.size(2)
        graph_feat_dim = transaction_features.size(1)
        
        # Create a simple projection layer for this run
        graph_projection = nn.Linear(graph_feat_dim, seq_feat_dim).to(device)
        
        # Project graph features to sequence feature dimension
        graph_features = graph_projection(transaction_features)
        
        # Extract tabular features (project to same dimension)
        tabular_features = graph_projection(transaction_features.clone())
        
        # Get train/val masks and labels
        train_mask = self.graph['transaction'].train_mask
        val_mask = self.graph['transaction'].val_mask
        y = self.graph['transaction'].y
        
        # Prepare transaction descriptions
        train_descriptions = None
        val_descriptions = None
        
        if transaction_descriptions is not None:
            # Get indices for train and validation
            train_indices = torch.where(train_mask)[0].cpu().numpy()
            val_indices = torch.where(val_mask)[0].cpu().numpy()
            
            # Extract descriptions for train and validation sets
            if len(transaction_descriptions) >= len(train_indices):
                train_descriptions = [transaction_descriptions[i] for i in train_indices]
                val_descriptions = [transaction_descriptions[i] for i in val_indices]
        
        # Initialize OneCycleLR scheduler
        total_steps = num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=warmup_pct,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )
        
        # Initialize metrics
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Mixed precision training
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        graph_features, sequences, tabular_features,
                        seq_timestamps, t0, t1, train_descriptions
                    )
                    
                    # Compute loss
                    train_loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])
                
                # Backward pass with gradient scaling
                scaler.scale(train_loss).backward()
                
                # Gradient clipping
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Standard forward pass
                logits = self.model(
                    graph_features, sequences, tabular_features,
                    seq_timestamps, t0, t1, train_descriptions
                )
                
                # Compute loss
                train_loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])
                
                # Backward pass
                train_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Compute accuracy
            train_acc = self._compute_accuracy(logits[train_mask], y[train_mask])
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                # Forward pass
                val_logits = self.model(
                    graph_features, sequences, tabular_features,
                    seq_timestamps, t0, t1, val_descriptions
                )
                
                # Compute loss and accuracy
                val_loss = torch.nn.functional.cross_entropy(val_logits[val_mask], y[val_mask])
                val_acc = self._compute_accuracy(val_logits[val_mask], y[val_mask])
            
            # Update metrics
            metrics['train_loss'].append(train_loss.item())
            metrics['train_acc'].append(train_acc)
            metrics['val_loss'].append(val_loss.item())
            metrics['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss.item():.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss.item():.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f"Loaded best model from epoch {best_model_state['epoch']+1} "
                  f"with validation accuracy {best_model_state['val_acc']:.4f}")
        
        return metrics
    
    def evaluate(self, sequences: torch.Tensor, seq_lengths: torch.Tensor,
                seq_timestamps: torch.Tensor, edge_time: torch.Tensor,
                node_time: torch.Tensor, time_features: torch.Tensor,
                t0: float, t1: float, transaction_descriptions: List[str] = None) -> dict:
        """
        Evaluate the trained model on the test set.
        
        Args:
            sequences: Transaction sequences [batch_size, seq_len, input_dim]
            seq_lengths: Sequence lengths [batch_size]
            seq_timestamps: Sequence timestamps [batch_size, seq_len]
            edge_time: Edge timestamps [num_edges]
            node_time: Node timestamps [num_nodes]
            time_features: Sequence time features [batch_size, seq_len, time_dim]
            t0: Start time for ODE integration
            t1: End time for ODE integration
            transaction_descriptions: List of transaction descriptions
            
        Returns:
            Dictionary containing test metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized before evaluation")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get device
        device = next(self.model.parameters()).device
        
        # Use only transaction node features 
        # Project graph features to same dimension as sequence features for compatibility
        transaction_features = self.graph['transaction'].x
        
        # Get the first sequence's feature dimension as reference
        seq_feat_dim = sequences.size(2)
        graph_feat_dim = transaction_features.size(1)
        
        # Create a simple projection layer for this run
        graph_projection = nn.Linear(graph_feat_dim, seq_feat_dim).to(device)
        
        # Project graph features to sequence feature dimension
        graph_features = graph_projection(transaction_features)
        
        # Extract tabular features (project to same dimension)
        tabular_features = graph_projection(transaction_features.clone())
        
        # Get test mask and labels
        test_mask = self.graph['transaction'].test_mask
        y = self.graph['transaction'].y
        
        # Prepare test descriptions
        test_descriptions = None
        if transaction_descriptions is not None:
            # Get indices for test set
            test_indices = torch.where(test_mask)[0].cpu().numpy()
            
            # Extract descriptions for test set
            if len(transaction_descriptions) >= len(test_indices):
                test_descriptions = [transaction_descriptions[i] for i in test_indices]
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(
                graph_features, sequences, tabular_features,
                seq_timestamps, t0, t1, test_descriptions
            )
            
            # Compute loss and accuracy
            test_loss = torch.nn.functional.cross_entropy(logits[test_mask], y[test_mask])
            test_acc = self._compute_accuracy(logits[test_mask], y[test_mask])
            
            # Compute additional metrics
            y_pred = torch.argmax(logits[test_mask], dim=1).cpu().numpy()
            y_true = y[test_mask].cpu().numpy()
            
            # Compute top-k accuracy
            top3_acc = self._compute_topk_accuracy(logits[test_mask], y[test_mask], k=3)
            top5_acc = self._compute_topk_accuracy(logits[test_mask], y[test_mask], k=5)
        
        return {
            'test_loss': test_loss.item(),
            'test_acc': test_acc,
            'test_top3_acc': top3_acc,
            'test_top5_acc': top5_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def save_model(self, output_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'use_hyperbolic': self.use_hyperbolic,
                'use_neural_ode': self.use_neural_ode,
                'use_ensemble': self.use_ensemble,
                'ensemble_size': self.ensemble_size if self.use_ensemble else None,
                'max_seq_length': self.max_seq_length
            }
        }, output_path)
    
    def load_model(self, input_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            input_path: Path to the saved model
        """
        # Load model checkpoint
        checkpoint = torch.load(input_path)
        
        # Initialize model with saved configuration
        config = checkpoint['model_config']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_hyperbolic = config['use_hyperbolic']
        self.use_neural_ode = config['use_neural_ode']
        self.use_ensemble = config['use_ensemble']
        self.ensemble_size = config['ensemble_size'] if self.use_ensemble else None
        self.max_seq_length = config['max_seq_length']
        
        # Initialize model
        self.initialize_model(self.input_dim)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _compute_accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute classification accuracy.
        
        Args:
            logits: Model output logits
            y: Ground truth labels
            
        Returns:
            Classification accuracy
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        return correct / total
    
    def _compute_topk_accuracy(self, logits: torch.Tensor, y: torch.Tensor, k: int = 3) -> float:
        """
        Compute top-k classification accuracy.
        
        Args:
            logits: Model output logits
            y: Ground truth labels
            k: Number of top predictions to consider
            
        Returns:
            Top-k classification accuracy
        """
        _, topk_preds = torch.topk(logits, k, dim=1)
        correct = torch.any(topk_preds == y.unsqueeze(1), dim=1).sum().item()
        total = y.size(0)
        return correct / total


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
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/hyper_temporal_training_metrics.png')
    plt.close()


def main():
    """
    Main function for training and evaluating the hyper-temporal transaction classifier.
    """
    print("\n=== State-of-the-Art Hyper-Temporal Transaction Classification with Text Processing ===")
    
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
    
    # Generate synthetic transaction descriptions
    print("\nGenerating synthetic transaction descriptions...")
    transactions_df = generate_synthetic_transaction_descriptions(transactions_df)
    
    print(f"Generated {len(transactions_df)} transactions with {num_merchants} merchants and {num_categories} categories")
    print("\nSample transactions with descriptions:")
    print(transactions_df[['transaction_id', 'merchant_id', 'category_id', 'amount', 'transaction_description']].head())
    
    # Initialize hyper-temporal classifier
    print("\nInitializing hyper-temporal transaction classifier with text processing...")
    classifier = HyperTemporalTransactionClassifier(
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.2,
        use_hyperbolic=True,
        use_neural_ode=True,
        use_ensemble=True,
        ensemble_size=5,
        max_seq_length=30,
        lr=2e-4,
        weight_decay=1e-5
    )
    
    # Prepare data
    print("\nBuilding transaction graph and sequences...")
    (graph, sequences, seq_lengths, seq_timestamps, 
     edge_time, node_time, time_features, t0, t1) = classifier.prepare_data(transactions_df)
    
    # Extract transaction descriptions for processing
    transaction_descriptions = transactions_df['transaction_description'].tolist()
    
    # Get input dimension from sequences
    input_dim = sequences.size(2)
    
    # Get actual number of categories from the graph builder
    num_categories = len(classifier.graph_builder.category_mapping)
    print(f"Actual number of categories in the data: {num_categories}")
    
    # Initialize model with the actual dimensions
    graph_features = transaction_features  # Already retrieved from the graph
    graph_input_dim = graph_features.size(1) if graph_features.dim() == 2 else graph_features.size(2)
    
    # Initialize model with flexible dimensions for handling mismatches
    classifier.initialize_model(input_dim)
    
    # Update the model to handle different input dimensions
    if hasattr(classifier.model, 'graph_input_dim'):
        classifier.model.graph_input_dim = graph_input_dim
        
        # Reinitialize graph projection layers
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier.model.graph_projection = nn.Sequential(
            nn.Linear(graph_input_dim, classifier.hidden_dim),
            nn.LayerNorm(classifier.hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier.dropout)
        ).to(device)
        
        # Add dimension alignment layers
        if not hasattr(classifier.model, 'dim_alignment'):
            classifier.model.dim_alignment = nn.ModuleDict({
                'graph': nn.Linear(graph_input_dim, input_dim),
                'tabular': nn.Linear(input_dim, input_dim)
            }).to(device)
        else:
            classifier.model.dim_alignment['graph'] = nn.Linear(graph_input_dim, input_dim).to(device)
        
    print(f"Model initialized with:")
    print(f"- Sequence input dim: {input_dim}")
    print(f"- Graph input dim: {graph_input_dim}")
    print(f"- Hidden dim: {classifier.hidden_dim}")
    print(f"- Text processing: {'enabled' if classifier.model.use_text_processor else 'disabled'}")
    
    # Print model information
    print("\nHyper-Temporal Model Architecture with Text Processing:")
    print("- Integrated text processing using FinBERT and LLM processors")
    print("- Hyperbolic geometry for hierarchical transaction modeling")
    print("- Multi-modal fusion with cross-attention and gating mechanisms")
    print("- Dynamic contextual temporal layers with multiple time scales")
    print("- Neural ODEs for continuous-time dynamics modeling")
    print("- Advanced ensemble with mixture-of-experts aggregation")
    
    # Train model
    print("\nTraining hyper-temporal model with text processing (this may take a while)...")
    
    # Check shapes of all inputs before training
    print(f"Graph features shape: {graph_features.shape}")
    print(f"Sequence features shape: {sequences.shape}")
    print(f"Tabular features shape: {tabular_features.shape}")
    print(f"Timestamps shape: {seq_timestamps.shape}")
    print(f"Number of descriptions: {len(transaction_descriptions)}")
    
    # Pass proper graph_input_dim if needed
    if hasattr(classifier.model, 'graph_input_dim') and graph_features.size(1) != sequences.size(2):
        print(f"Setting graph_input_dim={graph_features.size(1)} to handle dimension mismatch")
        classifier.model.graph_input_dim = graph_features.size(1)
        
        # Reinitialize graph projection layers with correct dimensions
        classifier.model.graph_projection = nn.Linear(
            graph_features.size(1), classifier.hidden_dim
        ).to(device)
        
        classifier.model.dim_alignment['graph'] = nn.Linear(
            graph_features.size(1), sequences.size(2)
        ).to(device)
    
    metrics = classifier.train(
        sequences, seq_lengths, seq_timestamps, edge_time, node_time, time_features, t0, t1,
        transaction_descriptions=transaction_descriptions,
        num_epochs=100, patience=15, warmup_pct=0.1
    )
    
    # Plot training metrics
    plot_training_metrics(metrics)
    print("Training metrics plotted to 'plots/hyper_temporal_training_metrics.png'")
    
    # Evaluate model
    print("\nEvaluating hyper-temporal model on test set...")
    test_metrics = classifier.evaluate(
        sequences, seq_lengths, seq_timestamps, edge_time, node_time, time_features, t0, t1,
        transaction_descriptions=transaction_descriptions
    )
    print(f"Test Loss: {test_metrics['test_loss']:.4f} | Test Accuracy: {test_metrics['test_acc']:.4f}")
    print(f"Top-3 Accuracy: {test_metrics['test_top3_acc']:.4f} | Top-5 Accuracy: {test_metrics['test_top5_acc']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/hyper_temporal_transaction_model_with_text.pt'
    classifier.save_model(model_path)
    print(f"\nHyper-temporal model with text processing saved to {model_path}")
    
    print("\nAdvantages of the hyper-temporal approach with text processing:")
    print("1. Text understanding from transaction descriptions using FinBERT/LLM processors")
    print("2. Hyperbolic geometry for capturing hierarchical relationships in transaction data")
    print("3. Multi-modal fusion for combining text, graph, sequence, and tabular features")
    print("4. Dynamic contextual temporal modeling for capturing evolving patterns at multiple time scales")
    print("5. Neural ODEs for modeling continuous-time dynamics in transaction sequences")
    print("6. Mixture-of-experts ensemble with diverse text processors for optimal performance")
    
    print("\n=== Hyper-Temporal Transaction Classification with Text Processing Complete ===")


if __name__ == '__main__':
    main()