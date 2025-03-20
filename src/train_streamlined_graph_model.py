#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import custom modules
from src.models.hybrid_transaction_model import EnhancedHybridTransactionModel
from src.data_processing.transaction_graph import build_transaction_relationship_graph, extract_graph_features
from torch.utils.data import Dataset, DataLoader
from src.utils.model_utils import configure_for_hardware, plot_training_curves


# Configuration class
class Config:
    # Data configuration
    data_dir = "/path/to/parquet/files"  # Will be overridden by command line args
    output_dir = "../models/enhanced_model_output"
    batch_size = 64
    num_workers = 4
    prefetch_factor = 2
    max_files = 100  # Max number of parquet files to process
    
    # Model configuration
    hidden_dim = 256
    num_heads = 8
    num_graph_layers = 2
    num_temporal_layers = 2
    dropout = 0.2
    use_hyperbolic = True
    use_neural_ode = False  # Set to False for faster training
    use_text = False  # Set to True if transaction descriptions are available
    multi_task = True
    num_relations = 5  # company, merchant, industry, price, temporal
    
    # Training configuration
    learning_rate = 3e-4
    weight_decay = 1e-5
    num_epochs = 10
    patience = 3
    grad_clip = 0.5  # Reduced from 1.0 for stronger gradient clipping
    
    # GPU optimization
    use_amp = True  # Use mixed precision training
    use_cuda_graphs = True  # Use CUDA graphs for optimization
    cuda_graph_batch_size = None  # Will be set to batch_size
    
    # XGBoost integration
    extract_embeddings = True
    embedding_output_file = "transaction_embeddings.pkl"
    
    # Metrics
    eval_steps = 100
    log_steps = 10


# Define a custom collate function for the DataLoader to handle DataFrames
def df_collate_fn(batch):
    # Just return batch indices
    return list(range(len(batch)))


class ParquetTransactionDataset(Dataset):
    def __init__(self, parquet_files, preprocess_fn=None, transform_fn=None):
        self.parquet_files = parquet_files
        self.preprocess_fn = preprocess_fn
        self.transform_fn = transform_fn
        
        self.file_row_counts = []
        self.total_rows = 0
        
        import pyarrow.parquet as pq
        if not parquet_files:
            print("Warning: No parquet files provided to dataset")
            self.total_rows = 0
            self.lookup = []
            return
            
        for file in tqdm(parquet_files, desc="Counting rows"):
            try:
                metadata = pq.read_metadata(file)
                row_count = metadata.num_rows
                
                if row_count == 0 and metadata.num_row_groups > 0:
                    first_group = metadata.row_group(0)
                    rows_per_group = first_group.num_rows
                    row_count = rows_per_group * metadata.num_row_groups
            except Exception:
                row_count = 200  # Default estimate
                
            self.file_row_counts.append(row_count)
            self.total_rows += row_count
            
        if self.total_rows == 0:
            self.total_rows = 1000
            self.file_row_counts = [1000 // len(parquet_files)] * len(parquet_files)
            
        self.lookup = self._build_lookup()
        print(f"Dataset initialized with {len(parquet_files)} files and {self.total_rows} total rows")
        
    def _build_lookup(self):
        """Build efficient lookup table that maps global index to (file_idx, row_idx)"""
        lookup = [(0, 0)] * self.total_rows
        
        idx = 0
        for file_idx, row_count in enumerate(self.file_row_counts):
            if row_count == 0:  # Skip empty or invalid files
                continue
                
            for row_idx in range(row_count):
                lookup[idx] = (file_idx, row_idx)
                idx += 1
                
        return lookup[:idx]  # Trim in case we skipped some files
        
    def __len__(self):
        return self.total_rows
    
    def _read_row_from_file(self, file_path, row_idx):
        """Read a specific row from a file without loading the entire file"""
        try:
            import pyarrow.parquet as pq
            
            try:
                table = pq.read_table(file_path, use_threads=True)
                df = table.slice(row_idx, 1).to_pandas()
                if not df.empty:
                    return df
            except Exception:
                pass
            
            # Fall back to pandas
            df = pd.read_parquet(file_path)
            if len(df) > row_idx:
                return df.iloc[row_idx:row_idx+1]
            
            raise ValueError(f"Row {row_idx} could not be accessed in {file_path}")
                
        except Exception as e:
            print(f"Error reading row {row_idx} from {file_path}: {str(e)}")
            return pd.DataFrame()
        
    def __getitem__(self, idx):
        """Get a specific item (transaction) from the dataset by index"""
        if idx >= len(self.lookup):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.lookup)} items")
        
        try:    
            file_idx, row_idx = self.lookup[idx]
            
            if file_idx >= len(self.parquet_files):
                return pd.DataFrame()
                
            file_path = self.parquet_files[file_idx]
            df = self._read_row_from_file(file_path, row_idx)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            if self.preprocess_fn:
                df = self.preprocess_fn(df)
            
            if self.transform_fn:
                return self.transform_fn(df)
            else:
                return df
                
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {str(e)}")
            return pd.DataFrame()
        
    def get_sample_batch(self, sample_size=20):
        """Get a small sample batch for metadata and model initialization"""
        if not self.parquet_files:
            raise ValueError("No parquet files available for sampling")
            
        file_path = self.parquet_files[0]
        
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(file_path)
            
            target_cols = ['category_id']
            if 'tax_account_type' in schema.names:
                target_cols.append('tax_account_type')
                
            for extra_col in ['txn_id', 'merchant_id', 'amount']:
                if extra_col in schema.names:
                    target_cols.append(extra_col)
            
            target_cols = target_cols[:5]
            
            table = pq.read_table(file_path, columns=target_cols)
            sample_df = table.slice(0, sample_size).to_pandas()
            
            if self.preprocess_fn:
                sample_df = self.preprocess_fn(sample_df)
            
            return sample_df
                
        except Exception:
            sample_df = pd.read_parquet(file_path, engine='pyarrow')
            sample_df = sample_df.head(sample_size)
            return sample_df
        
    def get_batch_df(self, indices):
        """Get a batch of rows as a single DataFrame"""
        if not indices:
            raise ValueError("Empty indices list provided to get_batch_df")
        
        dfs = []
        
        # Group indices by file to minimize file open/close operations
        file_indices = {}
        for idx in indices:
            if idx >= len(self.lookup):
                continue
                
            file_idx, row_idx = self.lookup[idx]
            if file_idx not in file_indices:
                file_indices[file_idx] = []
            file_indices[file_idx].append((idx, row_idx))
        
        # Process each file once instead of once per row
        for file_idx, row_data in file_indices.items():
            if file_idx >= len(self.parquet_files):
                continue
                
            file_path = self.parquet_files[file_idx]
            try:
                # Sort row indices to improve access patterns
                row_data.sort(key=lambda x: x[1])
                row_indices = [r[1] for r in row_data]
                
                for row_idx in row_indices:
                    df = self._read_row_from_file(file_path, row_idx)
                    if not df.empty:
                        if self.preprocess_fn:
                            df = self.preprocess_fn(df)
                        dfs.append(df)
                        
            except Exception:
                continue
        
        # Check if we got any valid data
        if not dfs:
            raise ValueError("All dataframes in batch were empty")
        
        # Combine all dataframes and reset index
        try:
            result = pd.concat(dfs, ignore_index=True)
            return result
        except Exception:
            # Last resort: if concat fails, return the first valid DataFrame
            return dfs[0]


def get_parquet_files(data_dir, max_files=None):
    """Get list of parquet files from directory, filtering for transaction data"""
    all_parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    transaction_data_files = [f for f in all_parquet_files if 'transaction_data_batch' in os.path.basename(f)]
    
    if not transaction_data_files:
        print("Warning: No transaction_data_batch files found, using all parquet files")
        parquet_files = all_parquet_files
    else:
        print(f"Using {len(transaction_data_files)} transaction_data_batch files")
        parquet_files = transaction_data_files
    
    # Apply max_files limit if specified
    if max_files is not None and len(parquet_files) > max_files:
        parquet_files = parquet_files[:max_files]
        
    print(f"Selected {len(parquet_files)} parquet files")
    return parquet_files


def preprocess_transactions(df):
    """Preprocess transaction DataFrame for model input"""
    # Ensure required columns are present
    required_columns = ['company_id', 'merchant_id', 'amount', 'category_id']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in dataset")
    
    # Handle timestamps
    timestamp_cols = ['timestamp', 'review_timestamp', 'update_timestamp', 
                       'books_create_timestamp', 'partition_timestamp', 
                       'generated_timestamp']
    
    for col in timestamp_cols:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    elif 'generated_timestamp' in df.columns:
        df = df.sort_values('generated_timestamp')
    
    return df


def prepare_model_inputs(batch_df, model, device):
    """Prepare model inputs from a batch DataFrame"""
    if batch_df.empty:
        raise ValueError("Empty DataFrame provided to prepare_model_inputs")
    
    # Use the model's data preparation function
    data = model.prepare_data_from_dataframe(batch_df)
    
    # Move tensors to device
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device, non_blocking=True)
        elif isinstance(value, np.ndarray):
            data[key] = torch.from_numpy(value).to(device, non_blocking=True)
    
    # Check for required label column
    if 'category_id' not in batch_df.columns:
        raise ValueError("Required column 'category_id' not found in batch DataFrame")
    
    # Prepare labels dictionary
    labels = {}
    
    # Handle category_id
    category_values = batch_df['category_id'].values
    
    # Convert to numeric if needed
    if pd.api.types.is_object_dtype(category_values):
        category_codes, _ = pd.factorize(batch_df['category_id'])
        category_values = category_codes
    
    category_tensor = torch.tensor(
        category_values, 
        dtype=torch.long,
        device=device
    )
    labels['category'] = category_tensor
    
    # Add tax_account_type if available
    if 'tax_account_type' in batch_df.columns:
        tax_type_values = batch_df['tax_account_type'].values
        
        if pd.api.types.is_object_dtype(tax_type_values):
            tax_type_codes, _ = pd.factorize(batch_df['tax_account_type'])
            tax_type_values = tax_type_codes
        
        tax_type_tensor = torch.tensor(
            tax_type_values,
            dtype=torch.long,
            device=device
        )
        labels['tax_type'] = tax_type_tensor
    
    return data, labels


def create_cuda_graph(model, sample_data, config, device):
    """Create CUDA graph for model inference"""
    # if not torch.cuda.is_available() or not config.use_cuda_graphs:
    #     return None
    
    print("Creating CUDA graph for optimized inference...")
    
    required_inputs = [
        'x', 'edge_index', 'edge_type', 'edge_attr', 'seq_features',
        'timestamps', 'tabular_features', 't0', 't1', 
        'company_features', 'company_ids', 'batch_size', 'seq_len'
    ]
    
    # Filter and extract only required inputs
    inputs = {}
    for key in required_inputs:
        if key in sample_data:
            inputs[key] = sample_data[key]
    
    # Move tensors to device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, non_blocking=True)
    
    # Create static input batch
    static_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            static_inputs[k] = torch.zeros_like(v, device=device, dtype=v.dtype)
            static_inputs[k].copy_(v)
        else:
            static_inputs[k] = v

    # Set model to eval mode for graph capture
    model.eval()
    
    # Warm up
    for _ in range(3):
        model(**static_inputs)
        torch.cuda.synchronize()
    
    # Capture graph
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_outputs = model(**static_inputs)
        
        return {
            'graph': g,
            'static_inputs': static_inputs,
            'static_outputs': static_outputs
        }
    except Exception as e:
        print(f"Error creating CUDA graph: {str(e)}")
        return None


def run_with_cuda_graph(cuda_graph, data):
    """Run inference using CUDA graph"""
    if cuda_graph is None:
        return None
    
    # Update static inputs with new data
    for k, v in data.items():
        if k in cuda_graph['static_inputs'] and isinstance(v, torch.Tensor):
            if isinstance(cuda_graph['static_inputs'][k], torch.Tensor):
                if cuda_graph['static_inputs'][k].shape != v.shape:
                    return None  # Shape mismatch, can't use graph
                cuda_graph['static_inputs'][k].copy_(v, non_blocking=True)
    
    # Run the graph
    cuda_graph['graph'].replay()
    torch.cuda.synchronize()
    
    # Return cached outputs
    return cuda_graph['static_outputs']


def initialize_model(hidden_dim, num_categories, num_tax_types, config, device):
    """Initialize the EnhancedHybridTransactionModel"""
    model = EnhancedHybridTransactionModel(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        output_dim=num_categories,
        num_heads=config.num_heads,
        num_graph_layers=config.num_graph_layers,
        num_temporal_layers=config.num_temporal_layers,
        dropout=config.dropout,
        use_hyperbolic=config.use_hyperbolic,
        use_neural_ode=config.use_neural_ode,
        use_text=config.use_text,
        multi_task=config.multi_task,
        tax_type_dim=num_tax_types,
        num_relations=config.num_relations,
        graph_weight=0.6,
        temporal_weight=0.4,
        use_dynamic_weighting=True
    ).to(device)
    
    return model


def evaluate(model, dataloader, dataset, device, config, cuda_graph=None):
    """Evaluate the model on the given dataset"""
    model.eval()
    
    # Initialize metrics
    total_loss = 0
    category_correct = 0
    category_total = 0
    all_preds = []
    all_labels = []
    
    # Evaluate without gradients
    with torch.no_grad():
        for batch_indices in tqdm(dataloader, desc="Evaluation", leave=False):
            if not batch_indices:
                continue
                
            # Extract actual indices for this batch
            start_idx = batch_indices[0]
            end_idx = start_idx + len(batch_indices)
            actual_indices = list(range(start_idx, min(end_idx, len(dataset))))
            
            if not actual_indices:
                continue
            
            # Get batch dataframe
            batch_df = dataset.get_batch_df(actual_indices)
            batch_size = len(batch_df)
            
            # Prepare inputs
            data, labels = prepare_model_inputs(batch_df, model, device)
            
            # Replace NaN/Inf in input tensors for stability
            for key, tensor in data.items():
                if isinstance(tensor, torch.Tensor) and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                    data[key] = torch.nan_to_num(tensor, nan=0.0)
            
            # Forward pass
            if cuda_graph is not None and config.use_cuda_graphs:
                outputs = run_with_cuda_graph(cuda_graph, data)
                if outputs is None:  # Fall back if graph fails
                    outputs = model(**data)
            else:
                outputs = model(**data)
            
            # Handle multi-task vs single-task output
            if isinstance(outputs, tuple):
                # Multi-task model (category and tax type)
                category_logits, _ = outputs
                
                # Check and sanitize outputs if they contain NaN/Inf
                if torch.isnan(category_logits).any() or torch.isinf(category_logits).any():
                    category_logits = torch.nan_to_num(category_logits, nan=0.0)
                
                # Calculate loss with sanitized inputs
                category_loss = nn.CrossEntropyLoss(reduction='sum')(category_logits, labels['category'])
                
                # Get predictions
                category_preds = category_logits.argmax(dim=1)
                correct = (category_preds == labels['category']).sum().item()
                
                # Track metrics
                category_correct += correct
                category_total += batch_size
                
                # Store predictions for metrics
                all_preds.extend(category_preds.cpu().numpy())
                all_labels.extend(labels['category'].cpu().numpy())
                
                loss = category_loss
                
            else:
                # Single task model
                # Check and sanitize outputs if they contain NaN/Inf
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    outputs = torch.nan_to_num(outputs, nan=0.0)
                    
                loss = nn.CrossEntropyLoss(reduction='sum')(outputs, labels['category'])
                
                # Get predictions
                category_preds = outputs.argmax(dim=1)
                correct = (category_preds == labels['category']).sum().item()
                
                # Track metrics
                category_correct += correct
                category_total += batch_size
                
                # Store predictions for metrics
                all_preds.extend(category_preds.cpu().numpy())
                all_labels.extend(labels['category'].cpu().numpy())
            
            # Update total loss
            total_loss += loss.item()
    
    # Calculate final metrics
    avg_loss = total_loss / max(1, category_total)
    category_accuracy = category_correct / max(1, category_total)
    
    # Calculate F1 score
    if len(all_preds) > 1 and len(all_labels) > 1:
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"F1 Score (weighted): {f1:.4f}")
    
    return avg_loss, category_accuracy


def train(model, train_dataset, val_dataset, config, device):
    """Train the model"""
    print(f"\n{'='*80}\nStarting Training\n{'='*80}")
    print(f"Batch size: {config.batch_size}")
    print(f"Hidden dimension: {config.hidden_dim}")
    # Create optimizer with improved numerical stability settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8,  # Increased epsilon for better numerical stability
        amsgrad=True  # Enable AMSGrad variant for better convergence
    )
    
    # Learning rate scheduler with warmup for stability
    # First create the base scheduler
    base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=config.patience // 2,
        verbose=True,
        min_lr=1e-6  # Set minimum learning rate
    )
    
    # Implement a simple warmup wrapper for the scheduler
    class WarmupScheduler:
        def __init__(self, optimizer, base_scheduler, warmup_steps=100, init_lr=1e-5):
            self.optimizer = optimizer
            self.base_scheduler = base_scheduler
            self.warmup_steps = warmup_steps
            self.init_lr = init_lr
            self.step_count = 0
            self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
            
        def step(self, val_loss=None):
            self.step_count += 1
            
            # Apply warmup
            if self.step_count <= self.warmup_steps:
                # Linearly increase learning rate
                for i, pg in enumerate(self.optimizer.param_groups):
                    progress = self.step_count / self.warmup_steps
                    target_lr = self.base_lrs[i]
                    current_lr = self.init_lr + (target_lr - self.init_lr) * progress
                    pg['lr'] = current_lr
            else:
                # Use base scheduler
                if val_loss is not None:
                    self.base_scheduler.step(val_loss)
    
    # Create warmup scheduler
    scheduler = WarmupScheduler(
        optimizer, 
        base_scheduler, 
        warmup_steps=100,
        init_lr=config.learning_rate * 0.1
    )
    
    # Setup mixed precision training if enabled
    scaler = GradScaler(enabled=config.use_amp) if torch.cuda.is_available() else None
    
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        collate_fn=df_collate_fn
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        collate_fn=df_collate_fn
    )
    
    # Setup training state tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Metrics history
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'learning_rates': []
    }
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize CUDA graph to None (will create later)
    cuda_graph = None
    
    # We'll implement a simpler fix for NaN values by registering a hook for all modules
    # This avoids dictionary iteration errors while still providing protection against NaNs
    def register_nan_hooks(model):
        """Register hooks to handle NaN values in module outputs"""
        handles = []
        
        def sanitize_hook(module, input, output):
            # Apply NaN cleaning to the output
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
            elif isinstance(output, tuple):
                # Handle tuple outputs (like in multi-task models)
                sanitized_output = []
                for tensor in output:
                    if isinstance(tensor, torch.Tensor):
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            sanitized_output.append(torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4))
                        else:
                            sanitized_output.append(tensor)
                    else:
                        sanitized_output.append(tensor)
                return tuple(sanitized_output)
            return output
        
        # Register hook for all modules
        for module in model.modules():
            handle = module.register_forward_hook(sanitize_hook)
            handles.append(handle)
        
        # Store handles to prevent garbage collection
        model._hook_handles = handles
        return model
    
    # Register NaN sanitization hooks on all modules
    register_nan_hooks(model)
    
    # Define loss function with label smoothing for better stability
    def create_stable_cross_entropy(label_smoothing=0.1, reduction='mean'):
        """Create a cross entropy loss function with label smoothing and stability measures"""
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)
        
        def stable_loss_fn(logits, targets):
            # Sanitize inputs - replace NaNs and Infs
            sanitized_logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Clamp values to prevent extreme outputs
            sanitized_logits = torch.clamp(sanitized_logits, min=-1e4, max=1e4)
            
            # Apply log softmax with improved numerical stability
            log_probs = F.log_softmax(sanitized_logits, dim=-1)
            
            # Compute loss with better numerical stability
            try:
                loss = loss_fn(sanitized_logits, targets)
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    # Fallback to a simpler NLL loss if cross entropy produces NaN
                    nll_loss = F.nll_loss(log_probs, targets, reduction=reduction)
                    return nll_loss
                return loss
            except Exception:
                # Fallback computation of cross entropy
                one_hot = F.one_hot(targets, num_classes=sanitized_logits.size(-1)).float()
                if label_smoothing > 0:
                    # Apply label smoothing
                    smooth_targets = one_hot * (1 - label_smoothing) + label_smoothing / sanitized_logits.size(-1)
                    loss = -(smooth_targets * log_probs).sum(dim=-1)
                else:
                    loss = -(one_hot * log_probs).sum(dim=-1)
                
                if reduction == 'mean':
                    return loss.mean()
                elif reduction == 'sum':
                    return loss.sum()
                return loss
                
        return stable_loss_fn
    
    # Create stable loss function
    category_loss_fn = create_stable_cross_entropy(label_smoothing=0.1)
    tax_type_loss_fn = create_stable_cross_entropy(label_smoothing=0.1)
    
    # Function to detect and handle NaN gradients during backward pass
    def check_and_handle_nan_grads(model):
        """Check for NaN gradients and handle them appropriately"""
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
                    # Replace NaN/Inf gradients with zeros
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        return has_nan
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Set model to training mode
        model.train()
        
        # Reset epoch metrics
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        # Progress bar for this epoch
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        # Process each batch
        for step, batch_indices in pbar:
            if not batch_indices:
                continue
                
            # Get batch dataframe
            start_idx = batch_indices[0] 
            end_idx = start_idx + len(batch_indices)
            actual_indices = list(range(start_idx, min(end_idx, len(train_dataset))))
            
            if not actual_indices:
                continue
            
            batch_df = train_dataset.get_batch_df(actual_indices)
            batch_size = len(batch_df)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Prepare inputs
            data, labels = prepare_model_inputs(batch_df, model, device)
            
            # Prevent NaNs in input data
            for key, tensor in data.items():
                if isinstance(tensor, torch.Tensor):
                    # Check for and replace NaN values
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        data[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Forward and backward pass with mixed precision if enabled
            try:
                if config.use_amp and scaler is not None:
                    with autocast(enabled=config.use_amp):
                        outputs = model(**data)
                        
                        # Calculate loss
                        if isinstance(outputs, tuple):
                            category_logits, tax_type_logits = outputs
                            
                            # Check for NaNs in outputs
                            if torch.isnan(category_logits).any() or torch.isinf(category_logits).any():
                                category_logits = torch.nan_to_num(category_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                            
                            # Stabilized loss calculation
                            category_loss = category_loss_fn(category_logits, labels['category'])
                            
                            if 'tax_type' in labels:
                                if torch.isnan(tax_type_logits).any() or torch.isinf(tax_type_logits).any():
                                    tax_type_logits = torch.nan_to_num(tax_type_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                                tax_type_loss = tax_type_loss_fn(tax_type_logits, labels['tax_type'])
                                # Use dynamic loss weighting to avoid one component dominating
                                loss = 0.7 * category_loss + 0.3 * tax_type_loss
                            else:
                                loss = category_loss
                                
                            # Get predictions
                            category_preds = category_logits.argmax(dim=1)
                            
                        else:
                            # Check for NaNs in outputs
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
                            
                            # Stabilized loss calculation
                            loss = category_loss_fn(outputs, labels['category'])
                            category_preds = outputs.argmax(dim=1)
                        
                        # Check if loss is NaN and handle it
                        if torch.isnan(loss) or torch.isinf(loss):
                            # Fallback to a simple one-hot loss computation
                            print(f"WARNING: NaN loss detected at step {step}. Using fallback loss.")
                            # Skip this batch if we can't compute a valid loss
                            continue
                        
                        # Calculate accuracy
                        correct = (category_preds == labels['category']).sum().item()
                    
                    # Scale loss and backward with additional checks
                    scaler.scale(loss).backward()
                    
                    # Check for NaN gradients
                    has_nan_grads = check_and_handle_nan_grads(model)
                    if has_nan_grads:
                        print(f"WARNING: NaN gradients detected at step {step}. Gradients sanitized.")
                    
                    # Apply gradient clipping to scaled gradients with stronger clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=False)
                    
                    # Step optimizer and update scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    outputs = model(**data)
                    
                    # Calculate loss
                    if isinstance(outputs, tuple):
                        category_logits, tax_type_logits = outputs
                        
                        # Check for NaNs in outputs
                        if torch.isnan(category_logits).any() or torch.isinf(category_logits).any():
                            category_logits = torch.nan_to_num(category_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                        
                        # Stabilized loss calculation
                        category_loss = category_loss_fn(category_logits, labels['category'])
                        
                        if 'tax_type' in labels:
                            if torch.isnan(tax_type_logits).any() or torch.isinf(tax_type_logits).any():
                                tax_type_logits = torch.nan_to_num(tax_type_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                            tax_type_loss = tax_type_loss_fn(tax_type_logits, labels['tax_type'])
                            # Use dynamic loss weighting to avoid one component dominating
                            loss = 0.7 * category_loss + 0.3 * tax_type_loss
                        else:
                            loss = category_loss
                            
                        # Get predictions
                        category_preds = category_logits.argmax(dim=1)
                        
                    else:
                        # Check for NaNs in outputs
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
                        
                        # Stabilized loss calculation
                        loss = category_loss_fn(outputs, labels['category'])
                        category_preds = outputs.argmax(dim=1)
                    
                    # Check if loss is NaN and handle it
                    if torch.isnan(loss) or torch.isinf(loss):
                        # Skip this batch if we can't compute a valid loss
                        print(f"WARNING: NaN loss detected at step {step}. Skipping batch.")
                        continue
                    
                    # Calculate accuracy
                    correct = (category_preds == labels['category']).sum().item()
                    
                    # Standard backward and optimize
                    loss.backward()
                    
                    # Check for NaN gradients
                    has_nan_grads = check_and_handle_nan_grads(model)
                    if has_nan_grads:
                        print(f"WARNING: NaN gradients detected at step {step}. Gradients sanitized.")
                    
                    # Apply gradient clipping with stronger clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=False)
                    
                    # Update weights
                    optimizer.step()
            except RuntimeError as e:
                print(f"Error during training at step {step}: {str(e)}")
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    print("Dimension mismatch error detected. Skipping this batch.")
                    continue
                else:
                    # For other errors, print details and continue with next batch
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Update metrics
            epoch_loss += loss.item() * batch_size
            epoch_correct += correct
            epoch_samples += batch_size
            
            # Update progress bar
            if epoch_samples > 0:
                train_loss = epoch_loss / epoch_samples
                train_acc = epoch_correct / epoch_samples
                
                pbar.set_description(
                    f"Train Loss: {train_loss:.4f}, "
                    f"Acc: {train_acc:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Periodic validation during training
            if step > 0 and step % config.eval_steps == 0:
                print(f"\nPerforming validation at step {step}/{len(train_loader)}")
                model.eval()
                # Initialize CUDA graph for inference if enabled
                if config.use_cuda_graphs and cuda_graph is None and torch.cuda.is_available():
                #config.cuda_graph_batch_size = len(batch_df)
                    cuda_graph = create_cuda_graph(model, data, config, device)
                    
                # Run evaluation
                #model.eval()
                val_loss, val_acc = evaluate(model, val_loader, val_dataset, device, config, cuda_graph)
                model.train()
                
                # Print validation results
                print(f"Step {step}/{len(train_loader)}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        # End of epoch calculations
        train_loss = epoch_loss / max(1, epoch_samples)
        train_acc = epoch_correct / max(1, epoch_samples)
        
        # Full validation at end of epoch
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, val_dataset, device, config, cuda_graph)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['train_accs'].append(train_acc)
        metrics['val_accs'].append(val_acc)
        metrics['learning_rates'].append(current_lr)
        
        # Print epoch summary
        lr_info = f", LR: {current_lr:.6f}" + (f" → {new_lr:.6f}" if new_lr != current_lr else "")
        print(f"\nEpoch {epoch + 1}/{config.num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}{lr_info}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config.__dict__
                },
                os.path.join(config.output_dir, 'best_model.pt')
            )
            
            print(f"✓ Saved new best model at epoch {epoch + 1} with validation loss {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            
            # Check for early stopping
            if patience_counter >= config.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs without improvement.")
                break
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == config.num_epochs - 1:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                },
                checkpoint_path
            )
        
        # Plot training curves
        if (epoch + 1) % 5 == 0:
            try:
                plot_training_curves(metrics, config, epoch + 1)
            except Exception:
                pass
    
    print(f"\n{'='*80}\nTraining completed\n{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
    
    return metrics


def extract_embeddings_for_xgboost(model, dataset, output_file, config, device):
    """Extract node embeddings for XGBoost integration"""
    model.eval()
    
    embeddings_list = []
    labels_list = []
    transaction_ids = []
    
    # Create dataloader
    batch_size = min(config.batch_size, 32)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=df_collate_fn
    )
    
    with torch.no_grad():
        for batch_indices in tqdm(dataloader, desc="Extracting embeddings"):
            try:
                if not batch_indices:
                    continue
                    
                start_idx = batch_indices[0]
                end_idx = start_idx + len(batch_indices)
                actual_indices = list(range(start_idx, min(end_idx, len(dataset))))
                
                # Get batch dataframe
                batch_df = dataset.get_batch_df(actual_indices)
                
                # Prepare inputs
                data, labels = prepare_model_inputs(batch_df, model, device)
                
                # Extract embeddings
                embeddings = model.extract_embeddings(data)
                
                # Store embeddings and labels
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels['category'].cpu().numpy())
                
                if 'txn_id' in batch_df.columns:
                    transaction_ids.extend(batch_df['txn_id'].tolist())
            except Exception:
                continue
    
    if not embeddings_list:
        print("No embeddings were extracted.")
        return None
    
    # Concatenate embeddings and labels
    embeddings_array = np.vstack(embeddings_list)
    labels_array = np.concatenate(labels_list)
    
    # Create DataFrame with embeddings
    embeddings_df = pd.DataFrame(embeddings_array)
    embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_array.shape[1])]
    
    # Add labels
    embeddings_df['category_id'] = labels_array
    
    # Add transaction IDs if available
    if transaction_ids:
        embeddings_df['txn_id'] = transaction_ids
    
    # Save to file
    output_path = os.path.join(config.output_dir, output_file)
    embeddings_df.to_pickle(output_path)
    
    print(f"Saved {len(embeddings_df)} embeddings to {output_path}")
    return embeddings_df


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Enhanced Hybrid Transaction Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing parquet files')
    data_group.add_argument('--output_dir', type=str, default='./models/enhanced_model_output',
                        help='Directory to save model outputs')
    data_group.add_argument('--max_files', type=int, default=100,
                        help='Maximum number of parquet files to process')
    data_group.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension size (auto-configured if not specified)')
    model_group.add_argument('--num_heads', type=int, default=None,
                        help='Number of attention heads')
    model_group.add_argument('--num_graph_layers', type=int, default=2,
                        help='Number of graph neural network layers')
    model_group.add_argument('--num_temporal_layers', type=int, default=2,
                        help='Number of temporal layers')
    model_group.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    model_group.add_argument('--no_hyperbolic', action='store_false', dest='use_hyperbolic',
                        help='Disable hyperbolic encoding')
    model_group.add_argument('--use_neural_ode', action='store_true',
                        help='Enable neural ODE for continuous-time modeling')
    model_group.add_argument('--use_text', action='store_true',
                        help='Enable text processing if descriptions are available')
    
    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    train_group.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    train_group.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    train_group.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # Hardware/optimization configuration
    hw_group = parser.add_argument_group('Hardware and Optimization')
    hw_group.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable automatic mixed precision')
    hw_group.add_argument('--no_cuda_graphs', action='store_false', dest='use_cuda_graphs',
                        help='Disable CUDA graphs optimization')
    hw_group.add_argument('--num_workers', type=int, default=None,
                        help='Number of dataloader workers (auto-configured if not specified)')
    hw_group.add_argument('--cpu_only', action='store_true',
                        help='Force CPU usage even if GPU is available')
    hw_group.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluate every N steps')
    
    # Additional features
    features_group = parser.add_argument_group('Additional Features')
    features_group.add_argument('--extract_embeddings', action='store_true',
                        help='Extract embeddings for XGBoost integration')
    features_group.add_argument('--embedding_output', type=str, default='transaction_embeddings.pkl',
                        help='File to save extracted embeddings')
    features_group.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Force CPU if requested
    if args.cpu_only and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    return args


def main():
    """Main training script entry point"""
    print(f"\n{'='*80}")
    print(f"Enhanced Transaction Graph Model Training Script")
    print(f"{'='*80}")
    
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration with base settings
    config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_files:
        config.max_files = args.max_files
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.num_graph_layers:
        config.num_graph_layers = args.num_graph_layers
    if args.num_temporal_layers:
        config.num_temporal_layers = args.num_temporal_layers
    if args.dropout:
        config.dropout = args.dropout
    
    config.use_hyperbolic = args.use_hyperbolic
    if args.use_neural_ode:
        config.use_neural_ode = True
    if args.use_text:
        config.use_text = True
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.patience:
        config.patience = args.patience
    if args.grad_clip:
        config.grad_clip = args.grad_clip
    
    config.use_amp = args.use_amp
    config.use_cuda_graphs = args.use_cuda_graphs
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.eval_steps:
        config.eval_steps = args.eval_steps
    if args.extract_embeddings:
        config.extract_embeddings = True
    if args.embedding_output:
        config.embedding_output_file = args.embedding_output
    
    # Configure for available hardware
    device, config = configure_for_hardware(config)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, 'training_config.txt')
    with open(config_path, 'w') as f:
        f.write("Enhanced Transaction Graph Model Training Configuration\n")
        f.write("="*50 + "\n")
        for key, value in sorted(vars(config).items()):
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")
    
    # Get parquet files
    parquet_files = get_parquet_files(config.data_dir, config.max_files)
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {config.data_dir}")
    
    # Split files into train and validation sets
    train_files, val_files = train_test_split(
        parquet_files, 
        test_size=args.test_split, 
        random_state=args.seed
    )
    
    print(f"Training on {len(train_files)} files, validating on {len(val_files)} files")
    
    # Create datasets
    train_dataset = ParquetTransactionDataset(train_files, preprocess_fn=preprocess_transactions)
    val_dataset = ParquetTransactionDataset(val_files, preprocess_fn=preprocess_transactions)
    
    print(f"Train dataset: {len(train_dataset):,} transactions")
    print(f"Validation dataset: {len(val_dataset):,} transactions")
    
    # Default values for model initialization
    num_categories = 400
    num_tax_account_types = 20
    
    # Initialize model
    model = initialize_model(config.hidden_dim, num_categories, num_tax_account_types, config, device)
    
    # Train the model
    metrics = train(model, train_dataset, val_dataset, config, device)
    
    # Extract embeddings if requested
    if config.extract_embeddings:
        print("\nExtracting Embeddings for XGBoost Integration")
        # Skip loading best model - use current model instead for extract_embeddings
        extract_embeddings_for_xgboost(
            model, 
            train_dataset, 
            config.embedding_output_file, 
            config, 
            device
        )
    
    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    main()