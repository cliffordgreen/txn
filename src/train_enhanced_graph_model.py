#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Hybrid Transaction Model Training Script

This script implements the training process for the EnhancedHybridTransactionModel,
which combines:
1. Graph-based transaction relationships (company, merchant, industry, price)
2. Temporal patterns with company-based grouping
3. Hyperbolic encoding for hierarchical relationships

Optimized for p3.2xlarge AWS instances with V100 GPU acceleration.

Usage:
    python train_enhanced_graph_model.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
"""

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
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Memory monitoring functions
def get_memory_usage():
    """Return current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # If psutil is not available, return -1
        return -1

def log_memory_usage(msg=""):
    """Log memory usage for debugging"""
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    current_usage = get_memory_usage()
    gpu_memory = ""
    if torch.cuda.is_available():
        gpu_memory = f", GPU: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB"
    
    print(f"MEMORY [{msg}]: RAM: {current_usage:.1f} MB{gpu_memory}")
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
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
    grad_clip = 1.0
    
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
        
        # More memory-efficient counting of rows
        self.file_row_counts = []
        self.total_rows = 0
        
        import pyarrow.parquet as pq
        # Default to a reasonable number if counting fails
        if not parquet_files:
            print("Warning: No parquet files provided to dataset")
            self.total_rows = 0
            self.lookup = []
            return
            
        # Estimate row counts to avoid loading entire files
        for file in tqdm(parquet_files, desc="Counting rows"):
            try:
                try:
                    # Most memory-efficient: use PyArrow to read only metadata
                    metadata = pq.read_metadata(file)
                    row_count = metadata.num_rows
                    
                    # Handle case where metadata doesn't have row count
                    if row_count == 0:
                        # Try reading just the first row group's metadata
                        if metadata.num_row_groups > 0:
                            first_group = metadata.row_group(0)
                            rows_per_group = first_group.num_rows
                            row_count = rows_per_group * metadata.num_row_groups
                except Exception as e:
                    # If metadata approach fails, use a fixed sample size to estimate
                    print(f"Metadata approach failed for {file}: {str(e)}")
                    # Assume a reasonable default row count for initialization
                    # We'll set a fixed count per file to avoid memory issues
                    if 'transaction_data_batch' in file:
                        # Training files tend to have ~500 rows each
                        row_count = 500
                    else:
                        # Smaller for other files
                        row_count = 200
                    print(f"Estimating {row_count} rows for {os.path.basename(file)}")
            except Exception as e:
                print(f"Warning: Could not determine row count for {file}: {str(e)}")
                print(f"Using default row count")
                row_count = 100  # Small default to prevent memory issues
            
            self.file_row_counts.append(row_count)
            self.total_rows += row_count
            
        # If total rows is still 0, set a reasonable default
        if self.total_rows == 0:
            print("Warning: Could not determine row count for any files. Using default.")
            self.total_rows = 1000
            self.file_row_counts = [1000 // len(parquet_files)] * len(parquet_files)
            
        # Build lookup table more efficiently using _build_lookup
        self.lookup = self._build_lookup()
        
        print(f"Dataset initialized with {len(parquet_files)} files and {self.total_rows} total rows")
        
    def _build_lookup(self):
        """Build efficient lookup table that maps global index to (file_idx, row_idx)"""
        lookup = []
        # Pre-allocate the full list size for better performance
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
            # Try using pyarrow for more efficient row access
            import pyarrow.parquet as pq
            
            # First approach: Try to read specific row ranges with pyarrow (most efficient)
            try:
                table = pq.read_table(file_path, use_threads=True)
                df = table.slice(row_idx, 1).to_pandas()
                if not df.empty:
                    return df
            except Exception as slice_error:
                # If slice method fails, continue to next approach
                pass
            
            # Second approach: Use pandas with iloc for specific row access
            try:
                # Read with pyarrow engine and use direct row indexing
                table = pd.read_parquet(file_path, engine='pyarrow')
                row = table.iloc[row_idx:row_idx+1]
                
                if not row.empty:
                    return row
            except Exception as iloc_error:
                # If iloc fails, continue to next approach
                pass
                
            # Third approach: Try to read with row filters if 'index' column exists
            # This can be much faster for large files where we need specific rows
            try:
                # Check if file has an index column we can filter on
                metadata = pq.read_metadata(file_path)
                schema = metadata.schema.to_arrow_schema()
                
                # Look for possible index columns
                index_names = ['index', 'idx', 'row_idx', 'id', 'txn_id', 'transaction_id']
                idx_col = None
                
                for name in index_names:
                    if name in schema.names:
                        idx_col = name
                        break
                
                if idx_col:
                    # Use filter to read only this specific row
                    filters = [(idx_col, '==', row_idx)]
                    df = pd.read_parquet(file_path, filters=filters)
                    if not df.empty:
                        return df
            except Exception as filter_error:
                # If filter approach fails, continue to last resort
                pass
                
            # Last resort: Read the whole file and extract the row
            # This is inefficient but serves as a fallback
            df = pd.read_parquet(file_path)
            if len(df) > row_idx:
                return df.iloc[row_idx:row_idx+1]
            
            # If we get here, we couldn't find the row
            raise ValueError(f"Row {row_idx} could not be accessed in {file_path}")
                
        except Exception as e:
            print(f"Error reading row {row_idx} from {file_path}: {str(e)}")
            # Return empty DataFrame with proper error logging
            return pd.DataFrame()
        
    def __getitem__(self, idx):
        """Get a specific item (transaction) from the dataset by index"""
        # Check if index is within bounds
        if idx >= len(self.lookup):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.lookup)} items")
        
        try:    
            # Get file and row indices from lookup table
            file_idx, row_idx = self.lookup[idx]
            
            # Skip invalid file indices
            if file_idx >= len(self.parquet_files):
                print(f"Warning: Invalid file index {file_idx} in lookup table")
                return pd.DataFrame()
                
            file_path = self.parquet_files[file_idx]
            
            # Read the specific row from the file
            df = self._read_row_from_file(file_path, row_idx)
            
            # Check if we got a valid DataFrame
            if df is None or df.empty:
                print(f"Warning: Failed to read row {row_idx} from file {file_path}")
                return pd.DataFrame()
            
            # Apply preprocessing if provided
            if self.preprocess_fn:
                try:
                    df = self.preprocess_fn(df)
                except Exception as e:
                    print(f"Error in preprocessing: {str(e)}")
                    return pd.DataFrame()
            
            # Apply transformation if provided
            if self.transform_fn:
                try:
                    return self.transform_fn(df)
                except Exception as e:
                    print(f"Error in transform: {str(e)}")
                    return pd.DataFrame()
            else:
                return df
                
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {str(e)}")
            return pd.DataFrame()
        
    def get_sample_batch(self, sample_size=100):
        """Get a small sample batch for metadata and model initialization using a memory-efficient approach"""
        print(f"Getting sample batch of {sample_size} rows with memory-efficient approach...")
        
        # Use a very small sample size to avoid memory issues
        sample_size = min(20, sample_size, self.total_rows)  # Strict limit of 20 rows
        
        # We'll only use the first file to avoid loading too much
        if not self.parquet_files:
            raise ValueError("No parquet files available for sampling")
            
        file_path = self.parquet_files[0]
        
        try:
            # Use PyArrow for the most controlled memory usage
            import pyarrow.parquet as pq
            
            # First get the schema to check for columns
            schema = pq.read_schema(file_path)
            columns = schema.names
            
            # Identify the key column we need
            target_cols = ['category_id']
            if 'tax_type_id' in columns:
                target_cols.append('tax_type_id')
                
            # Add a few columns that might be useful for debugging
            for extra_col in ['txn_id', 'merchant_id', 'amount']:
                if extra_col in columns:
                    target_cols.append(extra_col)
                    
            # Limit to no more than 5 columns total to minimize memory
            target_cols = target_cols[:5]
            
            # Read only these columns and only the first few rows
            try:
                # Safest: most constrained reading approach
                table = pq.read_table(file_path, columns=target_cols, num_rows=sample_size)
                sample_df = table.to_pandas()
                print(f"Successfully read {len(sample_df)} rows with {len(target_cols)} columns")
                
                # Apply preprocessing if needed - but with caution
                if self.preprocess_fn:
                    try:
                        sample_df = self.preprocess_fn(sample_df)
                    except Exception as e:
                        print(f"Error in preprocessing sample: {str(e)}")
                        # Continue with unprocessed data
                
                return sample_df
                
            except Exception as e:
                print(f"Error with PyArrow targeted read: {str(e)}")
                # Fall through to pandas approach
                
        except Exception as e:
            print(f"Error with PyArrow setup: {str(e)}")
            
        # Fallback to pandas with strict constraints
        try:
            # Hardcode minimal column selection
            try:
                sample_df = pd.read_parquet(file_path, engine='pyarrow', columns=['category_id'], nrows=10)
            except:
                # If that fails, try without column selection
                sample_df = pd.read_parquet(file_path, engine='pyarrow', nrows=5)
                
            print(f"Fallback read {len(sample_df)} rows with pandas")
            return sample_df
            
        except Exception as e:
            print(f"All sampling methods failed: {str(e)}")
            # Return empty DataFrame as last resort
            return pd.DataFrame({'category_id': [0, 1, 2, 3, 4]})
        
    def get_batch_df(self, indices):
        """Get a batch of rows as a single DataFrame with improved error handling and performance"""
        if not indices:
            raise ValueError("Empty indices list provided to get_batch_df")
        
        dfs = []
        error_count = 0
        attempted = 0
        
        # Group indices by file to minimize file open/close operations
        file_indices = {}
        for idx in indices:
            if idx >= len(self.lookup):
                print(f"Warning: Index {idx} out of bounds")
                continue
                
            file_idx, row_idx = self.lookup[idx]
            if file_idx not in file_indices:
                file_indices[file_idx] = []
            file_indices[file_idx].append((idx, row_idx))
        
        # Process each file once instead of once per row
        for file_idx, row_data in file_indices.items():
            if file_idx >= len(self.parquet_files):
                print(f"Warning: Invalid file index {file_idx}")
                continue
                
            file_path = self.parquet_files[file_idx]
            try:
                # Try to load all rows from this file at once
                # Sort row indices to improve access patterns
                row_data.sort(key=lambda x: x[1])  # Sort by row_idx
                row_indices = [r[1] for r in row_data]
                orig_indices = [r[0] for r in row_data]
                
                # For small number of rows scattered throughout file, use individual access
                if len(row_indices) < 5:
                    for global_idx, row_idx in zip(orig_indices, row_indices):
                        attempted += 1
                        try:
                            df = self._read_row_from_file(file_path, row_idx)
                            if not df.empty:
                                if self.preprocess_fn:
                                    df = self.preprocess_fn(df)
                                dfs.append(df)
                        except Exception as e:
                            error_count += 1
                            if error_count <= 3:  # Limit error messages
                                print(f"Error reading row {row_idx} from {file_path}: {str(e)}")
                            continue
                else:
                    # For larger numbers of rows, try to read chunks efficiently
                    try:
                        # Check if rows are contiguous or nearly contiguous
                        is_contiguous = all(row_indices[i+1] - row_indices[i] <= 5 
                                           for i in range(len(row_indices)-1))
                        
                        if is_contiguous:
                            # Read the whole range at once
                            start_idx = min(row_indices)
                            end_idx = max(row_indices) + 1
                            
                            # Read the slice
                            import pyarrow.parquet as pq
                            table = pq.read_table(file_path)
                            df = table.slice(start_idx, end_idx - start_idx).to_pandas()
                            
                            # Filter to only keep the exact rows we want
                            idx_set = set(row_indices)
                            # Add a temporary index matching the file row numbers
                            df['_temp_idx_'] = range(start_idx, start_idx + len(df))
                            df = df[df['_temp_idx_'].isin(idx_set)]
                            df = df.drop('_temp_idx_', axis=1)
                            
                            if not df.empty and self.preprocess_fn:
                                df = self.preprocess_fn(df)
                            dfs.append(df)
                            attempted += len(row_indices)
                        else:
                            # Read rows individually but in batches
                            batch_size = 10
                            for i in range(0, len(row_indices), batch_size):
                                batch_indices = row_indices[i:i+batch_size]
                                batch_dfs = []
                                
                                for row_idx in batch_indices:
                                    attempted += 1
                                    try:
                                        df = self._read_row_from_file(file_path, row_idx)
                                        if not df.empty:
                                            batch_dfs.append(df)
                                    except Exception as e:
                                        error_count += 1
                                        continue
                                
                                if batch_dfs:
                                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                                    if self.preprocess_fn:
                                        batch_df = self.preprocess_fn(batch_df)
                                    dfs.append(batch_df)
                            
                    except Exception as e:
                        # Fall back to individual row access
                        print(f"Error with batch read from {file_path}, falling back to row-by-row: {str(e)}")
                        for global_idx, row_idx in zip(orig_indices, row_indices):
                            attempted += 1
                            try:
                                df = self._read_row_from_file(file_path, row_idx)
                                if not df.empty:
                                    if self.preprocess_fn:
                                        df = self.preprocess_fn(df)
                                    dfs.append(df)
                            except Exception as inner_e:
                                error_count += 1
                                continue
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
        
        # Check if we got any valid data
        if not dfs:
            error_msg = f"All dataframes in batch were empty (attempted {attempted} rows, {error_count} errors)"
            print(error_msg)
            raise ValueError(error_msg)
        
        # Combine all dataframes and reset index
        try:
            result = pd.concat(dfs, ignore_index=True)
            return result
        except Exception as e:
            print(f"Error combining dataframes: {str(e)}")
            # Last resort: if concat fails, return the first valid DataFrame
            return dfs[0]


def get_parquet_files(data_dir, max_files=None):
    """Get list of parquet files from directory, filtering for transaction data"""
    # Get all parquet files
    all_parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    # Filter for transaction_data_batch files
    transaction_data_files = [f for f in all_parquet_files if 'transaction_data_batch' in os.path.basename(f)]
    
    # If no transaction data files found, use all parquet files
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
    for file in parquet_files:
        print(f"  - {os.path.basename(file)}")
        
    return parquet_files


def preprocess_transactions(df):
    """Preprocess transaction DataFrame for model input"""
    # Ensure required columns are present
    required_columns = ['company_id', 'merchant_id', 'amount', 'category_id']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in dataset")
    
    # Optimize memory usage
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    # Ensure timestamp is in proper format
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    return df


def prepare_model_inputs(batch_df, model, device):
    """Prepare model inputs from a batch DataFrame"""
    # Check for empty dataframe
    if batch_df.empty:
        raise ValueError("Empty DataFrame provided to prepare_model_inputs")
    
    try:
        # Use the model's data preparation function
        data = model.prepare_data_from_dataframe(batch_df)
        
        # Efficiently move tensors to device in a single pass
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Use non-blocking transfer for better GPU utilization
                data[key] = value.to(device, non_blocking=True)
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to tensors directly on the target device
                data[key] = torch.from_numpy(value).to(device, non_blocking=True)
        
        # Check for required label column
        if 'category_id' not in batch_df.columns:
            raise ValueError("Required column 'category_id' not found in batch DataFrame")
        
        # Prepare labels dictionary
        labels = {}
        
        # Convert labels directly to tensors on the target device with strict type checking
        category_values = batch_df['category_id'].values
        
        # Check for and handle string values in category_id
        print(f"Original category_id dtype: {batch_df['category_id'].dtype}")
        print(f"Sample category_id values: {batch_df['category_id'].head().tolist()}")
        
        # Handle pandas categorical type directly
        if hasattr(batch_df['category_id'], 'cat'):
            print("Converting pandas categorical to numeric codes")
            # Get the categorical codes (integers) instead of the string values
            category_values = batch_df['category_id'].cat.codes.values
            # Handle -1 values (which represent NaN in categorical codes)
            category_values = np.where(category_values < 0, 0, category_values)
        # Handle string or object dtype
        elif batch_df['category_id'].dtype == object:
            print("Converting string/object category_id values to integers")
            try:
                # Try to convert to numeric directly
                numeric_vals = pd.to_numeric(batch_df['category_id'], errors='coerce')
                # Check if conversion worked for most values
                if numeric_vals.isna().mean() < 0.5:  # Less than 50% NaN
                    print("Direct numeric conversion mostly succeeded")
                    category_values = numeric_vals.fillna(0).astype(np.int64).values
                else:
                    # Use factorize for string values
                    print("Using factorize for string conversion")
                    category_codes, uniques = pd.factorize(batch_df['category_id'])
                    print(f"Mapped to {len(uniques)} unique categories")
                    category_values = category_codes.astype(np.int64)
            except Exception as e:
                print(f"Error in direct conversion: {str(e)}")
                # Fallback to factorize
                print("Falling back to factorize for conversion")
                category_codes, _ = pd.factorize(batch_df['category_id'])
                category_values = category_codes.astype(np.int64)
        else:
            # Numeric type, but ensure it's the right format
            try:
                # Ensure we have int64 values
                category_values = batch_df['category_id'].astype(np.int64).values
            except Exception as e:
                print(f"Error converting numeric category_id: {str(e)}")
                # Last resort: factorize everything
                category_codes, _ = pd.factorize(batch_df['category_id'])
                category_values = category_codes.astype(np.int64)
        
        # Create tensor with explicit dtype and enhanced error handling
        try:
            # Print conversion results for debugging
            print(f"Converted category values dtype: {category_values.dtype}, shape: {category_values.shape}")
            print(f"Converted sample values: {category_values[:5]}")
            
            # Ensure the values are 1D (flatten if needed)
            if len(category_values.shape) > 1:
                print(f"Flattening category values from shape {category_values.shape}")
                category_values = category_values.flatten()
            
            # Check for any NaN values
            if np.isnan(category_values).any():
                print("Warning: NaN values found in category_values, replacing with 0")
                category_values = np.nan_to_num(category_values, nan=0).astype(np.int64)
            
            # Create the tensor
            category_tensor = torch.tensor(
                category_values, 
                dtype=torch.long,
                device=device  # Create tensor directly on target device
            )
            labels['category'] = category_tensor
            
            # Final safety check
            if torch.isnan(category_tensor).any() or torch.isinf(category_tensor).any():
                print("Warning: NaN or Inf values in category tensor after creation")
                category_tensor = torch.nan_to_num(category_tensor, nan=0)
                labels['category'] = category_tensor
                
        except Exception as e:
            print(f"Error creating category tensor: {str(e)}")
            print(f"Category values detail - dtype: {category_values.dtype}, shape: {category_values.shape}")
            print(f"Sample values: {category_values[:5]}")
            # Last resort fallback - create zeros tensor of correct shape
            category_tensor = torch.zeros(len(batch_df), dtype=torch.long, device=device)
            labels['category'] = category_tensor
            
        # Add tax_type if available - with similar safeguards
        if 'tax_type_id' in batch_df.columns:
            tax_type_values = batch_df['tax_type_id'].values
            
            # Check for and handle string values in tax_type_id
            if tax_type_values.dtype == object:  # Object dtype suggests strings or mixed types
                print("Warning: Found non-numeric tax_type_id values. Converting to integers...")
                try:
                    # Try to convert to numeric values first
                    tax_type_values = pd.to_numeric(batch_df['tax_type_id'], errors='coerce').fillna(0).astype(np.int64).values
                except Exception as e:
                    print(f"Error converting tax_type_id to numeric: {str(e)}")
                    # Fallback: use factorize to create integer codes for strings
                    tax_type_codes, _ = pd.factorize(batch_df['tax_type_id'])
                    tax_type_values = tax_type_codes.astype(np.int64)
            
            # Create tensor with explicit dtype and error checking
            try:
                tax_type_tensor = torch.tensor(
                    tax_type_values,
                    dtype=torch.long,
                    device=device  # Create tensor directly on target device
                )
                labels['tax_type'] = tax_type_tensor
            except Exception as e:
                print(f"Error creating tax_type tensor: {str(e)}")
                # Last resort fallback - create zeros tensor of correct shape
                tax_type_tensor = torch.zeros(len(batch_df), dtype=torch.long, device=device)
                labels['tax_type'] = tax_type_tensor
        
        return data, labels
        
    except Exception as e:
        print(f"Error preparing model inputs: {str(e)}")
        # Re-raise with additional context
        raise RuntimeError(f"Failed to prepare model inputs: {str(e)}") from e


def create_cuda_graph(model, sample_data):
    """Create CUDA graph for model inference with enhanced error handling"""
    if not torch.cuda.is_available() or not config.use_cuda_graphs:
        print("CUDA graphs not available or disabled")
        return None
    
    try:
        print("Creating CUDA graph for optimized inference...")
        
        # Check CUDA compatibility
        cuda_version = torch.version.cuda
        if cuda_version is None or float(cuda_version.split('.')[0]) < 10:
            print(f"CUDA version {cuda_version} may not fully support CUDA graphs. Continuing anyway...")

        # Get model's expected input signature to extract only needed inputs
        # Get device - directly use the device passed to model creation rather than from parameters
        # which could be uninitialized with LazyLinear layers
        
        # Filter inputs based on model's forward method signature
        # This gets only the inputs the model actually needs
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
            else:
                print(f"Warning: Required input '{key}' not found in sample data")
        
        # Move tensors to device efficiently
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device, non_blocking=True)
        
        # Create static input batch with precise memory management
        static_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # Reserve memory on the same device with same dtype
                static_inputs[k] = torch.zeros_like(v, device=device, dtype=v.dtype)
                # Copy the content
                static_inputs[k].copy_(v)
            else:
                static_inputs[k] = v
        
        # Set model to eval mode for graph capture
        model.eval()
        
        # Warm up without no_grad to match the graph capture conditions
        # no_grad can cause issues if the model internals use requires_grad flags
        print("Warming up for CUDA graph capture...")
        for _ in range(3):
            # Deliberate omission of torch.no_grad() to match graph capture condition
            model(**static_inputs)
            # Ensure CUDA synchronization between runs
            torch.cuda.synchronize()
        
        # Clear CUDA cache before graph capture
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check memory before capturing
        print(f"Memory before graph capture: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Capture graph
        print("Capturing CUDA graph...")
        g = torch.cuda.CUDAGraph()
        
        # Capture with a smaller try/except block for precise error reporting
        try:
            with torch.cuda.graph(g):
                static_outputs = model(**static_inputs)
        except Exception as e:
            print(f"Error during CUDA graph capture: {str(e)}")
            print("Falling back to regular inference")
            return None
            
        print(f"CUDA graph captured successfully. Output type: {type(static_outputs)}")
        
        return {
            'graph': g,
            'static_inputs': static_inputs,
            'static_outputs': static_outputs
        }
        
    except Exception as e:
        print(f"Error creating CUDA graph: {str(e)}")
        print("Falling back to regular inference")
        return None


def run_with_cuda_graph(cuda_graph, data):
    """Run inference using CUDA graph with enhanced error handling and performance"""
    # Check if cuda_graph is valid
    if cuda_graph is None:
        raise ValueError("Invalid CUDA graph provided")
    
    try:
        # Synchronize before updating inputs to ensure any previous work is complete
        torch.cuda.synchronize()
        
        # Update static inputs with new data, with safety checks
        for k, v in data.items():
            if k in cuda_graph['static_inputs'] and isinstance(v, torch.Tensor):
                if isinstance(cuda_graph['static_inputs'][k], torch.Tensor):
                    # Check shape compatibility to avoid runtime errors
                    if cuda_graph['static_inputs'][k].shape != v.shape:
                        raise ValueError(
                            f"Shape mismatch for input '{k}': "
                            f"Expected {cuda_graph['static_inputs'][k].shape}, got {v.shape}"
                        )
                    
                    # Check dtype compatibility 
                    if cuda_graph['static_inputs'][k].dtype != v.dtype:
                        # Auto-convert if possible but warn
                        print(f"Warning: dtype mismatch for input '{k}'. Converting to {cuda_graph['static_inputs'][k].dtype}")
                        # Convert to target dtype before copy
                        v = v.to(dtype=cuda_graph['static_inputs'][k].dtype)
                    
                    # Efficient non-blocking copy with safety
                    cuda_graph['static_inputs'][k].copy_(v, non_blocking=True)
        
        # Ensure all copies are complete before running the graph
        torch.cuda.synchronize()
        
        # Run the graph
        cuda_graph['graph'].replay()
        
        # Synchronize to ensure graph execution is complete
        torch.cuda.synchronize()
        
        # Return cached outputs (already updated by the graph)
        return cuda_graph['static_outputs']
        
    except Exception as e:
        print(f"Error running CUDA graph: {str(e)}")
        # Re-raise to allow caller to handle the error
        raise RuntimeError(f"CUDA graph execution failed: {str(e)}") from e


def initialize_model(hidden_dim, num_categories, num_tax_types, config, device):
    """Initialize the EnhancedHybridTransactionModel"""
    model = EnhancedHybridTransactionModel(
        input_dim=hidden_dim,  # We'll project inputs to match this dimension
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


def evaluate(model, dataloader, dataset, device, cuda_graph=None):
    """Evaluate the model on the given dataset with improved error handling"""
    # Ensure model is in evaluation mode
    model.eval()
    
    # Initialize metrics
    total_loss = 0
    total_acc = 0
    total_samples = 0
    category_correct = 0
    category_total = 0
    tax_type_correct = 0
    tax_type_total = 0
    
    # Track predictions for detailed metrics
    all_preds = []
    all_labels = []
    all_tax_preds = []
    all_tax_labels = []
    
    # Use a try-except block for the entire evaluation to catch and report errors
    try:
        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Process each batch
            for batch_indices in tqdm(dataloader, desc="Evaluation", leave=False):
                # Skip empty batches
                if not batch_indices or len(batch_indices) == 0:
                    continue
                    
                try:
                    # Extract actual indices for this batch
                    start_idx = batch_indices[0]
                    end_idx = start_idx + len(batch_indices)
                    actual_indices = list(range(start_idx, min(end_idx, len(dataset))))
                    
                    if not actual_indices:
                        continue
                    
                    # Get batch dataframe
                    batch_df = dataset.get_batch_df(actual_indices)
                    
                    # Prepare inputs
                    data, labels = prepare_model_inputs(batch_df, model, device)
                    
                    # Forward pass with appropriate method
                    try:
                        if cuda_graph is not None and config.use_cuda_graphs:
                            # Use CUDA graph for faster inference when available
                            outputs = run_with_cuda_graph(cuda_graph, data)
                        else:
                            # Regular forward pass
                            outputs = model(**data)  # Use ** unpacking for cleaner code
                    except Exception as forward_error:
                        print(f"Error during model forward pass: {forward_error}")
                        # Fall back to regular forward pass if CUDA graph fails
                        if cuda_graph is not None:
                            print("Falling back to regular forward pass")
                            outputs = model(**data)
                        else:
                            # Re-raise if both methods fail
                            raise
                    
                    # Handle multi-task vs single-task output
                    batch_size = len(batch_df)
                    
                    if isinstance(outputs, tuple):
                        # Multi-task model (category and tax type)
                        category_logits, tax_type_logits = outputs
                        
                        # Apply safety measures to logits before loss calculation
                        category_logits = torch.nan_to_num(category_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                        category_logits = torch.clamp(category_logits, min=-20.0, max=20.0)
                        
                        # Use label smoothing and reduction='sum' for enhanced stability
                        try:
                            category_loss = nn.CrossEntropyLoss(
                                reduction='sum',
                                label_smoothing=0.1  # Add label smoothing for numerical stability
                            )(category_logits, labels['category'])
                            
                            # Guard against NaN loss
                            if torch.isnan(category_loss) or torch.isinf(category_loss):
                                raise ValueError("NaN or Inf detected in loss")
                        except Exception as e:
                            print(f"Loss calculation error: {str(e)} - using alternate calculation")
                            # Fall back to more stable loss calculation
                            category_logits_safe = F.log_softmax(category_logits, dim=1)
                            category_loss = F.nll_loss(
                                category_logits_safe, 
                                labels['category'],
                                reduction='sum'
                            )
                        
                        # Get category predictions with NaN handling
                        category_preds = category_logits.argmax(dim=1)
                        correct = (category_preds == labels['category']).sum().item()
                        category_correct += correct
                        category_total += batch_size
                        
                        # Store predictions for detailed metrics
                        all_preds.extend(category_preds.cpu().numpy())
                        all_labels.extend(labels['category'].cpu().numpy())
                        
                        # Compute total loss based on available labels
                        if 'tax_type' in labels:
                            # Apply safety measures to tax type logits before loss calculation
                            tax_type_logits = torch.nan_to_num(tax_type_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                            tax_type_logits = torch.clamp(tax_type_logits, min=-20.0, max=20.0)
                            
                            # Calculate tax type loss with safeguards
                            try:
                                tax_type_loss = nn.CrossEntropyLoss(
                                    reduction='sum',
                                    label_smoothing=0.1  # Add label smoothing for numerical stability
                                )(tax_type_logits, labels['tax_type'])
                                
                                # Guard against NaN loss
                                if torch.isnan(tax_type_loss) or torch.isinf(tax_type_loss):
                                    raise ValueError("NaN or Inf detected in tax_type_loss")
                            except Exception as e:
                                print(f"Tax loss calculation error: {str(e)} - using alternate calculation")
                                # Fall back to more stable loss calculation
                                tax_type_logits_safe = F.log_softmax(tax_type_logits, dim=1)
                                tax_type_loss = F.nll_loss(
                                    tax_type_logits_safe, 
                                    labels['tax_type'],
                                    reduction='sum'
                                )
                            
                            # Combine losses with safeguards
                            loss = 0.7 * category_loss + 0.3 * tax_type_loss
                            
                            # Final sanity check on combined loss
                            if torch.isnan(loss) or torch.isinf(loss):
                                print("Combined loss is NaN or Inf - using category loss only")
                                loss = category_loss
                            
                            # Get tax type predictions
                            tax_type_preds = tax_type_logits.argmax(dim=1)
                            tax_correct = (tax_type_preds == labels['tax_type']).sum().item()
                            tax_type_correct += tax_correct
                            tax_type_total += batch_size
                            
                            # Store tax predictions for detailed metrics
                            all_tax_preds.extend(tax_type_preds.cpu().numpy())
                            all_tax_labels.extend(labels['tax_type'].cpu().numpy())
                        else:
                            # Use only category loss if tax_type not available
                            loss = category_loss
                    else:
                        # Single task model (category only)
                        # Apply safety measures to logits before loss calculation
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
                        outputs = torch.clamp(outputs, min=-20.0, max=20.0)
                        
                        # Use label smoothing and reduction='sum' for enhanced stability
                        try:
                            category_loss = nn.CrossEntropyLoss(
                                reduction='sum',
                                label_smoothing=0.1  # Add label smoothing for numerical stability
                            )(outputs, labels['category'])
                            
                            # Guard against NaN loss
                            if torch.isnan(category_loss) or torch.isinf(category_loss):
                                raise ValueError("NaN or Inf detected in loss")
                        except Exception as e:
                            print(f"Loss calculation error: {str(e)} - using alternate calculation")
                            # Fall back to more stable loss calculation
                            outputs_safe = F.log_softmax(outputs, dim=1)
                            category_loss = F.nll_loss(
                                outputs_safe, 
                                labels['category'],
                                reduction='sum'
                            )
                            
                            # If still NaN, use constant loss as fallback
                            if torch.isnan(category_loss) or torch.isinf(category_loss):
                                print("Still getting NaN loss - using constant loss")
                                category_loss = torch.tensor(2.5 * outputs.size(0), device=outputs.device)
                        
                        loss = category_loss
                        
                        # Get category predictions with safeguards
                        category_preds = outputs.argmax(dim=1)
                        correct = (category_preds == labels['category']).sum().item()
                        category_correct += correct
                        category_total += batch_size
                        
                        # Store predictions for metrics
                        all_preds.extend(category_preds.cpu().numpy())
                        all_labels.extend(labels['category'].cpu().numpy())
                    
                    # Update metrics (use sum reduction for proper weighting)
                    total_loss += loss.item()
                    total_samples += batch_size
                    
                except Exception as batch_error:
                    print(f"Error processing evaluation batch: {str(batch_error)}")
                    # Continue with next batch instead of breaking
                    continue
        
        # Calculate final metrics
        if total_samples == 0:
            print("Warning: No samples were successfully processed during evaluation")
            return float('inf'), 0.0
        
        # Compute averages
        avg_loss = total_loss / total_samples
        
        # Compute accuracies
        category_accuracy = category_correct / max(1, category_total)
        
        # Log additional metrics if available
        if all_preds and len(all_preds) > 1 and len(all_labels) > 1:
            try:
                # Compute F1 score for detailed reporting
                f1 = f1_score(all_labels, all_preds, average='weighted')
                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
                
                print(f"\nDetailed category metrics:")
                print(f"F1 Score (weighted): {f1:.4f}")
                print(f"Precision (weighted): {precision:.4f}")
                print(f"Recall (weighted): {recall:.4f}")
                
                # Report tax type metrics if available
                if tax_type_total > 0:
                    tax_accuracy = tax_type_correct / max(1, tax_type_total)
                    print(f"Tax Type Accuracy: {tax_accuracy:.4f}")
                    
                    if all_tax_preds and len(all_tax_preds) > 1 and len(all_tax_labels) > 1:
                        tax_f1 = f1_score(all_tax_labels, all_tax_preds, average='weighted')
                        print(f"Tax Type F1 Score: {tax_f1:.4f}")
            except Exception as metric_error:
                print(f"Error computing detailed metrics: {metric_error}")
        
        return avg_loss, category_accuracy
        
    except Exception as e:
        print(f"Unexpected error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return default values to allow training to continue
        return float('inf'), 0.0


def train(model, train_dataset, val_dataset, config, device):
    """Train the model with robust error handling and optimized training loop"""
    print(f"\n{'='*80}\nStarting Training\n{'='*80}")
    print(f"Batch size: {config.batch_size}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Using {'mixed precision' if config.use_amp else 'full precision'} training")
    print(f"Using device: {device}")
    
    # Create optimizer with improved defaults
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8,  # More stable epsilon value
        betas=(0.9, 0.999)  # Default betas, explicitly stated for clarity
    )
    
    # Learning rate scheduler with improved settings
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,  # Halve the learning rate
        patience=config.patience // 2,
        verbose=True,
        min_lr=1e-6  # Set a minimum learning rate
    )
    
    # Setup mixed precision training if enabled
    scaler = GradScaler(enabled=config.use_amp) if torch.cuda.is_available() else None
    
    # Create train dataloader with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Shuffle for better training
        num_workers=0,  # Safe default - can be increased based on hardware
        pin_memory=torch.cuda.is_available(),  # Only pin if CUDA is available
        collate_fn=df_collate_fn,
        drop_last=False  # Keep all samples
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=0,  # Safe default
        pin_memory=torch.cuda.is_available(),
        collate_fn=df_collate_fn,
        drop_last=False
    )
    
    # Setup training state tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    global_step = 0
    
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
    
    try:
        # Training loop
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            
            # Explicitly set model to training mode at the start of each epoch
            model.train()
            
            # Reset epoch metrics
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            # Progress bar for this epoch
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
            
            # Process each batch
            for step, batch_indices in pbar:
                # Skip empty batches
                if not batch_indices or len(batch_indices) == 0:
                    continue
                    
                try:
                    # Handle indices from custom collate function
                    start_idx = batch_indices[0] 
                    end_idx = start_idx + len(batch_indices)
                    actual_indices = list(range(start_idx, min(end_idx, len(train_dataset))))
                    
                    if not actual_indices:
                        continue
                    
                    # Get batch dataframe
                    batch_df = train_dataset.get_batch_df(actual_indices)
                    batch_size = len(batch_df)
                    
                    # Zero gradients BEFORE preparing inputs (important for correct gradient flow)
                    optimizer.zero_grad(set_to_none=True)  # More efficient than just zero_grad()
                    
                    # Prepare inputs
                    data, labels = prepare_model_inputs(batch_df, model, device)
                    
                    # Define a helper function for computing loss to avoid code duplication
                    def compute_loss_and_acc(outputs, labels):
                        if isinstance(outputs, tuple):
                            # Multi-task model
                            category_logits, tax_type_logits = outputs
                            
                            # Compute primary category loss
                            # Apply safety measures to logits before loss calculation
                            category_logits = torch.nan_to_num(category_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                            category_logits = torch.clamp(category_logits, min=-20.0, max=20.0)
                            
                            # Use label smoothing for enhanced stability
                            try:
                                category_loss = nn.CrossEntropyLoss(
                                    label_smoothing=0.1  # Add label smoothing for numerical stability
                                )(category_logits, labels['category'])
                                
                                # Guard against NaN loss
                                if torch.isnan(category_loss) or torch.isinf(category_loss):
                                    raise ValueError("NaN or Inf detected in validation loss")
                            except Exception as e:
                                print(f"Validation loss calculation error: {str(e)} - using alternate calculation")
                                # Fall back to more stable loss calculation
                                category_logits_safe = F.log_softmax(category_logits, dim=1)
                                category_loss = F.nll_loss(
                                    category_logits_safe, 
                                    labels['category']
                                )
                            
                            # Get category predictions and accuracy
                            category_preds = category_logits.argmax(dim=1)
                            correct = (category_preds == labels['category']).sum().item()
                            
                            # Compute additional loss if tax_type is present
                            if 'tax_type' in labels:
                                tax_type_loss = nn.CrossEntropyLoss()(tax_type_logits, labels['tax_type'])
                                loss = 0.7 * category_loss + 0.3 * tax_type_loss
                            else:
                                loss = category_loss
                        else:
                            # Single task model
                            category_loss = nn.CrossEntropyLoss()(outputs, labels['category'])
                            loss = category_loss
                            
                            # Get category predictions and accuracy
                            category_preds = outputs.argmax(dim=1)
                            correct = (category_preds == labels['category']).sum().item()
                            
                        return loss, correct
                    
                    # Forward pass with or without mixed precision
                    if config.use_amp and scaler is not None:
                        # Mixed precision training path
                        with autocast(enabled=config.use_amp):
                            # Forward pass with cleaner ** unpacking
                            outputs = model(**data)
                            loss, correct = compute_loss_and_acc(outputs, labels)
                        
                        # Scale loss, compute gradients, and optimize with scaled gradients
                        scaler.scale(loss).backward()
                        
                        # Apply gradient clipping to scaled gradients
                        if config.grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                            
                        # Step optimizer and update scaler
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard precision training path
                        outputs = model(**data)
                        loss, correct = compute_loss_and_acc(outputs, labels)
                        
                        # Standard backward and optimize
                        loss.backward()
                        
                        # Apply gradient clipping if enabled
                        if config.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                            
                        # Update weights
                        optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item() * batch_size
                    epoch_correct += correct
                    epoch_samples += batch_size
                    global_step += 1
                    
                    # Update progress bar every step
                    if epoch_samples > 0:  # Guard against division by zero
                        train_loss = epoch_loss / epoch_samples
                        train_acc = epoch_correct / epoch_samples
                        
                        # Update progress bar with current metrics
                        pbar.set_description(
                            f"Train Loss: {train_loss:.4f}, "
                            f"Acc: {train_acc:.4f}, "
                            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                        )
                    
                    # Periodic validation during training
                    if step > 0 and step % config.eval_steps == 0:
                        print(f"\nPerforming validation at step {step}/{len(train_loader)}")
                        
                        # Switch to evaluation mode
                        model.eval()
                        
                        # Initialize CUDA graph for inference if enabled and not already created
                        if config.use_cuda_graphs and cuda_graph is None and torch.cuda.is_available():
                            # Only create CUDA graph after model has processed some batches
                            cuda_graph = create_cuda_graph(model, data)
                        
                        # Run evaluation
                        val_loss, val_acc = evaluate(model, val_loader, val_dataset, device, cuda_graph)
                        
                        # Print validation results
                        print(f"Step {step}/{len(train_loader)}, "
                              f"Train Loss: {epoch_loss / max(1, epoch_samples):.4f}, "
                              f"Train Acc: {epoch_correct / max(1, epoch_samples):.4f}, "
                              f"Val Loss: {val_loss:.4f}, "
                              f"Val Acc: {val_acc:.4f}")
                        
                        # Switch back to training mode
                        model.train()
                        
                except Exception as batch_error:
                    print(f"Error processing training batch: {str(batch_error)}")
                    import traceback
                    traceback.print_exc()
                    continue  # Skip to next batch instead of breaking
            
            # End of epoch: Calculate final training metrics
            if epoch_samples == 0:
                print("Warning: No samples were processed in this epoch")
                train_loss = float('inf')
                train_acc = 0.0
            else:
                train_loss = epoch_loss / epoch_samples
                train_acc = epoch_correct / epoch_samples
            
            # Full validation at end of epoch
            model.eval()
            val_loss, val_acc = evaluate(model, val_loader, val_dataset, device, cuda_graph)
            
            # Update learning rate based on validation loss
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Track metrics history
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
                
                # Save best model with comprehensive checkpoint
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'config': config.__dict__,
                        'metrics': metrics
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
            
            # Save regular checkpoint (less frequently to save disk space)
            if (epoch + 1) % 2 == 0 or epoch == config.num_epochs - 1:  # Every 2 epochs or last epoch
                checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 
                        'scaler': scaler.state_dict() if scaler else None,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'metrics': metrics,
                        'config': config.__dict__
                    },
                    checkpoint_path
                )
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Plot current progress
            if (epoch + 1) % 5 == 0:  # Every 5 epochs
                try:
                    plot_training_curves(metrics, config, epoch + 1)
                except Exception as plot_error:
                    print(f"Error plotting training curves: {plot_error}")
        
        print(f"\n{'='*80}\nTraining completed\n{'='*80}")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        
        # Final metrics dictionary
        return metrics
        
    except Exception as e:
        print(f"\nUnexpected error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save emergency checkpoint if possible
        try:
            if epoch > 0:  # Only if we've completed at least one epoch
                emergency_path = os.path.join(config.output_dir, "emergency_checkpoint.pt")
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                        'error': str(e)
                    },
                    emergency_path
                )
                print(f"Saved emergency checkpoint to {emergency_path}")
        except:
            print("Failed to save emergency checkpoint")
            
        # Return whatever metrics we have so far
        return metrics


def extract_embeddings_for_xgboost(model, dataset, output_file, config, device):
    """Extract node embeddings for XGBoost integration"""
    model.eval()
    
    embeddings_list = []
    labels_list = []
    transaction_ids = []
    
    # Create dataloader with smaller batch size for embedding extraction
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
                if len(batch_indices) == 0:
                    continue
                    
                # Handle the case where batch_indices is a list of indices
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
            except Exception as e:
                print(f"Error extracting embeddings for batch: {str(e)}")
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


def plot_training_curves(metrics, config, current_epoch=None):
    """Plot comprehensive training metrics with enhanced visualization"""
    # Create more comprehensive plots (2x2 grid)
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Training Progress - Enhanced Transaction Graph Model", fontsize=16)
    
    # If current_epoch is provided, use it for file naming and plot annotation
    epoch_info = f"" if current_epoch is None else f" (Epoch {current_epoch})"
    
    # 1. Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_losses'], 'b-', label='Train Loss')
    plt.plot(metrics['val_losses'], 'r-', label='Validation Loss')
    
    # Highlight best validation loss
    if len(metrics['val_losses']) > 0:
        best_epoch = np.argmin(metrics['val_losses'])
        best_loss = metrics['val_losses'][best_epoch]
        plt.plot(best_epoch, best_loss, 'ro', markersize=8, label=f'Best: {best_loss:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss{epoch_info}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    plt.subplot(2, 2, 2)
    plt.plot(metrics['train_accs'], 'b-', label='Train Accuracy')
    plt.plot(metrics['val_accs'], 'r-', label='Validation Accuracy')
    
    # Highlight best validation accuracy
    if len(metrics['val_accs']) > 0:
        best_acc_epoch = np.argmax(metrics['val_accs'])
        best_acc = metrics['val_accs'][best_acc_epoch]
        plt.plot(best_acc_epoch, best_acc, 'ro', markersize=8, label=f'Best: {best_acc:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy{epoch_info}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Learning rate over time
    if 'learning_rates' in metrics and len(metrics['learning_rates']) > 0:
        plt.subplot(2, 2, 3)
        plt.plot(metrics['learning_rates'], 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        # Use log scale if learning rate changes significantly
        if max(metrics['learning_rates']) / min(metrics['learning_rates']) > 5:
            plt.yscale('log')
    
    # 4. Train-Validation Gap (to monitor overfitting)
    if len(metrics['train_losses']) > 0 and len(metrics['val_losses']) > 0:
        plt.subplot(2, 2, 4)
        # Calculate the gaps
        train_val_loss_gaps = []
        for t_loss, v_loss in zip(metrics['train_losses'], metrics['val_losses']):
            # Calculate relative gap as percentage
            if t_loss > 0:  # Avoid division by zero
                gap = (v_loss - t_loss) / t_loss * 100
                train_val_loss_gaps.append(gap)
            else:
                train_val_loss_gaps.append(0)
                
        plt.plot(train_val_loss_gaps, 'c-', label='Loss Gap')  
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Val-Train Gap (%)')
        plt.title('Overfitting Monitor')
        
        # Add overfitting guidance
        mean_last_gaps = np.mean(train_val_loss_gaps[-3:]) if len(train_val_loss_gaps) >= 3 else (
            train_val_loss_gaps[-1] if train_val_loss_gaps else 0)
            
        if mean_last_gaps > 20:
            plt.text(0.5, 0.9, 'Potential Overfitting', ha='center', transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='red', alpha=0.2))
        elif mean_last_gaps < -20:  # Underfitting case
            plt.text(0.5, 0.9, 'Potential Underfitting', ha='center', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='blue', alpha=0.2))
        else:
            plt.text(0.5, 0.9, 'Good Fit', ha='center', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='green', alpha=0.2))
            
        plt.grid(True, alpha=0.3)
    
    # Add configuration summary as text
    plt.figtext(0.5, 0.01, 
               f"Model: {config.hidden_dim}d, {'Hyperbolic' if config.use_hyperbolic else 'Euclidean'}, "
               f"BS={config.batch_size}, LR={config.learning_rate:g}", 
               ha="center", fontsize=10)
    
    # Ensure proper spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure with epoch info if provided
    filename = 'training_curves.png' if current_epoch is None else f'training_curves_epoch_{current_epoch}.png'
    plt.savefig(os.path.join(config.output_dir, filename), dpi=120)
    plt.close()


def configure_for_hardware(config):
    """Configure settings based on available hardware with enhanced detection and performance tuning"""
    # Initialize variables to track hardware capabilities
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available()
    
    # Determine compute device with priorities: CUDA -> MPS -> CPU
    if has_cuda:
        device_type = 'cuda'
        device = torch.device('cuda')
        print(f"🚀 Using CUDA GPU acceleration")
    elif has_mps:
        device_type = 'mps'
        device = torch.device('mps')
        print(f"🚀 Using MPS (Apple Silicon) acceleration")
    else:
        device_type = 'cpu'
        device = torch.device('cpu')
        print(f"⚠️ Using CPU (no GPU acceleration available)")
    
    # Configure settings based on detected hardware
    if device_type == 'cuda':
        # Get detailed GPU information
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        
        # Get GPU memory information in GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        current_allocated = torch.cuda.memory_allocated(0) / 1e9
        current_reserved = torch.cuda.memory_reserved(0) / 1e9
        available_memory = total_memory - current_allocated
        
        print(f"GPU: {gpu_name} ({gpu_count} available)")
        print(f"CUDA Version: {cuda_version}")
        print(f"Memory: {current_allocated:.2f} GB allocated, {current_reserved:.2f} GB reserved")
        print(f"Total GPU Memory: {total_memory:.2f} GB, Available: {available_memory:.2f} GB")
        
        # Define GPU tiers based on card type and memory
        is_v100 = 'V100' in gpu_name
        is_a100 = 'A100' in gpu_name
        is_h100 = 'H100' in gpu_name
        is_t4 = 'T4' in gpu_name
        is_ampere = any(x in gpu_name for x in ['A10', 'A40', 'A100', '3090', '4090'])
        is_high_end = is_v100 or is_a100 or is_h100 or ('TITAN' in gpu_name) or ('RTX' in gpu_name and ('2080' in gpu_name or '3080' in gpu_name))
        
        # Configure for specific GPU types
        if is_h100:
            # H100 configuration - maximum performance
            print("🔥 Using optimized configuration for H100 GPU")
            config.batch_size = min(256, max(128, int(available_memory * 30)))  # Dynamic batch size
            config.hidden_dim = 1024
            config.num_heads = 16
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(8, os.cpu_count() or 4)
            config.use_neural_ode = True  # H100 can handle this complexity
            
        elif is_a100:
            # A100 configuration - very high performance
            print("🔥 Using optimized configuration for A100 GPU")
            config.batch_size = min(192, max(96, int(available_memory * 25)))
            config.hidden_dim = 768
            config.num_heads = 12
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(6, os.cpu_count() or 4)
            
        elif is_v100:
            # V100 configuration - high performance
            print("🔥 Using optimized configuration for V100 GPU")
            config.batch_size = min(128, max(64, int(available_memory * 20)))
            config.hidden_dim = 512
            config.num_heads = 8
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(4, os.cpu_count() or 2)
            
        elif is_t4:
            # T4 configuration - medium performance
            print("🔹 Using optimized configuration for T4 GPU")
            config.batch_size = min(64, max(32, int(available_memory * 15)))
            config.hidden_dim = 256
            config.num_heads = 4
            config.use_amp = True
            config.use_cuda_graphs = True
            config.num_workers = min(2, os.cpu_count() or 1)
            
        else:
            # Generic GPU configuration based on available memory
            print(f"⚙️ Using dynamically configured settings for {gpu_name}")
            
            # Configure based on available memory
            if available_memory > 20:
                # For GPUs with more than 20GB memory
                config.batch_size = min(128, max(64, int(available_memory * 6)))
                config.hidden_dim = 512
                config.num_heads = 8
                config.use_neural_ode = False  # Safer default
                config.num_workers = min(4, os.cpu_count() or 2)
            elif available_memory > 10:
                # For GPUs with more than 10GB memory
                config.batch_size = min(64, max(32, int(available_memory * 5)))
                config.hidden_dim = 256
                config.num_heads = 4
                config.use_neural_ode = False
                config.num_workers = min(2, os.cpu_count() or 1)
            else:
                # For smaller GPUs (8GB or less)
                config.batch_size = min(32, max(16, int(available_memory * 4)))
                config.hidden_dim = 128
                config.num_heads = 2
                config.use_neural_ode = False
                config.num_workers = min(1, os.cpu_count() or 1)
            
            # Enable mixed precision for all CUDA devices
            config.use_amp = True
            
            # Only enable CUDA graphs for GPUs with compute capability >= 7.0 (Volta+)
            major, _ = torch.cuda.get_device_capability(0)
            config.use_cuda_graphs = (major >= 7)
            if not config.use_cuda_graphs:
                print("⚠️ CUDA Graphs disabled: GPU compute capability too low (requires 7.0+)")
        
    elif device_type == 'mps':
        # Apple Silicon (M1/M2/M3) configuration
        print("🍎 Using Apple Silicon configuration")
        config.batch_size = 32
        config.hidden_dim = 128
        config.num_heads = 4
        config.use_amp = False  # MPS doesn't support AMP yet
        config.use_cuda_graphs = False
        config.use_neural_ode = False
        config.num_workers = min(2, os.cpu_count() or 1)
        
    else:
        # CPU configuration
        print("💻 Using CPU configuration")
        cpu_count = os.cpu_count() or 2
        print(f"Available CPU cores: {cpu_count}")
        
        # Smaller model configuration for CPU
        config.batch_size = 16
        config.hidden_dim = 64
        config.num_heads = 2
        config.use_amp = False
        config.use_cuda_graphs = False
        config.use_neural_ode = False
        config.num_workers = 0  # Safer default for CPU
        config.prefetch_factor = None
        
        # Reduce complexity for CPU training
        config.num_graph_layers = min(config.num_graph_layers, 2)
        config.num_temporal_layers = min(config.num_temporal_layers, 2)
        config.use_hyperbolic = False
    
    # Always set cuda_graph_batch_size for consistency
    config.cuda_graph_batch_size = config.batch_size
    
    # Print final configuration summary
    print("\n📊 Hardware-Optimized Configuration:")
    print(f"Batch Size: {config.batch_size}")
    print(f"Hidden Dimension: {config.hidden_dim}")
    print(f"Attention Heads: {config.num_heads}")
    print(f"Mixed Precision: {'Enabled' if config.use_amp else 'Disabled'}")
    print(f"CUDA Graphs: {'Enabled' if config.use_cuda_graphs else 'Disabled'}")
    print(f"Data Workers: {config.num_workers}")
    print(f"Graph Layers: {config.num_graph_layers}")
    print(f"Temporal Layers: {config.num_temporal_layers}")
    print(f"Hyperbolic Encoding: {'Enabled' if config.use_hyperbolic else 'Disabled'}")
    print(f"Neural ODE: {'Enabled' if config.use_neural_ode else 'Disabled'}")
    
    return device, config


def parse_args():
    """Parse command line arguments with improved defaults and more options"""
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
    train_group.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (auto-configured if not specified)')
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
    
    # Set random seed immediately
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Force CPU if requested
    if args.cpu_only and torch.cuda.is_available():
        print("⚠️ GPU detected but CPU usage forced by --cpu_only flag")
        # Prevent CUDA from being used
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    return args


def main():
    """Main training script entry point with enhanced error handling and diagnostics"""
    # Record start time for overall script timing
    script_start_time = time.time()
    
    try:
        print(f"\n{'='*80}")
        print(f"Enhanced Transaction Graph Model Training Script")
        print(f"{'='*80}")
        
        # Parse command line arguments
        args = parse_args()
        
        # Create configuration with base settings
        config = Config()
        
        # Override config with command line arguments
        # Data configuration
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.max_files:
            config.max_files = args.max_files
        
        # Model configuration
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
        config.use_hyperbolic = args.use_hyperbolic  # This can be turned off with --no_hyperbolic
        if args.use_neural_ode:
            config.use_neural_ode = True
        if args.use_text:
            config.use_text = True
            
        # Training configuration
        if args.batch_size:
            config.batch_size = args.batch_size
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
            
        # Hardware/optimization
        config.use_amp = args.use_amp  # Can be disabled with --no_amp
        config.use_cuda_graphs = args.use_cuda_graphs  # Can be disabled with --no_cuda_graphs
        if args.num_workers is not None:
            config.num_workers = args.num_workers
        if args.eval_steps:
            config.eval_steps = args.eval_steps
            
        # Additional features
        if args.extract_embeddings:
            config.extract_embeddings = True
        if args.embedding_output:
            config.embedding_output_file = args.embedding_output
            
        # Configure for available hardware - this will set device and hardware-specific optimizations
        device, config = configure_for_hardware(config)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save the full configuration for reproducibility
        config_path = os.path.join(config.output_dir, 'training_config.txt')
        with open(config_path, 'w') as f:
            f.write("Enhanced Transaction Graph Model Training Configuration\n")
            f.write("="*50 + "\n")
            for key, value in sorted(vars(config).items()):
                if not key.startswith('__'):
                    f.write(f"{key}: {value}\n")
        
        print(f"✓ Configuration saved to {config_path}")
        
        # Data loading phase
        print("\n📂 Data Loading Phase")
        print("-" * 40)
        
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
        print(f"✓ Found {len(parquet_files)} parquet files")
        print(f"✓ Training on {len(train_files)} files, validating on {len(val_files)} files")
        
        # Diagnostic: Check file sizes
        try:
            total_file_size_gb = sum(os.path.getsize(f) for f in parquet_files) / 1e9
            print(f"✓ Total data size: {total_file_size_gb:.2f} GB")
        except Exception as e:
            print(f"⚠️ Could not determine total file size: {str(e)}")
        
        # Create datasets with progress reporting
        print("\n🔄 Creating datasets...")
        log_memory_usage("before dataset creation")
        
        train_dataset = ParquetTransactionDataset(train_files, preprocess_fn=preprocess_transactions)
        log_memory_usage("after train dataset creation")
        
        val_dataset = ParquetTransactionDataset(val_files, preprocess_fn=preprocess_transactions)
        log_memory_usage("after validation dataset creation")
        
        print(f"✓ Train dataset: {len(train_dataset):,} transactions")
        print(f"✓ Validation dataset: {len(val_dataset):,} transactions")
        
        # Sample a small batch to get metadata and schema
        print("\n🔍 Analyzing dataset schema...")
        log_memory_usage("before schema analysis")
        
        # Default values in case sampling fails
        num_categories = 400
        num_tax_types = 20
        
        try:
            print("Getting sample batch of 100 rows with low memory approach...")
            # Clear memory before sample batch loading
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # If we have train files, read a very small sample from just one file
            if train_files:
                # Get the first file
                sample_file = train_files[0]
                print(f"Reading schema from first file: {os.path.basename(sample_file)}")
                
                # Read minimal data - reduced columns and row count first to analyze memory usage
                try:
                    # Use PyArrow direct API for better memory control
                    import pyarrow.parquet as pq
                    
                    # First just get the schema without loading data
                    parquet_schema = pq.read_schema(sample_file)
                    print(f"File has {len(parquet_schema.names)} columns")
                    
                    # Read just a few rows first to be cautious
                    table = pq.read_table(sample_file, columns=['category_id', 'tax_type_id'] if 'category_id' in parquet_schema.names else None, rows=10)
                    small_sample_df = table.to_pandas()
                    print(f"Successfully read {len(small_sample_df)} rows for schema analysis")
                    
                    # Show how many categories we have from the small sample
                    if 'category_id' in small_sample_df.columns:
                        num_small_categories = small_sample_df['category_id'].nunique()
                        print(f"Small sample contains {num_small_categories} unique categories")
                        
                        # If we need more categories, try a larger sample
                        if num_small_categories < 10 and len(small_sample_df) < 100:
                            print("Small sample has few categories, trying to read more rows...")
                            try:
                                # Try to read up to 100 rows
                                table = pq.read_table(sample_file, columns=['category_id', 'tax_type_id'], rows=100)
                                sample_df = table.to_pandas()
                                num_categories = sample_df['category_id'].nunique() if 'category_id' in sample_df.columns else 400
                                num_tax_types = sample_df['tax_type_id'].nunique() if 'tax_type_id' in sample_df.columns else 20
                            except Exception as e:
                                print(f"Error reading larger sample: {str(e)}")
                        else:
                            # Use values from small sample
                            num_categories = num_small_categories if 'category_id' in small_sample_df.columns else 400
                            num_tax_types = small_sample_df['tax_type_id'].nunique() if 'tax_type_id' in small_sample_df.columns else 20
                    
                    print("Partial schema analysis (based on limited columns):")
                    for col in small_sample_df.columns:
                        print(f"  - {col}: {small_sample_df[col].dtype}")
                
                except Exception as e:
                    print(f"Error with PyArrow approach: {str(e)}")
                    print("Trying alternate method with Pandas...")
                    
                    try:
                        # Fallback to pandas with strict constraints
                        sample_df = pd.read_parquet(sample_file, engine='pyarrow', columns=['category_id', 'tax_type_id'], nrows=50)
                        
                        # Get counts
                        num_categories = sample_df['category_id'].nunique() if 'category_id' in sample_df.columns else 400
                        num_tax_types = sample_df['tax_type_id'].nunique() if 'tax_type_id' in sample_df.columns else 20
                    except Exception as e2:
                        print(f"Error with pandas approach: {str(e2)}")
                        print("Using default values")
            else:
                print("No training files available for schema analysis")
            
            print(f"\n✓ Number of unique categories: {num_categories}")
            print(f"✓ Number of unique tax types: {num_tax_types}")
                
        except Exception as e:
            print(f"⚠️ Error analyzing sample data: {str(e)}")
            print("Using default category and tax type counts")
            num_categories = 400
            num_tax_types = 20
        
        # Model initialization phase
        print("\n🧠 Model Initialization Phase")
        print("-" * 40)
        log_memory_usage("before model initialization")
        
        # Clear memory before model creation
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        model = initialize_model(config.hidden_dim, num_categories, num_tax_types, config, device)
        log_memory_usage("after model initialization")
        
        # Count parameters - handle LazyLinear layers
        try:
            # First try to count parameters directly
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✓ Model initialized with {total_params:,} total parameters")
            print(f"✓ Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        except ValueError as e:
            # If we get an error about uninitialized parameters (LazyLinear)
            print("⚠️ Model uses LazyLinear layers - parameters will be initialized during first forward pass")
            print("⚠️ Parameter count will be available after training begins")
        
        # Training phase
        print("\n🏋️ Training Phase")
        print("-" * 40)
        
        # Train the model with timing
        training_start_time = time.time()
        try:
            metrics = train(model, train_dataset, val_dataset, config, device)
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            
            hours, remainder = divmod(training_duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\n✓ Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Get best results
            best_epoch = metrics.get('best_epoch', 0)
            best_val_loss = metrics.get('best_val_loss', float('inf'))
            
            if 'val_accs' in metrics and len(metrics['val_accs']) > 0:
                best_acc_epoch = np.argmax(metrics['val_accs'])
                best_acc = metrics['val_accs'][best_acc_epoch]
                print(f"✓ Best validation accuracy: {best_acc:.4f} at epoch {best_acc_epoch + 1}")
            
            # Plot training curves
            try:
                plot_training_curves(metrics, config)
                print(f"✓ Training curves saved to {config.output_dir}/training_curves.png")
            except Exception as plot_error:
                print(f"⚠️ Error plotting training curves: {str(plot_error)}")
            
            # Load best model for evaluation or embedding extraction
            best_model_path = os.path.join(config.output_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                print("\n📊 Model Evaluation Phase")
                print("-" * 40)
                
                try:
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")
                    print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
                    print(f"  - Validation accuracy: {checkpoint['val_acc']:.4f}")
                    
                    # Extract embeddings for XGBoost if requested
                    if config.extract_embeddings:
                        print("\n🔍 Extracting Embeddings for XGBoost Integration")
                        print("-" * 40)
                        embeddings_df = extract_embeddings_for_xgboost(
                            model, 
                            train_dataset, 
                            config.embedding_output_file, 
                            config, 
                            device
                        )
                        
                        if embeddings_df is not None:
                            # Show embedding details
                            embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
                            print(f"✓ Embedding dimension: {len(embedding_cols)}")
                            print(f"✓ Extracted {len(embeddings_df):,} embeddings")
                            print(f"✓ Embeddings saved to: {os.path.join(config.output_dir, config.embedding_output_file)}")
                except Exception as load_error:
                    print(f"⚠️ Error loading best model: {str(load_error)}")
            else:
                print(f"⚠️ Best model checkpoint not found at {best_model_path}")
        
        except Exception as train_error:
            print(f"❌ Error during training: {str(train_error)}")
            import traceback
            traceback.print_exc()
        
        # Hardware usage statistics
        print("\n💻 Hardware Usage Statistics")
        print("-" * 40)
        
        if torch.cuda.is_available():
            print("GPU Memory Usage:")
            print(f"  - Peak Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"  - Peak Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
            print(f"  - Current Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  - Current Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            
            # Reset peak stats
            torch.cuda.reset_peak_memory_stats()
        
        # Report CPU usage if psutil is available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1e9  # In GB
            cpu_percent = process.cpu_percent()
            print(f"CPU Usage: {cpu_percent:.1f}%")
            print(f"Memory Usage: {memory_usage:.2f} GB")
        except ImportError:
            print("Note: Install 'psutil' for CPU/memory usage statistics")
        
        # Calculate total runtime
        script_end_time = time.time()
        total_duration = script_end_time - script_start_time
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n⏱️ Total script execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"\n✅ Training script completed successfully!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Unhandled error in main script execution:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to write error to log file
        try:
            error_log_path = "./training_error.log"
            with open(error_log_path, 'w') as error_file:
                error_file.write(f"Error occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                error_file.write(f"Error: {str(e)}\n\n")
                traceback.print_exc(file=error_file)
            print(f"Error details saved to {error_log_path}")
        except:
            print("Could not save error details to log file")
            
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()