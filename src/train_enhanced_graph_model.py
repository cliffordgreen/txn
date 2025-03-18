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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
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


class ParquetTransactionDataset(Dataset):
    def __init__(self, parquet_files, preprocess_fn=None, transform_fn=None):
        self.parquet_files = parquet_files
        self.preprocess_fn = preprocess_fn
        self.transform_fn = transform_fn
        
        # Get total number of rows across all files
        self.file_row_counts = []
        self.total_rows = 0
        
        for file in tqdm(parquet_files, desc="Counting rows"):
            # For very large files, just read metadata or count rows efficiently
            try:
                # Try pyarrow method to get row count
                import pyarrow.parquet as pq
                row_count = pq.read_metadata(file).num_rows
            except:
                try:
                    # Fall back to pandas with just column names
                    row_count = pd.read_parquet(file, columns=['txn_id']).shape[0]
                except:
                    # Last resort: skip this file
                    print(f"Warning: Could not determine row count for {file}, skipping")
                    continue
                
            self.file_row_counts.append(row_count)
            self.total_rows += row_count
            
        # Create lookup table for file index and row index
        self.lookup = []
        for file_idx, row_count in enumerate(self.file_row_counts):
            self.lookup.extend([(file_idx, row_idx) for row_idx in range(row_count)])
        
        print(f"Dataset initialized with {len(parquet_files)} files and {self.total_rows} total rows")
        
    def __len__(self):
        return self.total_rows
    
    def _read_row_from_file(self, file_path, row_idx):
        """Read a specific row from a file without loading the entire file"""
        try:
            # First try to use fast row access
            table = pd.read_parquet(file_path, engine='pyarrow')
            row = table.iloc[row_idx:row_idx+1]
            
            if row.empty:
                raise ValueError(f"Row {row_idx} not found in {file_path}")
                
            return row
        except Exception as e:
            print(f"Error reading row {row_idx} from {file_path}: {str(e)}")
            return pd.DataFrame()
        
    def __getitem__(self, idx):
        if idx >= len(self.lookup):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.lookup)} items")
            
        file_idx, row_idx = self.lookup[idx]
        file_path = self.parquet_files[file_idx]
        
        df = self._read_row_from_file(file_path, row_idx)
        
        if df.empty:
            return pd.DataFrame()
        
        if self.preprocess_fn:
            df = self.preprocess_fn(df)
            
        if self.transform_fn:
            return self.transform_fn(df)
        else:
            return df
        
    def get_sample_batch(self, sample_size=100):
        """Get a small sample batch for metadata and model initialization"""
        print(f"Getting sample batch of {sample_size} rows...")
        
        # Use a small sample size to avoid memory issues
        sample_size = min(sample_size, self.total_rows)
        
        # Sample evenly across files
        rows_per_file = max(1, sample_size // len(self.parquet_files))
        
        sample_dfs = []
        
        for file_idx, file_path in enumerate(self.parquet_files):
            if file_idx >= len(self.file_row_counts):
                continue
                
            row_count = self.file_row_counts[file_idx]
            if row_count == 0:
                continue
                
            # Sample rows from this file
            try:
                # Try to read just a few rows from the start of the file
                sample_rows = min(rows_per_file, row_count)
                file_df = pd.read_parquet(file_path, engine='pyarrow').head(sample_rows)
                
                if self.preprocess_fn:
                    file_df = self.preprocess_fn(file_df)
                    
                sample_dfs.append(file_df)
                
                # Stop if we have enough samples
                if sum(len(df) for df in sample_dfs) >= sample_size:
                    break
            except Exception as e:
                print(f"Error sampling from {file_path}: {str(e)}")
                continue
                
        if not sample_dfs:
            raise ValueError("Could not get any sample data from the dataset")
            
        return pd.concat(sample_dfs, ignore_index=True)
        
    def get_batch_df(self, indices):
        """Get a batch of rows as a single DataFrame"""
        dfs = []
        for idx in indices:
            try:
                df = self.__getitem__(idx)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"Error getting item {idx}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("All dataframes in batch were empty")
            
        return pd.concat(dfs, ignore_index=True)


def get_parquet_files(data_dir, max_files=None):
    """Get list of parquet files from directory"""
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    if max_files is not None and len(parquet_files) > max_files:
        parquet_files = parquet_files[:max_files]
        
    print(f"Found {len(parquet_files)} parquet files")
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
    # Use the model's data preparation function
    data = model.prepare_data_from_dataframe(batch_df)
    
    # Move tensors to the correct device
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    
    # Extract labels
    labels = {
        'category': torch.tensor(batch_df['category_id'].values, dtype=torch.long).to(device)
    }
    
    if 'tax_type_id' in batch_df.columns:
        labels['tax_type'] = torch.tensor(batch_df['tax_type_id'].values, dtype=torch.long).to(device)
    
    return data, labels


def create_cuda_graph(model, sample_data):
    """Create CUDA graph for model inference"""
    if not torch.cuda.is_available() or not config.use_cuda_graphs:
        return None
    
    print("Creating CUDA graph for optimized inference...")
    
    # Extract required inputs from sample data
    inputs = {
        'x': sample_data['x'],
        'edge_index': sample_data['edge_index'],
        'edge_type': sample_data['edge_type'],
        'edge_attr': sample_data['edge_attr'],
        'seq_features': sample_data['seq_features'],
        'timestamps': sample_data['timestamps'],
        'tabular_features': sample_data['tabular_features'],
        't0': sample_data['t0'],
        't1': sample_data['t1'],
        'company_features': sample_data['company_features'],
        'company_ids': sample_data['company_ids'],
        'batch_size': sample_data['batch_size'],
        'seq_len': sample_data['seq_len']
    }
    
    # Ensure all inputs are on the correct device
    device = next(model.parameters()).device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    
    # Create static input batch
    static_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            static_inputs[k] = v.clone()
        else:
            static_inputs[k] = v
    
    # Set model to eval mode
    model.eval()
    
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            model(**static_inputs)
    
    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_outputs = model(**static_inputs)
    
    return {
        'graph': g,
        'static_inputs': static_inputs,
        'static_outputs': static_outputs
    }


def run_with_cuda_graph(cuda_graph, data):
    """Run inference using CUDA graph"""
    # Update static inputs with new data
    for k, v in data.items():
        if k in cuda_graph['static_inputs'] and isinstance(v, torch.Tensor):
            cuda_graph['static_inputs'][k].copy_(v)
    
    # Run the graph
    cuda_graph['graph'].replay()
    
    # Return outputs
    return cuda_graph['static_outputs']


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
    """Evaluate the model on the given dataset"""
    model.eval()
    
    total_loss = 0
    total_acc = 0
    samples_processed = 0
    
    with torch.no_grad():
        for batch_indices in tqdm(dataloader, desc="Evaluation", leave=False):
            # Get batch dataframe
            batch_df = dataset.get_batch_df(batch_indices)
            
            # Prepare inputs
            data, labels = prepare_model_inputs(batch_df, model, device)
            
            # Forward pass
            if cuda_graph is not None and config.use_cuda_graphs:
                # Use CUDA graph for faster inference
                outputs = run_with_cuda_graph(cuda_graph, data)
            else:
                # Regular forward pass
                outputs = model(
                    x=data['x'],
                    edge_index=data['edge_index'],
                    edge_type=data['edge_type'],
                    edge_attr=data['edge_attr'],
                    seq_features=data['seq_features'],
                    timestamps=data['timestamps'],
                    tabular_features=data['tabular_features'],
                    t0=data['t0'],
                    t1=data['t1'],
                    company_features=data['company_features'],
                    company_ids=data['company_ids'],
                    batch_size=data['batch_size'],
                    seq_len=data['seq_len']
                )
            
            # Compute loss
            if isinstance(outputs, tuple):
                category_logits, tax_type_logits = outputs
                category_loss = nn.CrossEntropyLoss()(category_logits, labels['category'])
                
                if 'tax_type' in labels:
                    tax_type_loss = nn.CrossEntropyLoss()(tax_type_logits, labels['tax_type'])
                    loss = 0.7 * category_loss + 0.3 * tax_type_loss
                else:
                    loss = category_loss
                    
                # Compute accuracy
                preds = category_logits.argmax(dim=1)
                acc = (preds == labels['category']).float().mean().item()
            else:
                # Single task model
                loss = nn.CrossEntropyLoss()(outputs, labels['category'])
                preds = outputs.argmax(dim=1)
                acc = (preds == labels['category']).float().mean().item()
            
            # Update metrics
            total_loss += loss.item() * len(batch_indices)
            total_acc += acc * len(batch_indices)
            samples_processed += len(batch_indices)
    
    # Calculate average metrics
    avg_loss = total_loss / samples_processed
    avg_acc = total_acc / samples_processed
    
    return avg_loss, avg_acc


def train(model, train_dataset, val_dataset, config, device):
    """Train the model with the given datasets"""
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=config.patience // 2,
        verbose=True
    )
    
    # Mixed precision
    scaler = GradScaler() if config.use_amp else None
    
    # Create train dataloader with random sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    # Initialize metrics tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initial CUDA graph (will be updated later)
    cuda_graph = None
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        model.train()
        
        epoch_loss = 0
        epoch_acc = 0
        samples_processed = 0
        
        # Progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for step, batch_indices in pbar:
            # Get batch dataframe
            batch_df = train_dataset.get_batch_df(batch_indices)
            
            # Prepare inputs
            data, labels = prepare_model_inputs(batch_df, model, device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if config.use_amp:
                with autocast():
                    outputs = model(
                        x=data['x'],
                        edge_index=data['edge_index'],
                        edge_type=data['edge_type'],
                        edge_attr=data['edge_attr'],
                        seq_features=data['seq_features'],
                        timestamps=data['timestamps'],
                        tabular_features=data['tabular_features'],
                        t0=data['t0'],
                        t1=data['t1'],
                        company_features=data['company_features'],
                        company_ids=data['company_ids'],
                        batch_size=data['batch_size'],
                        seq_len=data['seq_len']
                    )
                    
                    # Compute loss
                    if isinstance(outputs, tuple):
                        category_logits, tax_type_logits = outputs
                        category_loss = nn.CrossEntropyLoss()(category_logits, labels['category'])
                        
                        if 'tax_type' in labels:
                            tax_type_loss = nn.CrossEntropyLoss()(tax_type_logits, labels['tax_type'])
                            loss = 0.7 * category_loss + 0.3 * tax_type_loss
                        else:
                            loss = category_loss
                            
                        # Compute accuracy
                        preds = category_logits.argmax(dim=1)
                        acc = (preds == labels['category']).float().mean().item()
                    else:
                        # Single task model
                        loss = nn.CrossEntropyLoss()(outputs, labels['category'])
                        preds = outputs.argmax(dim=1)
                        acc = (preds == labels['category']).float().mean().item()
                    
                # Backward and optimize with mixed precision
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Without mixed precision
                outputs = model(
                    x=data['x'],
                    edge_index=data['edge_index'],
                    edge_type=data['edge_type'],
                    edge_attr=data['edge_attr'],
                    seq_features=data['seq_features'],
                    timestamps=data['timestamps'],
                    tabular_features=data['tabular_features'],
                    t0=data['t0'],
                    t1=data['t1'],
                    company_features=data['company_features'],
                    company_ids=data['company_ids'],
                    batch_size=data['batch_size'],
                    seq_len=data['seq_len']
                )
                
                # Compute loss
                if isinstance(outputs, tuple):
                    category_logits, tax_type_logits = outputs
                    category_loss = nn.CrossEntropyLoss()(category_logits, labels['category'])
                    
                    if 'tax_type' in labels:
                        tax_type_loss = nn.CrossEntropyLoss()(tax_type_logits, labels['tax_type'])
                        loss = 0.7 * category_loss + 0.3 * tax_type_loss
                    else:
                        loss = category_loss
                        
                    # Compute accuracy
                    preds = category_logits.argmax(dim=1)
                    acc = (preds == labels['category']).float().mean().item()
                else:
                    # Single task model
                    loss = nn.CrossEntropyLoss()(outputs, labels['category'])
                    preds = outputs.argmax(dim=1)
                    acc = (preds == labels['category']).float().mean().item()
                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item() * len(batch_indices)
            epoch_acc += acc * len(batch_indices)
            samples_processed += len(batch_indices)
            
            # Update progress bar
            if step % config.log_steps == 0:
                pbar.set_description(
                    f"Train Loss: {epoch_loss / samples_processed:.4f}, "
                    f"Acc: {epoch_acc / samples_processed:.4f}"
                )
            
            # Validation during epoch
            if step > 0 and step % config.eval_steps == 0:
                # Switch to eval mode
                model.eval()
                
                # Initialize CUDA graph for faster inference if needed
                if config.use_cuda_graphs and cuda_graph is None and torch.cuda.is_available():
                    cuda_graph = create_cuda_graph(model, data)
                
                val_loss, val_acc = evaluate(model, val_loader, val_dataset, device, cuda_graph)
                
                print(f"Step {step}/{len(train_loader)}, "
                      f"Train Loss: {epoch_loss / samples_processed:.4f}, "
                      f"Train Acc: {epoch_acc / samples_processed:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
                
                # Switch back to train mode
                model.train()
        
        # End of epoch evaluation
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, val_dataset, device, cuda_graph)
        
        # Calculate epoch metrics
        train_loss = epoch_loss / samples_processed
        train_acc = epoch_acc / samples_processed
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch + 1}/{config.num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
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
            
            print(f"Saved new best model at epoch {epoch + 1} with validation loss {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best val loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            
            if patience_counter >= config.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Save checkpoint
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'config': config.__dict__
            },
            os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }


def extract_embeddings_for_xgboost(model, dataset, output_file, config, device):
    """Extract node embeddings for XGBoost integration"""
    model.eval()
    
    embeddings_list = []
    labels_list = []
    transaction_ids = []
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch_indices in tqdm(dataloader, desc="Extracting embeddings"):
            # Get batch dataframe
            batch_df = dataset.get_batch_df(batch_indices)
            
            # Prepare inputs
            data, labels = prepare_model_inputs(batch_df, model, device)
            
            # Extract embeddings
            embeddings = model.extract_embeddings(data)
            
            # Store embeddings and labels
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels['category'].cpu().numpy())
            
            if 'txn_id' in batch_df.columns:
                transaction_ids.extend(batch_df['txn_id'].tolist())
    
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


def plot_training_curves(training_history, config):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_losses'], label='Train Loss')
    plt.plot(training_history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(training_history['train_accs'], label='Train Accuracy')
    plt.plot(training_history['val_accs'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'training_curves.png'))
    plt.close()


def configure_for_hardware(config):
    """Configure settings based on available hardware"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_name = torch.cuda.get_device_name(0)
        
        # For p3.2xlarge instances with V100 GPU
        if 'V100' in gpu_name:
            print("Using p3.2xlarge configuration with V100 GPU")
            config.batch_size = 128
            config.hidden_dim = 512
            config.use_amp = True
            config.use_cuda_graphs = True
            
        # For g4dn.xlarge instances with T4 GPU
        elif 'T4' in gpu_name:
            print("Using g4dn.xlarge configuration with T4 GPU")
            config.batch_size = 64
            config.hidden_dim = 256
            config.use_amp = True
            config.use_cuda_graphs = True
            
        # For H100 GPUs
        elif 'H100' in gpu_name:
            print("Using configuration for H100 GPU")
            config.batch_size = 256
            config.hidden_dim = 1024
            config.use_amp = True
            config.use_cuda_graphs = True
            
        # For other GPUs
        else:
            print(f"Using configuration for {gpu_name}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_mem > 20:
                # For GPUs with more than 20GB memory
                config.batch_size = 128
                config.hidden_dim = 512
            elif total_mem > 10:
                # For GPUs with more than 10GB memory
                config.batch_size = 64
                config.hidden_dim = 256
            else:
                # For smaller GPUs
                config.batch_size = 32
                config.hidden_dim = 128
                
        # Print GPU info
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU configuration")
        config.batch_size = 16
        config.hidden_dim = 64
        config.use_amp = False
        config.use_cuda_graphs = False
        config.num_workers = 2
        
    # Ensure cuda_graph_batch_size is set
    config.cuda_graph_batch_size = config.batch_size
    
    return device, config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Enhanced Hybrid Transaction Model')
    parser.add_argument('--data_dir', type=str, default=None, 
                        help='Directory containing parquet files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save model outputs')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of parquet files to process')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    
    return parser.parse_args()


def main():
    """Main training script entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration
    config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_files:
        config.max_files = args.max_files
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    
    # Configure for available hardware
    device, config = configure_for_hardware(config)
    print(f"Using device: {device}")
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in vars(config).items():
        if not key.startswith('__'):
            print(f"{key}: {value}")
    
    # Get parquet files
    parquet_files = get_parquet_files(config.data_dir, config.max_files)
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {config.data_dir}")
    
    # Split files into train and validation sets
    train_files, val_files = train_test_split(parquet_files, test_size=0.2, random_state=42)
    print(f"Training on {len(train_files)} files, validating on {len(val_files)} files")
    
    # Create datasets
    train_dataset = ParquetTransactionDataset(train_files, preprocess_fn=preprocess_transactions)
    val_dataset = ParquetTransactionDataset(val_files, preprocess_fn=preprocess_transactions)
    
    print(f"Train dataset: {len(train_dataset)} transactions")
    print(f"Validation dataset: {len(val_dataset)} transactions")
    
    # Sample a small batch to get category and tax type counts
    print("Getting metadata from sample batch...")
    sample_df = train_dataset.get_sample_batch(sample_size=100)
    
    num_categories = sample_df['category_id'].nunique() if 'category_id' in sample_df.columns else 0
    num_tax_types = sample_df['tax_type_id'].nunique() if 'tax_type_id' in sample_df.columns else 1
    
    print(f"Number of unique categories: {num_categories}")
    print(f"Number of unique tax types: {num_tax_types}")
    
    # If no categories were found in the sample, use reasonable defaults
    if num_categories == 0:
        print("Warning: No categories found in sample. Using default value of 400.")
        num_categories = 400
    
    if num_tax_types == 0:
        print("Warning: No tax types found in sample. Using default value of 20.")
        num_tax_types = 20
    
    # Initialize model
    model = initialize_model(config.hidden_dim, num_categories, num_tax_types, config, device)
    print(model)
    
    # Train the model
    start_time = time.time()
    training_history = train(model, train_dataset, val_dataset, config, device)
    end_time = time.time()
    
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best epoch: {training_history['best_epoch'] + 1} with validation loss {training_history['best_val_loss']:.4f}")
    
    # Plot training curves
    plot_training_curves(training_history, config)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1} with validation loss {checkpoint['val_loss']:.4f}")
    
    # Extract embeddings for XGBoost if requested
    if config.extract_embeddings:
        print("Extracting embeddings for XGBoost integration...")
        embeddings_df = extract_embeddings_for_xgboost(model, train_dataset, config.embedding_output_file, config, device)
        
        # Show embedding dimensions
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
        print(f"\nEmbedding dimension: {len(embedding_cols)}")
    
    # Memory analysis
    if torch.cuda.is_available():
        print("\nGPU Memory Analysis:")
        print(f"Peak Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Peak Memory Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error in main: {str(e)}")
        traceback.print_exc()