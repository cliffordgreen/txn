#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from torch.nn.parameter import UninitializedParameter

# Add UninitializedParameter to safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([UninitializedParameter])

# Add project root to path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
# Helper function to check if parameter is initialized
def is_initialized_parameter(p):
    """Check if a parameter is initialized"""
    if isinstance(p, UninitializedParameter):
        return False
    try:
        p.numel()  # Will raise an error if uninitialized
        return True
    except:
        return False

# Import custom modules
from src.models.hybrid_transaction_model import EnhancedHybridTransactionModel
from src.data_processing.transaction_graph import build_transaction_relationship_graph
from torch.utils.data import Dataset, DataLoader
from src.utils.model_utils import configure_for_hardware
from src.train_streamlined_graph_model import (
    Config, 
    ParquetTransactionDataset, 
    df_collate_fn, 
    get_parquet_files, 
    preprocess_transactions,
    prepare_model_inputs
)

class EvalConfig(Config):
    """Configuration class for evaluation, inherits from training Config"""
    def __init__(self):
        super().__init__()
        # Evaluation specific settings
        self.model_path = "../models/enhanced_model_output/best_model.pt"
        self.eval_data_dir = "../data/parquet_files"
        self.results_dir = "../evaluation_results"
        self.batch_size = 32  # Smaller batch size for evaluation
        self.generate_plots = True
        self.save_predictions = True
        self.verbose = True
        self.ignore_missing_fields = True
        self.prediction_output_file = "predictions.csv"
        self.report_output_file = "evaluation_report.json"
        
        # Feature extraction
        self.extract_features = True
        self.features_output_file = "extracted_features.pkl"

def load_model(model_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model onto
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    print(f"Loading model from: {model_path}")
    
    # Ensure UninitializedParameter is added to safe globals for PyTorch 2.6+ compatibility
    try:
        if hasattr(torch.serialization, 'is_safe_global'):
            if not torch.serialization.is_safe_global(UninitializedParameter):
                print("Adding UninitializedParameter to safe globals for PyTorch 2.6+ compatibility")
                torch.serialization.add_safe_globals([UninitializedParameter])
        else:
            # For older PyTorch versions, just add it directly
            print("Adding UninitializedParameter to safe globals for PyTorch compatibility")
            torch.serialization.add_safe_globals([UninitializedParameter])
    except Exception as e:
        print(f"Warning: Could not verify UninitializedParameter in safe globals: {str(e)}")
    
    # Load checkpoint with weights_only=False to handle PyTorch 2.6+ changes
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Model loaded with weights_only=False parameter")
    except TypeError:
        # For older PyTorch versions that don't have weights_only parameter
        print("Falling back to legacy loading method (PyTorch < 2.0)")
        checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    if 'config' in checkpoint:
        print("Found configuration in checkpoint")
        config_dict = checkpoint['config']
        config = Config()
        for key, value in config_dict.items():
            if not key.startswith('__'):
                setattr(config, key, value)
        print(f"Loaded configuration with hidden_dim={config.hidden_dim}")
    else:
        # Create default config if not in checkpoint
        config = Config()
        print("Warning: No configuration found in checkpoint. Using defaults.")
    
    # Initialize model architecture
    num_categories = 400  # Default, to be updated based on actual data
    num_tax_types = 20    # Default, to be updated based on actual data
    
    # Set default values for parameters that might not be in the config
    if not hasattr(config, 'num_relations'):
        config.num_relations = 5
        print("Setting default num_relations=5")
    
    if not hasattr(config, 'multi_task'):
        config.multi_task = True
        print("Setting default multi_task=True")
    
    # Create the model with the same architecture as the saved model
    model = EnhancedHybridTransactionModel(
        input_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim,
        output_dim=num_categories,
        num_heads=getattr(config, 'num_heads', 4),  # Default to 4 if not specified
        num_graph_layers=getattr(config, 'num_graph_layers', 2),
        num_temporal_layers=getattr(config, 'num_temporal_layers', 2),
        dropout=getattr(config, 'dropout', 0.2),
        use_hyperbolic=getattr(config, 'use_hyperbolic', True),
        use_neural_ode=getattr(config, 'use_neural_ode', False),
        use_text=getattr(config, 'use_text', False),
        multi_task=getattr(config, 'multi_task', True),
        tax_type_dim=num_tax_types,
        num_relations=getattr(config, 'num_relations', 5),
        graph_weight=0.6,
        temporal_weight=0.4,
        use_dynamic_weighting=True
    ).to(device)
    
    # Get model parameter count safely
    try:
        param_count = sum(p.numel() for p in model.parameters() if is_initialized_parameter(p))
        print(f"Model created with {param_count:,} parameters")
    except Exception as e:
        print(f"Could not count parameters: {str(e)}")
        
    # Ensure the model is fully initialized before loading state dict
    print("Initializing any uninitialized parameters...")
    for module in model.modules():
        if hasattr(module, '_init_weights') and callable(module._init_weights):
            module._init_weights()
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Model state loaded successfully with strict=True")
    except Exception as e:
        print(f"Error loading with strict=True: {str(e)}")
        print("Trying with strict=False...")
        missing_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if hasattr(missing_keys, 'missing_keys') and missing_keys.missing_keys:
            print(f"Warning: Missing keys: {missing_keys.missing_keys}")
        if hasattr(missing_keys, 'unexpected_keys') and missing_keys.unexpected_keys:
            print(f"Warning: Unexpected keys: {missing_keys.unexpected_keys}")
    
    # Set model to evaluation mode
    model.eval()
    print("Model set to evaluation mode")
    
    return model, config

def evaluate_model(model, dataset, device, config, return_predictions=False):
    """
    Evaluate the model on a dataset
    
    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        config: Evaluation configuration
        return_predictions: Whether to return predictions and labels
        
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Predicted labels (if return_predictions=True)
        true_labels: True labels (if return_predictions=True)
    """
    model.eval()
    
    # Two-pass approach:
    # 1. First pass: Run predictions on the ENTIRE dataset regardless of labels
    # 2. Second pass (if needed): Compute metrics only on samples with labels
    
    # Modified prepare_model_inputs that doesn't require labels
    def prepare_model_inputs_no_labels(batch_df, model, device):
        """Prepare model inputs without requiring labels"""
        if batch_df.empty:
            raise ValueError("Empty DataFrame provided to prepare_model_inputs")
        
        # Fill missing values to prevent errors in model.prepare_data_from_dataframe
        # Clone the dataframe to avoid modifying the original
        processed_df = batch_df.copy()
        
        # Fill nulls for numeric columns
        for col in processed_df.select_dtypes(include=['number']).columns:
            if processed_df[col].isna().any():
                processed_df[col] = processed_df[col].fillna(0)
                
        # Fill nulls for categorical and string columns
        for col in processed_df.select_dtypes(include=['category', 'object']).columns:
            if processed_df[col].isna().any():
                processed_df[col] = processed_df[col].fillna('UNKNOWN')
                
        # Fill nulls for datetime columns
        for col in processed_df.select_dtypes(include=['datetime']).columns:
            if processed_df[col].isna().any():
                # Use median of non-null values or current time
                non_null = processed_df[col].dropna()
                if len(non_null) > 0:
                    processed_df[col] = processed_df[col].fillna(non_null.median())
                else:
                    processed_df[col] = processed_df[col].fillna(pd.Timestamp('now'))
        
        # Use the model's data preparation function
        try:
            data = model.prepare_data_from_dataframe(processed_df)
        except Exception as e:
            if config.verbose:
                print(f"Error in model.prepare_data_from_dataframe: {str(e)}")
            
            # Create fallback data with minimal structure to ensure predictions still run
            # This is a simplified version - we should add proper fallback data based on model requirements
            fallback_dim = getattr(model, 'hidden_dim', 128)
            data = {
                'x': torch.zeros((len(processed_df), fallback_dim), device=device),
                'edge_index': torch.zeros((2, 1), dtype=torch.long, device=device),
                'edge_type': torch.zeros((1,), dtype=torch.long, device=device),
                'edge_attr': torch.zeros((1, 1), device=device)
            }
            
            if hasattr(model, 'prepare_data_from_dataframe'):
                # Check if the model has required fields and create placeholders
                if hasattr(model, 'uses_seq_features') and model.uses_seq_features:
                    seq_len = 5  # Default sequence length
                    batch_size = len(processed_df) // seq_len + (1 if len(processed_df) % seq_len > 0 else 0)
                    data['seq_features'] = torch.zeros((batch_size, seq_len, fallback_dim), device=device)
                    data['timestamps'] = torch.zeros((batch_size, seq_len), device=device)
                
                if hasattr(model, 'uses_tabular_features') and model.uses_tabular_features:
                    data['tabular_features'] = torch.zeros((len(processed_df), fallback_dim), device=device)
        
        # Move tensors to device (if not already)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.device != device:
                    data[key] = value.to(device, non_blocking=True)
            elif isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value).to(device, non_blocking=True)
        
        # Prepare labels dictionary if category_id is available
        labels = {}
        
        if 'category_id' in processed_df.columns:
            # Get actual batch size from seq_features if available
            seq_batch_size = data['seq_features'].size(0) if 'seq_features' in data else processed_df.shape[0]
            
            # Handle category_id
            max_idx = min(seq_batch_size, len(processed_df))
            if max_idx > 0:  # Only process if we have data
                category_values = processed_df['category_id'].values[:max_idx]
                
                # Convert to numeric if needed
                if pd.api.types.is_object_dtype(category_values):
                    category_codes, _ = pd.factorize(category_values)
                    category_values = category_codes
                
                category_tensor = torch.tensor(
                    category_values, 
                    dtype=torch.long,
                    device=device
                )
                labels['category'] = category_tensor
        
        return data, labels
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=df_collate_fn
    )
    
    # Initialize metrics and tracking variables
    total_loss = 0
    category_correct = 0
    category_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_txn_ids = []
    
    print(f"Starting evaluation on {len(dataset)} total transactions")
    
    # Track total processed transactions
    total_processed = 0
    total_with_valid_labels = 0
    total_with_predictions = 0
    
    # Evaluate without gradients
    with torch.no_grad():
        # Process each batch
        for batch_idx, batch_indices in enumerate(tqdm(dataloader, desc="Evaluation", disable=not config.verbose)):
            if not batch_indices:
                continue
                
            # Extract actual indices for this batch
            start_idx = batch_indices[0]
            end_idx = start_idx + len(batch_indices)
            actual_indices = list(range(start_idx, min(end_idx, len(dataset))))
            
            if not actual_indices:
                continue
            
            # Get batch dataframe
            try:
                batch_df = dataset.get_batch_df(actual_indices)
                batch_size = len(batch_df)
                total_processed += batch_size
                
                # Store transaction IDs if available
                if 'txn_id' in batch_df.columns:
                    txn_ids = batch_df['txn_id'].tolist()
                    all_txn_ids.extend(txn_ids)
                else:
                    # Create placeholder txn_ids if not available
                    txn_ids = [f"txn_{i}" for i in actual_indices]
                    all_txn_ids.extend(txn_ids)
            except Exception as e:
                if config.verbose:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
            
            # Prepare inputs WITHOUT requiring labels - never skip a batch
            success = True
            try:
                data, labels = prepare_model_inputs_no_labels(batch_df, model, device)
                
                # Check if labels are available
                has_category_label = 'category' in labels and labels['category'] is not None
                if has_category_label:
                    total_with_valid_labels += len(labels['category'])
                
                # Replace NaN/Inf in input tensors for stability
                for key, tensor in data.items():
                    if isinstance(tensor, torch.Tensor) and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                        if config.verbose and (batch_idx < 5 or batch_idx % 20 == 0):  # Limit logging
                            nan_count = torch.isnan(tensor).sum().item()
                            inf_count = torch.isinf(tensor).sum().item()
                            if nan_count > 0 or inf_count > 0:
                                print(f"Fixing {nan_count} NaN and {inf_count} Inf values in tensor '{key}' with shape {tensor.shape}")
                        data[key] = torch.nan_to_num(tensor, nan=0.0)
            except Exception as e:
                if config.verbose:
                    print(f"Error preparing inputs for batch {batch_idx}: {str(e)}")
                    if "category_id" in str(e) and 'category_id' not in batch_df.columns:
                        print(f"Missing category_id column in batch, available columns: {batch_df.columns.tolist()}")
                
                # Create minimal fallback data to ensure we can still make predictions
                # The prepare_model_inputs_no_labels function now has fallback handling, but this is an extra safety check
                fallback_dim = getattr(model, 'hidden_dim', 128)
                
                # Simple fallback data structure that should work with most models
                data = {
                    'x': torch.zeros((len(batch_df), fallback_dim), device=device),
                    'edge_index': torch.zeros((2, 1), dtype=torch.long, device=device),
                    'edge_type': torch.zeros((1,), dtype=torch.long, device=device)
                }
                labels = {}
                has_category_label = False
                # Mark this batch as having issues
                success = False
            
            # Forward pass
            try:
                # Only attempt forward pass if we have valid data
                if success:
                    outputs = model(**data)
                    
                    # Process predictions regardless of labels
                    if isinstance(outputs, tuple):
                        # Multi-task model (category and tax type)
                        category_logits, _ = outputs
                        
                        # Check and sanitize outputs if they contain NaN/Inf
                        if torch.isnan(category_logits).any() or torch.isinf(category_logits).any():
                            category_logits = torch.nan_to_num(category_logits, nan=0.0)
                        
                        # Get predictions and probabilities
                        category_probs = torch.softmax(category_logits, dim=1)
                        category_preds = category_logits.argmax(dim=1)
                        
                        # Store probabilities and predictions
                        all_probs.extend(category_probs.cpu().numpy())
                        batch_preds = category_preds.cpu().numpy()
                        all_preds.extend(batch_preds)
                        total_with_predictions += len(batch_preds)
                        
                        # Process labels if available
                        if has_category_label:
                            # Calculate loss
                            category_loss = nn.CrossEntropyLoss(reduction='sum')(category_logits, labels['category'])
                            
                            # Calculate accuracy metrics
                            correct = (category_preds == labels['category']).sum().item()
                            category_correct += correct
                            category_total += len(labels['category'])
                            
                            # Store labels for metrics
                            batch_labels = labels['category'].cpu().numpy()
                            all_labels.extend(batch_labels)
                            
                            # Update total loss
                            total_loss += category_loss.item()
                        else:
                            # Add placeholder NaN labels to maintain alignment with predictions
                            all_labels.extend([float('nan')] * len(batch_preds))
                        
                    else:
                        # Single task model
                        # Check and sanitize outputs if they contain NaN/Inf
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            outputs = torch.nan_to_num(outputs, nan=0.0)
                        
                        # Get predictions and probabilities
                        category_probs = torch.softmax(outputs, dim=1)
                        category_preds = outputs.argmax(dim=1)
                        
                        # Store probabilities and predictions
                        all_probs.extend(category_probs.cpu().numpy())
                        batch_preds = category_preds.cpu().numpy()
                        all_preds.extend(batch_preds)
                        total_with_predictions += len(batch_preds)
                        
                        # Process labels if available
                        if has_category_label:
                            # Calculate loss
                            loss = nn.CrossEntropyLoss(reduction='sum')(outputs, labels['category'])
                            
                            # Calculate accuracy metrics
                            correct = (category_preds == labels['category']).sum().item()
                            category_correct += correct
                            category_total += len(labels['category'])
                            
                            # Store labels for metrics
                            batch_labels = labels['category'].cpu().numpy()
                            all_labels.extend(batch_labels)
                            
                            # Update total loss
                            total_loss += loss.item()
                        else:
                            # Add placeholder NaN labels to maintain alignment with predictions
                            all_labels.extend([float('nan')] * len(batch_preds))
                else:
                    # Create a minimal set of predictions for the batch
                    # Just use zero predictions rather than random values
                    batch_size = len(batch_df)
                    
                    # Add txn_ids but skip generating actual predictions
                    # This way we'll have entries for all transactions but no actual predictions
                    # for the ones that couldn't be processed properly
                    
                    # Just track that these txn_ids were processed
                    total_with_predictions += batch_size
                    
                    if config.verbose:
                        print(f"Skipping predictions for batch {batch_idx} due to data preparation failure")
                
            except Exception as e:
                if config.verbose:
                    print(f"Error during forward pass for batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Track that we at least tried to process these transactions
                batch_size = len(batch_df)
                
                # Just record that we attempted to process this batch
                total_with_predictions += batch_size
                
                if config.verbose:
                    print(f"Skipping predictions for batch {batch_idx} due to forward pass failure")
    
    # Calculate final metrics
    metrics = {}
    
    print(f"Processed {total_processed}/{len(dataset)} transactions")
    print(f"Found {total_with_valid_labels} transactions with valid labels")
    print(f"Generated {len(all_preds)} predictions and {len(all_txn_ids)} transaction IDs")
    
    if category_total > 0:
        avg_loss = total_loss / category_total
        category_accuracy = category_correct / category_total
        
        metrics['loss'] = avg_loss
        metrics['accuracy'] = category_accuracy
        metrics['total_processed'] = total_processed
        metrics['total_with_labels'] = total_with_valid_labels
        metrics['total_with_predictions'] = len(all_preds)
        
        # Calculate additional metrics if we have predictions with labels
        # First filter out NaN labels for metric calculation
        valid_idx = [i for i, label in enumerate(all_labels) if not (isinstance(label, float) and np.isnan(label))]
        if valid_idx:
            valid_preds = [all_preds[i] for i in valid_idx]
            valid_labels = [all_labels[i] for i in valid_idx]
            
            if len(valid_preds) > 1 and len(valid_labels) > 1:
                metrics['f1_score'] = f1_score(valid_labels, valid_preds, average='weighted')
                metrics['precision'] = precision_score(valid_labels, valid_preds, average='weighted')
                metrics['recall'] = recall_score(valid_labels, valid_preds, average='weighted')
                
                # Compute class-wise metrics
                report = classification_report(valid_labels, valid_preds, output_dict=True)
                metrics['classification_report'] = report
    else:
        metrics['error'] = "No valid samples for evaluation"
        metrics['total_processed'] = total_processed
        metrics['total_with_labels'] = 0
        metrics['total_with_predictions'] = len(all_preds)
    
    if return_predictions:
        return metrics, all_preds, all_labels, all_probs, all_txn_ids
    
    return metrics

def extract_embeddings(model, dataset, device, config):
    """
    Extract embeddings from the model for feature analysis
    
    Args:
        model: Model to extract embeddings from
        dataset: Dataset to process
        device: Device to run extraction on
        config: Configuration
        
    Returns:
        embeddings_df: DataFrame with embeddings and metadata
    """
    model.eval()
    
    # Log model information for debugging dimension issues
    print(f"Model type: {type(model).__name__}")
    if hasattr(model, 'hidden_dim'):
        print(f"Model hidden_dim: {model.hidden_dim}")
    
    # Create dataloader with small batch size to avoid OOM
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=df_collate_fn
    )
    
    # Store embeddings and metadata
    embeddings_list = []
    labels_list = []
    txn_ids = []
    merchant_ids = []
    company_ids = []
    amounts = []
    processed_indices = []
    
    # Track the number of rows processed and successful extractions
    total_processed = 0
    successful_extractions = 0
    skipped_nulls = 0
    
    with torch.no_grad():
        for batch_indices in tqdm(dataloader, desc="Extracting embeddings", disable=not config.verbose):
            try:
                if not batch_indices:
                    continue
                    
                start_idx = batch_indices[0]
                end_idx = start_idx + len(batch_indices)
                actual_indices = list(range(start_idx, min(end_idx, len(dataset))))
                
                if not actual_indices:
                    continue
                
                total_processed += len(actual_indices)
                    
                # Get batch dataframe
                batch_df = dataset.get_batch_df(actual_indices)
                
                # Safety check: verify we have a valid batch_df
                if batch_df is None or len(batch_df) == 0:
                    print(f"Warning: Empty or None batch dataframe for indices {actual_indices}")
                    continue
                
                # Make a copy to avoid modifying the original data
                batch_df = batch_df.copy()
                
                # Fill null values in the dataframe before model processing
                # This follows the same approach as in the model's prepare_data_from_dataframe method
                # Check which columns have null values and report them
                null_columns = [col for col in batch_df.columns if batch_df[col].isna().any()]
                if config.verbose and null_columns:
                    null_counts = {col: batch_df[col].isna().sum() for col in null_columns}
                    print(f"Found columns with null values: {null_counts}")
                
                # Handle null values consistently with training preprocessing
                # Amount and numeric features
                if 'amount' in batch_df.columns and batch_df['amount'].isna().any():
                    print(f"Filling {batch_df['amount'].isna().sum()} null amount values with 0")
                    batch_df['amount'] = batch_df['amount'].fillna(0)
                
                # Handle all numeric columns consistently
                for col in batch_df.select_dtypes(include=['number']).columns:
                    if batch_df[col].isna().any():
                        null_count = batch_df[col].isna().sum()
                        batch_df[col] = batch_df[col].fillna(0)
                        if config.verbose:
                            print(f"Filled {null_count} nulls in numeric column '{col}' with 0")
                
                # Handle boolean columns
                for col in batch_df.select_dtypes(include=['bool']).columns:
                    if batch_df[col].isna().any():
                        null_count = batch_df[col].isna().sum()
                        batch_df[col] = batch_df[col].fillna(False)
                        if config.verbose:
                            print(f"Filled {null_count} nulls in boolean column '{col}' with False")
                
                # Handle categorical and object columns
                for col in batch_df.select_dtypes(include=['category', 'object']).columns:
                    if batch_df[col].isna().any():
                        null_count = batch_df[col].isna().sum()
                        # Special handling for company_type and similar fields
                        if col in ['company_type', 'company_size', 'merchant_category']:
                            batch_df[col] = batch_df[col].fillna('UNKNOWN')
                        elif col in ['merchant_id', 'company_id', 'user_id']:
                            # For ID columns, use 0 as a default ID value
                            batch_df[col] = batch_df[col].fillna(0)
                        else:
                            # Default handling for other categorical fields
                            batch_df[col] = batch_df[col].fillna('UNKNOWN')
                        if config.verbose:
                            print(f"Filled {null_count} nulls in column '{col}' with appropriate default")
                
                # Handle datetime columns - use the same logic from the model's _create_timestamps method
                datetime_cols = batch_df.select_dtypes(include=['datetime']).columns
                if len(datetime_cols) > 0:
                    for col in datetime_cols:
                        if batch_df[col].isna().any():
                            # Calculate median of non-null values
                            non_null_values = batch_df[col].dropna()
                            if len(non_null_values) > 0:
                                median_ts = non_null_values.median()
                                batch_df[col] = batch_df[col].fillna(median_ts)
                                if config.verbose:
                                    print(f"Filled {batch_df[col].isna().sum()} nulls in '{col}' with median timestamp")
                            else:
                                # If all values are null, use current time
                                import datetime
                                current_time = pd.Timestamp(datetime.datetime.now())
                                batch_df[col] = batch_df[col].fillna(current_time)
                                if config.verbose:
                                    print(f"Filled all nulls in '{col}' with current timestamp")
                
                # Verify no nulls remain
                remaining_nulls = batch_df.isna().sum().sum()
                if remaining_nulls > 0:
                    print(f"Warning: {remaining_nulls} null values remain after filling")
                    # Print columns with remaining nulls for debugging
                    remaining_null_cols = {col: batch_df[col].isna().sum() 
                                        for col in batch_df.columns 
                                        if batch_df[col].isna().any()}
                    print(f"Columns with remaining nulls: {remaining_null_cols}")
                    
                    # Instead of dropping rows, convert any remaining nulls more aggressively
                    for col in batch_df.columns:
                        if batch_df[col].isna().any():
                            print(f"Forcefully converting remaining nulls in '{col}' to appropriate defaults")
                            # Convert to string and replace nulls
                            batch_df[col] = batch_df[col].astype(str).fillna("UNKNOWN")
                            
                # Final verification - make absolutely sure no nulls remain
                if batch_df.isna().sum().sum() > 0:
                    print(f"CRITICAL: Still found {batch_df.isna().sum().sum()} nulls after aggressive conversion")
                    # As a last resort, drop problematic rows, but this should rarely happen
                    batch_df = batch_df.dropna()
                    if len(batch_df) == 0:
                        print("All rows contained null values that couldn't be filled, skipping batch")
                        continue
                
                # Prepare inputs
                try:
                    data, labels = prepare_model_inputs(batch_df, model, device)
                    
                    # Replace NaN/Inf in input tensors for stability
                    for key, tensor in data.items():
                        if isinstance(tensor, torch.Tensor) and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                            if config.verbose:
                                nan_count = torch.isnan(tensor).sum().item()
                                inf_count = torch.isinf(tensor).sum().item()
                                print(f"Fixing {nan_count} NaN and {inf_count} Inf values in tensor '{key}' with shape {tensor.shape}")
                            data[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                except Exception as e:
                    if config.verbose:
                        print(f"Error preparing inputs for embedding extraction: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    continue
                
                # Log tensor shapes for debugging dimension mismatches
                if config.verbose and total_processed < 100:  # Only log for first few batches
                    for key, tensor in data.items():
                        if isinstance(tensor, torch.Tensor):
                            print(f"Tensor '{key}' shape: {tensor.shape}, dtype: {tensor.dtype}")
                
                # Extract embeddings using the model's feature extractor
                try:
                    # Check if model has extract_embeddings method
                    if not hasattr(model, 'extract_embeddings'):
                        print(f"Error: Model does not have extract_embeddings method. Model type: {type(model).__name__}")
                        continue
                    
                    # Try extracting embeddings
                    try:
                        embeddings = model.extract_embeddings(data)
                    except AttributeError as e:
                        if 'object has no attribute' in str(e) and 'extract_embeddings' in str(e):
                            # Handle nested method, try with individual data items
                            print("Trying alternate embedding extraction approach...")
                            if 'graph_features' in data:
                                embeddings = model.extract_embeddings(data['graph_features'])
                            elif 'seq_features' in data:
                                embeddings = model.extract_embeddings(data['seq_features'])
                            elif 'tabular_features' in data:
                                embeddings = model.extract_embeddings(data['tabular_features'])
                            else:
                                # Fall back to first tensor we find
                                for key, tensor in data.items():
                                    if isinstance(tensor, torch.Tensor):
                                        print(f"Falling back to using {key} for embedding extraction")
                                        embeddings = model.extract_embeddings(tensor)
                                        break
                                else:
                                    raise ValueError("No suitable tensor found for embedding extraction")
                        else:
                            raise
                    
                    # Check for NaNs/Infs in embeddings
                    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                        nan_count = torch.isnan(embeddings).sum().item()
                        inf_count = torch.isinf(embeddings).sum().item()
                        
                        if config.verbose:
                            print(f"Warning: Found {nan_count} NaNs and {inf_count} Infs in embeddings")
                            
                        # If too many NaNs/Infs, skip the batch
                        total_values = embeddings.numel()
                        if (nan_count + inf_count) / total_values > 0.5:  # If more than 50% are invalid
                            print(f"Skipping batch with {nan_count + inf_count}/{total_values} invalid values")
                            skipped_nulls += len(embeddings)
                            continue
                            
                        # Fix NaNs/Infs
                        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Log the shape of the extracted embeddings
                    if config.verbose:
                        print(f"Extracted embeddings shape: {embeddings.shape}")
                    
                    # Ensure we have a numeric tensor with valid values
                    embeddings = torch.clamp(embeddings, min=-10.0, max=10.0)
                    
                    successful_extractions += len(embeddings)
                except Exception as e:
                    if config.verbose:
                        print(f"Error during embedding extraction: {str(e)}")
                        # Print stack trace for debugging
                        import traceback
                        traceback.print_exc()
                    continue
                
                # Store embeddings and labels
                embeddings_numpy = embeddings.cpu().numpy()
                labels_numpy = labels['category'].cpu().numpy()
                
                # Safety check to ensure valid numpy arrays
                if embeddings_numpy is None or not isinstance(embeddings_numpy, np.ndarray) or labels_numpy is None or not isinstance(labels_numpy, np.ndarray):
                    print(f"Warning: Invalid numpy arrays (embeddings: {type(embeddings_numpy)}, labels: {type(labels_numpy)})")
                    continue
                    
                # Make sure the arrays are 2D/1D respectively
                if embeddings_numpy.ndim == 1:
                    embeddings_numpy = embeddings_numpy.reshape(1, -1)
                if labels_numpy.ndim == 0:
                    labels_numpy = np.array([labels_numpy])
                
                # Verify that we have the same number of embeddings and labels
                if len(embeddings_numpy) != len(labels_numpy):
                    print(f"Warning: Mismatched number of embeddings ({len(embeddings_numpy)}) and labels ({len(labels_numpy)})")
                    # Take the minimum to ensure alignment
                    min_len = min(len(embeddings_numpy), len(labels_numpy))
                    embeddings_numpy = embeddings_numpy[:min_len]
                    labels_numpy = labels_numpy[:min_len]
                    
                embeddings_list.append(embeddings_numpy)
                labels_list.append(labels_numpy)
                processed_indices.extend(actual_indices[:len(embeddings_numpy)])
                
                # Store transaction metadata if available
                # Make sure we don't exceed the batch size when extracting metadata
                batch_size = len(embeddings_numpy)
                if 'txn_id' in batch_df.columns:
                    txn_ids.extend(batch_df['txn_id'].tolist()[:batch_size])
                if 'merchant_id' in batch_df.columns:
                    merchant_ids.extend(batch_df['merchant_id'].tolist()[:batch_size])
                if 'company_id' in batch_df.columns:
                    company_ids.extend(batch_df['company_id'].tolist()[:batch_size])
                if 'amount' in batch_df.columns:
                    amounts.extend(batch_df['amount'].tolist()[:batch_size])
            except Exception as e:
                if config.verbose:
                    print(f"Error during batch processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
    
    print(f"Processed {total_processed} rows, extracted embeddings for {successful_extractions} rows")
    print(f"Successfully processed {len(processed_indices)}/{len(dataset)} data points ({len(processed_indices)/len(dataset)*100:.1f}%)")
    print(f"Skipped {skipped_nulls} rows due to too many null values")
    
    if not embeddings_list:
        print("No embeddings were successfully extracted")
        return None
    
    try:
        # Concatenate embeddings and labels
        embeddings_array = np.vstack(embeddings_list)
        labels_array = np.concatenate(labels_list)
        
        print(f"Final extracted embeddings shape: {embeddings_array.shape}, labels shape: {labels_array.shape}")
        
        # Create DataFrame with embeddings
        embeddings_df = pd.DataFrame(embeddings_array)
        embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_array.shape[1])]
        
        # Add labels
        embeddings_df['category_id'] = labels_array
        
        # Add processed indices for reference
        embeddings_df['data_index'] = processed_indices
        
        # Add metadata if available - make sure lengths match
        metadata_length = len(embeddings_df)
        
        if txn_ids:
            txn_ids = txn_ids[:metadata_length]
            # If we have fewer txn_ids than rows, pad with NaNs
            if len(txn_ids) < metadata_length:
                txn_ids.extend([None] * (metadata_length - len(txn_ids)))
            embeddings_df['txn_id'] = txn_ids
            
        if merchant_ids:
            merchant_ids = merchant_ids[:metadata_length]
            if len(merchant_ids) < metadata_length:
                merchant_ids.extend([None] * (metadata_length - len(merchant_ids)))
            embeddings_df['merchant_id'] = merchant_ids
            
        if company_ids:
            company_ids = company_ids[:metadata_length]
            if len(company_ids) < metadata_length:
                company_ids.extend([None] * (metadata_length - len(company_ids)))
            embeddings_df['company_id'] = company_ids
            
        if amounts:
            amounts = amounts[:metadata_length]
            if len(amounts) < metadata_length:
                amounts.extend([None] * (metadata_length - len(amounts)))
            embeddings_df['amount'] = amounts
        
        # Print out a sample of the resulting DataFrame for verification
        if config.verbose and len(embeddings_df) > 0:
            print("\nSample of extracted embeddings DataFrame:")
            sample_cols = ['category_id', 'txn_id'] + [f'embedding_{i}' for i in range(min(3, embeddings_array.shape[1]))]
            sample_cols = [col for col in sample_cols if col in embeddings_df.columns]
            print(embeddings_df[sample_cols].head())
        
        return embeddings_df
    except Exception as e:
        print(f"Error creating embeddings DataFrame: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_evaluation_plots(predictions, labels, probas, config):
    """
    Generate evaluation plots and visualizations
    
    Args:
        predictions: Model predictions
        labels: True labels
        probas: Prediction probabilities
        config: Configuration
        
    Returns:
        plot_paths: Dictionary of created plot file paths
    """
    plot_paths = {}
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Plot confusion matrix
    try:
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        confusion_matrix_path = os.path.join(config.results_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['confusion_matrix'] = confusion_matrix_path
    except Exception as e:
        print(f"Error generating confusion matrix: {str(e)}")
    
    # Plot prediction confidence distribution
    try:
        # Extract max probability for each prediction
        max_probas = np.max(probas, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        # Separate correct and incorrect predictions
        correct_mask = predictions == labels
        
        plt.hist(max_probas[correct_mask], bins=20, alpha=0.7, label='Correct Predictions')
        plt.hist(max_probas[~correct_mask], bins=20, alpha=0.7, label='Incorrect Predictions')
        
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence (Max Probability)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        confidence_path = os.path.join(config.results_dir, 'confidence_distribution.png')
        plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['confidence_distribution'] = confidence_path
    except Exception as e:
        print(f"Error generating confidence distribution: {str(e)}")
    
    return plot_paths

def save_predictions(predictions, labels, probs, txn_ids, config):
    """
    Save predictions to a CSV file
    
    Args:
        predictions: Model predictions
        labels: True labels
        probs: Prediction probabilities
        txn_ids: Transaction IDs
        config: Configuration
        
    Returns:
        output_path: Path to the saved predictions file
    """
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    print(f"Saving predictions for dataset with {len(txn_ids)} transaction IDs")
    
    # Our primary goal is to create a DataFrame with entries for ALL transactions
    # Start by creating a DataFrame with all transaction IDs
    if txn_ids is not None:
        preds_df = pd.DataFrame({'txn_id': txn_ids})
    else:
        # If no txn_ids provided, create placeholder IDs
        preds_df = pd.DataFrame({'txn_id': [f"txn_{i}" for i in range(len(predictions) if predictions is not None else 0)]})
    
    # Add predictions if we have them (may be fewer than txn_ids)
    if predictions is not None and len(predictions) > 0:
        print(f"Found {len(predictions)} valid predictions")
        
        # Get valid indices for predictions (up to the length of predictions)
        valid_indices = list(range(min(len(predictions), len(preds_df))))
        
        # Add predictions column with NaN for padding
        preds_df['predicted_category'] = np.nan
        
        # Only set values for the valid indices
        if len(valid_indices) > 0:
            preds_df.loc[valid_indices, 'predicted_category'] = predictions[:len(valid_indices)]
    else:
        print("No valid predictions found")
        preds_df['predicted_category'] = np.nan
    
    # Add labels if we have them (may be fewer than txn_ids)
    if labels is not None and len(labels) > 0:
        print(f"Found {len(labels)} valid labels")
        
        # Add labels column with NaN for padding
        preds_df['true_category'] = np.nan
        
        # Get valid indices for labels (up to the length of labels)
        valid_indices = list(range(min(len(labels), len(preds_df))))
        
        # Only set values for the valid indices
        if len(valid_indices) > 0:
            preds_df.loc[valid_indices, 'true_category'] = labels[:len(valid_indices)]
    else:
        preds_df['true_category'] = np.nan
    
    # Add top k prediction confidences if we have probabilities
    top_k = 3
    if probs is not None and len(probs) > 0:
        try:
            # Check if probs is a list instead of numpy array and convert it
            if isinstance(probs, list):
                print(f"Converting probabilities from list to numpy array")
                # Check if elements are numpy arrays
                if all(isinstance(p, np.ndarray) for p in probs):
                    # Stack arrays
                    try:
                        probs = np.vstack([p for p in probs if isinstance(p, np.ndarray) and p.size > 0])
                    except ValueError:
                        # If arrays have different shapes, we can't easily stack them
                        print("Warning: Probability arrays have inconsistent shapes, skipping top-k extraction")
                        probs = None
                else:
                    # If not all elements are arrays, we need different handling
                    print("Warning: Probabilities list contains non-array elements, skipping top-k extraction")
                    probs = None
            
            # Continue only if we have a valid numpy array
            if probs is not None and isinstance(probs, np.ndarray):
                # Initialize probability columns
                for i in range(top_k):
                    preds_df[f'top_{i+1}_category'] = np.nan
                    preds_df[f'top_{i+1}_confidence'] = np.nan
                
                # Get valid indices (up to the length of probs)
                valid_indices = list(range(min(len(probs), len(preds_df))))
                
                # Only process valid rows that have probability data
                if valid_indices and len(probs.shape) > 1:
                    # Calculate top-k only for valid indices
                    valid_probs = probs[:len(valid_indices)]
                    top_indices = np.argsort(-valid_probs, axis=1)[:, :top_k]
                    top_probas = np.take_along_axis(valid_probs, top_indices, axis=1)
                    
                    # Add top-k predictions only for valid rows
                    for i in range(min(top_k, top_indices.shape[1])):
                        preds_df.loc[valid_indices, f'top_{i+1}_category'] = top_indices[:, i]
                        preds_df.loc[valid_indices, f'top_{i+1}_confidence'] = top_probas[:, i]
                    
                    print(f"Added top-{top_k} confidence scores for {len(valid_indices)} predictions")
                else:
                    print("No valid probability data to process")
        except Exception as e:
            print(f"Warning: Could not extract top-k probabilities: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Add correctness flag (but handle NaN values in labels)
    try:
        # Only compute for rows with valid true_category and predicted_category
        mask = (~pd.isna(preds_df['true_category'])) & (~pd.isna(preds_df['predicted_category']))
        preds_df['is_correct'] = np.nan
        preds_df.loc[mask, 'is_correct'] = preds_df.loc[mask, 'true_category'] == preds_df.loc[mask, 'predicted_category']
    except Exception as e:
        print(f"Warning: Could not compute correctness: {str(e)}")
        preds_df['is_correct'] = np.nan
    
    # Check for duplicate transaction IDs before saving
    if 'txn_id' in preds_df.columns:
        # Count unique transaction IDs
        unique_txn_ids = preds_df['txn_id'].nunique()
        
        if unique_txn_ids < len(preds_df):
            print(f"Detected {len(preds_df) - unique_txn_ids} duplicated transaction IDs")
            
            # Keep only the first occurrence of each transaction ID
            preds_df_deduped = preds_df.drop_duplicates(subset=['txn_id'])
            
            print(f"Removed duplicates: {len(preds_df)} rows → {len(preds_df_deduped)} unique rows")
            
            # Use the deduplicated DataFrame
            preds_df = preds_df_deduped
            
    # Save to CSV
    output_path = os.path.join(config.results_dir, config.prediction_output_file)
    preds_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(preds_df)} prediction rows to {output_path}")
    
    return output_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate Streamlined Graph Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=False,
                      help='Path to the trained model checkpoint')
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_dir', type=str, default='./data/parquet_files',
                      help='Directory containing parquet files for evaluation')
    data_group.add_argument('--results_dir', type=str, default='./evaluation_results',
                      help='Directory to save evaluation results')
    data_group.add_argument('--max_files', type=int, default=None,
                      help='Maximum number of parquet files to process')
    
    # Evaluation configuration
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    eval_group.add_argument('--no_plots', action='store_false', dest='generate_plots',
                      help='Disable generation of evaluation plots')
    eval_group.add_argument('--no_predictions', action='store_false', dest='save_predictions',
                      help='Disable saving predictions to file')
    eval_group.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    # Feature extraction
    feature_group = parser.add_argument_group('Feature Extraction')
    feature_group.add_argument('--extract_features', action='store_true',
                      help='Extract embeddings for further analysis')
    feature_group.add_argument('--features_output', type=str, default='extracted_features.pkl',
                      help='File to save extracted embeddings')
    
    # Hardware configuration
    hw_group = parser.add_argument_group('Hardware Configuration')
    hw_group.add_argument('--cpu_only', action='store_true',
                      help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Force CPU if requested
    if args.cpu_only and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    return args

def main():
    """Main evaluation script entry point"""
    print(f"\n{'='*80}")
    print(f"Streamlined Graph Model Evaluation")
    print(f"{'='*80}")
    
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration with base settings
    config = EvalConfig()
    
    # Override config with command line arguments
    if args.model_path:
        config.model_path = args.model_path
    if args.data_dir:
        config.eval_data_dir = args.data_dir
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.max_files:
        config.max_files = args.max_files
    if args.batch_size:
        config.batch_size = args.batch_size
    
    config.generate_plots = args.generate_plots
    config.save_predictions = args.save_predictions
    config.verbose = args.verbose
    config.extract_features = args.extract_features
    
    if args.features_output:
        config.features_output_file = args.features_output
    
    # Configure for available hardware
    device, config = configure_for_hardware(config)
    
    print(f"Evaluating model: {config.model_path}")
    print(f"Data directory: {config.eval_data_dir}")
    print(f"Results directory: {config.results_dir}")
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.results_dir, 'evaluation_config.txt')
    with open(config_path, 'w') as f:
        f.write("Streamlined Graph Model Evaluation Configuration\n")
        f.write("="*50 + "\n")
        for key, value in sorted(vars(config).items()):
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")
    
    # Load model
    try:
        model, model_config = load_model(config.model_path, device)
        print(f"Model loaded successfully")
        
        # Update config with model configuration
        for key, value in vars(model_config).items():
            if not key.startswith('__') and not hasattr(config, key):
                setattr(config, key, value)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get parquet files
    parquet_files = get_parquet_files(config.eval_data_dir, config.max_files)
    
    if not parquet_files:
        print(f"No parquet files found in {config.eval_data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files for evaluation")
    
    # Create dataset
    try:
        dataset = ParquetTransactionDataset(parquet_files, preprocess_fn=preprocess_transactions)
        print(f"Evaluation dataset: {len(dataset):,} transactions")
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return
    
    # Evaluate model
    start_time = time.time()
    try:
        # Extract and save embeddings - do this first and independently
        if config.extract_features:
            try:
                print("\nExtracting model embeddings for analysis...")
                embeddings_df = extract_embeddings(model, dataset, device, config)
                
                if embeddings_df is not None:
                    # Save embeddings
                    output_path = os.path.join(config.results_dir, config.features_output_file)
                    embeddings_df.to_pickle(output_path)
                    print(f"Extracted {len(embeddings_df)} embeddings saved to: {output_path}")
                else:
                    print("No embeddings could be extracted")
            except Exception as e:
                print(f"Error during embedding extraction: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\nStarting model evaluation...")
        metrics, predictions, labels, probs, txn_ids = evaluate_model(
            model, dataset, device, config, return_predictions=True
        )
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"Loss: {metrics.get('loss', 'N/A'):.4f}")
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"F1 Score (weighted): {metrics.get('f1_score', 'N/A'):.4f}")
        print(f"Precision (weighted): {metrics.get('precision', 'N/A'):.4f}")
        print(f"Recall (weighted): {metrics.get('recall', 'N/A'):.4f}")
        
        # Save evaluation report
        report_path = os.path.join(config.results_dir, config.report_output_file)
        with open(report_path, 'w') as f:
            # Add timestamp and metadata
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': config.model_path,
                'data_dir': config.eval_data_dir,
                'num_samples': len(dataset),
                'metrics': {k: v for k, v in metrics.items() if k != 'classification_report'},
                'class_metrics': metrics.get('classification_report', {})
            }
            json.dump(report_data, f, indent=2)
        
        print(f"\nEvaluation report saved to: {report_path}")
        
        # Generate and save plots
        if config.generate_plots and len(predictions) > 0:
            print("\nGenerating evaluation plots...")
            plot_paths = generate_evaluation_plots(predictions, labels, probs, config)
            print(f"Plots saved to: {', '.join(plot_paths.values())}")
        
        # Save predictions
        if config.save_predictions and len(predictions) > 0:
            print("\nSaving predictions...")
            output_path = save_predictions(predictions, labels, probs, txn_ids, config)
            print(f"Predictions saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nEvaluation completed in {execution_time:.2f} seconds")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()