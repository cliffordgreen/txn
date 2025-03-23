#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transaction prediction script that loads the trained model and makes predictions.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter
# Add UninitializedParameter to safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([UninitializedParameter])
import pickle
from datetime import datetime
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import our model components
from src.models.hybrid_transaction_model import EnhancedHybridTransactionModel
from src.data_processing.transaction_graph import build_transaction_relationship_graph

def load_model(model_path, device=None):
    """
    Load the trained model from the provided path.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on (defaults to best available)
        
    Returns:
        Loaded model instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 
                             'cpu')
    
    print(f"Loading model from {model_path} to {device}")
    
    # Load model state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    config = checkpoint['config']
    print(f"Model configuration: {config}")
    
    # Extract model parameters from config or use defaults
    input_dim = 128  # Default input dimension
    hidden_dim = config.get('hidden_dim', 128)
    output_dim = 400  # Default number of categories
    num_heads = config.get('num_heads', 4)
    num_graph_layers = config.get('num_graph_layers', 2)
    num_temporal_layers = config.get('num_temporal_layers', 2)
    dropout = config.get('dropout', 0.2)
    use_hyperbolic = config.get('use_hyperbolic', True)
    use_neural_ode = config.get('use_neural_ode', False)
    use_text = config.get('use_text', False)
    multi_task = True  # Default to multi-task
    tax_type_dim = 20  # Default tax type dimension
    
    # Initialize model with extracted configuration
    model = EnhancedHybridTransactionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_graph_layers=num_graph_layers,
        num_temporal_layers=num_temporal_layers,
        dropout=dropout,
        use_hyperbolic=use_hyperbolic,
        use_neural_ode=use_neural_ode,
        use_text=use_text,
        multi_task=multi_task,
        tax_type_dim=tax_type_dim,
        company_input_dim=None,  # This will be dynamically adjusted in forward pass
        num_relations=5,
        graph_weight=0.6,
        temporal_weight=0.4,
        use_dynamic_weighting=True
    )
    
    # Load model weights with strict=False to ignore missing/extra keys
    print("Loading model state dict...")
    
    # Check if the state dict is directly in the checkpoint or nested
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # If not found, assume the checkpoint itself is the state dict
        state_dict = checkpoint
    
    # Load with strict=False to handle missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
    
    model.to(device)
    model.eval()
    
    # Load category and tax type mappings if available
    category_map = checkpoint.get('category_map', {})
    tax_type_map = checkpoint.get('tax_type_map', {})
    
    return model, category_map, tax_type_map, config, device

def prepare_data(df, model, device):
    """
    Prepare data for prediction using the model's prepare_data_from_dataframe method.
    
    Args:
        df: DataFrame with transaction data
        model: Model instance to use for data preparation
        device: Device to load the data on
        
    Returns:
        Prepared data dictionary with tensors moved to the target device
    """
    print(f"Preparing data from DataFrame with {len(df)} rows")
    data = model.prepare_data_from_dataframe(df)
    
    # Move all tensors to the device
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
            
    return data

def make_predictions(model, data, category_map=None, tax_type_map=None):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Loaded model instance
        data: Prepared data dictionary
        category_map: Mapping from indices to category names
        tax_type_map: Mapping from indices to tax type names
        
    Returns:
        DataFrame with original data and predictions
    """
    model.eval()
    
    print("Making predictions...")
    with torch.no_grad():
        # Forward pass
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
        
        # Process outputs based on whether the model is multi-task
        if isinstance(outputs, tuple):
            category_logits, tax_type_logits = outputs
            category_probs = torch.softmax(category_logits, dim=1)
            tax_type_probs = torch.softmax(tax_type_logits, dim=1)
            
            # Get predicted categories and tax types
            predicted_category_indices = torch.argmax(category_probs, dim=1).cpu().numpy()
            predicted_tax_type_indices = torch.argmax(tax_type_probs, dim=1).cpu().numpy()
            
            # Get confidence scores
            category_confidences = torch.max(category_probs, dim=1)[0].cpu().numpy()
            tax_type_confidences = torch.max(tax_type_probs, dim=1)[0].cpu().numpy()
            
            # Map indices to names if mappings are provided
            if category_map is not None:
                predicted_categories = [category_map.get(idx, f"Category_{idx}") for idx in predicted_category_indices]
            else:
                predicted_categories = [f"Category_{idx}" for idx in predicted_category_indices]
                
            if tax_type_map is not None:
                predicted_tax_types = [tax_type_map.get(idx, f"TaxType_{idx}") for idx in predicted_tax_type_indices]
            else:
                predicted_tax_types = [f"TaxType_{idx}" for idx in predicted_tax_type_indices]
                
            # Create results dictionary
            results = {
                'predicted_category_idx': predicted_category_indices,
                'predicted_category': predicted_categories,
                'category_confidence': category_confidences,
                'predicted_tax_type_idx': predicted_tax_type_indices,
                'predicted_tax_type': predicted_tax_types,
                'tax_type_confidence': tax_type_confidences
            }
        else:
            # Single-task model (only category predictions)
            category_probs = torch.softmax(outputs, dim=1)
            predicted_category_indices = torch.argmax(category_probs, dim=1).cpu().numpy()
            category_confidences = torch.max(category_probs, dim=1)[0].cpu().numpy()
            
            if category_map is not None:
                predicted_categories = [category_map.get(idx, f"Category_{idx}") for idx in predicted_category_indices]
            else:
                predicted_categories = [f"Category_{idx}" for idx in predicted_category_indices]
                
            results = {
                'predicted_category_idx': predicted_category_indices,
                'predicted_category': predicted_categories,
                'category_confidence': category_confidences
            }
    
    return results

def extract_embeddings(model, data):
    """
    Extract embeddings from the model for the input data.
    
    Args:
        model: Loaded model instance
        data: Prepared data dictionary
        
    Returns:
        DataFrame with transaction IDs and embeddings
    """
    model.eval()
    
    print("Extracting embeddings...")
    with torch.no_grad():
        embeddings = model.extract_embeddings(data)
        
    if embeddings is not None:
        # Convert to numpy for storage
        embeddings_np = embeddings.cpu().numpy()
        return embeddings_np
    else:
        print("Warning: No embeddings could be extracted.")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained transaction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, 
                        default='./models/enhanced_model_output/best_model.pt',
                        help='Path to the trained model file')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input parquet file with transaction data')
    
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the output predictions (CSV format)')
    
    parser.add_argument('--embeddings_file', type=str, default=None,
                        help='Path to save embeddings (pickle format)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for predictions')
    
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to return')
    
    args = parser.parse_args()
    
    # Set device
    if args.cpu_only:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 
                             'cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model, category_map, tax_type_map, config, device = load_model(args.model_path, device)
    
    # Load input data
    print(f"Loading data from {args.input_file}")
    try:
        df = pd.read_parquet(args.input_file)
        print(f"Loaded {len(df)} transactions")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Prepare data for prediction
    data = prepare_data(df, model, device)
    
    # Make predictions
    results = make_predictions(model, data, category_map, tax_type_map)
    
    # Merge predictions with input data
    output_df = df.copy()
    
    # Add prediction columns
    for key, value in results.items():
        if len(value) == len(output_df):
            output_df[key] = value
        else:
            # Handle case where the number of predictions doesn't match input rows
            # (e.g., due to batching or data filtering)
            print(f"Warning: Number of predictions ({len(value)}) doesn't match input rows ({len(output_df)})")
            padded_value = np.zeros(len(output_df)) if isinstance(value[0], (int, float, np.number)) else [""] * len(output_df)
            padded_value[:len(value)] = value
            output_df[key] = padded_value
    
    # Extract embeddings if requested
    if args.embeddings_file:
        embeddings = extract_embeddings(model, data)
        if embeddings is not None:
            # Save embeddings to file
            embeddings_output = args.embeddings_file
            print(f"Saving embeddings to {embeddings_output}")
            
            # Create dictionary with transaction IDs and embeddings
            if 'txn_id' in df.columns:
                embeddings_dict = {
                    'txn_id': df['txn_id'].values[:len(embeddings)],
                    'embeddings': embeddings
                }
            else:
                # Use index as transaction ID if txn_id column is not available
                embeddings_dict = {
                    'index': np.arange(len(embeddings)),
                    'embeddings': embeddings
                }
                
            with open(embeddings_output, 'wb') as f:
                pickle.dump(embeddings_dict, f)
    
    # Save predictions if output file is specified
    if args.output_file:
        output_path = args.output_file
        print(f"Saving predictions to {output_path}")
        
        # Determine output format based on file extension
        file_ext = os.path.splitext(output_path)[1].lower()
        
        if file_ext == '.csv':
            output_df.to_csv(output_path, index=False)
        elif file_ext == '.parquet':
            output_df.to_parquet(output_path, index=False)
        else:
            print(f"Unsupported output format: {file_ext}. Saving as CSV.")
            output_df.to_csv(output_path, index=False)
    else:
        # If no output file is specified, print sample predictions
        print("\nSample predictions:")
        sample_cols = ['txn_id', 'amount'] if 'txn_id' in output_df.columns else ['amount']
        pred_cols = [col for col in output_df.columns if col.startswith('predicted_') or col.endswith('confidence')]
        
        print(output_df[sample_cols + pred_cols].head(10))
    
    print("Prediction completed successfully.")

if __name__ == "__main__":
    main()