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

# Add project root to path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

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
    # Load checkpoint with weights_only=False to handle PyTorch 2.6+ changes
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # For older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config()
        for key, value in config_dict.items():
            if not key.startswith('__'):
                setattr(config, key, value)
    else:
        # Create default config if not in checkpoint
        config = Config()
        print("Warning: No configuration found in checkpoint. Using defaults.")
    
    # Initialize model architecture
    num_categories = 400  # Default, to be updated based on actual data
    num_tax_types = 20    # Default, to be updated based on actual data
    
    model = EnhancedHybridTransactionModel(
        input_dim=config.hidden_dim,
        hidden_dim=config.hidden_dim,
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
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
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=df_collate_fn
    )
    
    # Initialize metrics
    total_loss = 0
    category_correct = 0
    category_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_txn_ids = []
    
    # Evaluate without gradients
    with torch.no_grad():
        for batch_indices in tqdm(dataloader, desc="Evaluation", disable=not config.verbose):
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
                
                # Store transaction IDs if available
                if 'txn_id' in batch_df.columns:
                    txn_ids = batch_df['txn_id'].tolist()
                    all_txn_ids.extend(txn_ids)
            except Exception as e:
                if config.verbose:
                    print(f"Error processing batch: {str(e)}")
                continue
            
            # Prepare inputs
            try:
                data, labels = prepare_model_inputs(batch_df, model, device)
                
                # Replace NaN/Inf in input tensors for stability
                for key, tensor in data.items():
                    if isinstance(tensor, torch.Tensor) and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                        data[key] = torch.nan_to_num(tensor, nan=0.0)
            except Exception as e:
                if config.verbose:
                    print(f"Error preparing inputs: {str(e)}")
                continue
            
            # Forward pass
            try:
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
                    
                    # Get predictions and probabilities
                    category_probs = torch.softmax(category_logits, dim=1)
                    category_preds = category_logits.argmax(dim=1)
                    correct = (category_preds == labels['category']).sum().item()
                    
                    # Store probabilities
                    all_probs.extend(category_probs.cpu().numpy())
                    
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
                    
                    # Get predictions and probabilities
                    category_probs = torch.softmax(outputs, dim=1)
                    category_preds = outputs.argmax(dim=1)
                    correct = (category_preds == labels['category']).sum().item()
                    
                    # Store probabilities
                    all_probs.extend(category_probs.cpu().numpy())
                    
                    # Track metrics
                    category_correct += correct
                    category_total += batch_size
                    
                    # Store predictions for metrics
                    all_preds.extend(category_preds.cpu().numpy())
                    all_labels.extend(labels['category'].cpu().numpy())
                
                # Update total loss
                total_loss += loss.item()
            except Exception as e:
                if config.verbose:
                    print(f"Error during forward pass: {str(e)}")
                continue
    
    # Calculate final metrics
    metrics = {}
    
    if category_total > 0:
        avg_loss = total_loss / category_total
        category_accuracy = category_correct / category_total
        
        metrics['loss'] = avg_loss
        metrics['accuracy'] = category_accuracy
        
        # Calculate additional metrics if we have predictions
        if len(all_preds) > 1 and len(all_labels) > 1:
            metrics['f1_score'] = f1_score(all_labels, all_preds, average='weighted')
            metrics['precision'] = precision_score(all_labels, all_preds, average='weighted')
            metrics['recall'] = recall_score(all_labels, all_preds, average='weighted')
            
            # Compute class-wise metrics
            report = classification_report(all_labels, all_preds, output_dict=True)
            metrics['classification_report'] = report
    else:
        metrics['error'] = "No valid samples for evaluation"
    
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
    
    with torch.no_grad():
        for batch_indices in tqdm(dataloader, desc="Extracting embeddings", disable=not config.verbose):
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
                
                # Extract embeddings using the model's feature extractor
                embeddings = model.extract_embeddings(data)
                
                # Store embeddings and labels
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels['category'].cpu().numpy())
                
                # Store transaction metadata if available
                if 'txn_id' in batch_df.columns:
                    txn_ids.extend(batch_df['txn_id'].tolist())
                if 'merchant_id' in batch_df.columns:
                    merchant_ids.extend(batch_df['merchant_id'].tolist())
                if 'company_id' in batch_df.columns:
                    company_ids.extend(batch_df['company_id'].tolist())
                if 'amount' in batch_df.columns:
                    amounts.extend(batch_df['amount'].tolist())
            except Exception as e:
                if config.verbose:
                    print(f"Error during embedding extraction: {str(e)}")
                continue
    
    if not embeddings_list:
        return None
    
    # Concatenate embeddings and labels
    embeddings_array = np.vstack(embeddings_list)
    labels_array = np.concatenate(labels_list)
    
    # Create DataFrame with embeddings
    embeddings_df = pd.DataFrame(embeddings_array)
    embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_array.shape[1])]
    
    # Add labels
    embeddings_df['category_id'] = labels_array
    
    # Add metadata if available
    if txn_ids:
        embeddings_df['txn_id'] = txn_ids[:len(embeddings_df)]
    if merchant_ids:
        embeddings_df['merchant_id'] = merchant_ids[:len(embeddings_df)]
    if company_ids:
        embeddings_df['company_id'] = company_ids[:len(embeddings_df)]
    if amounts:
        embeddings_df['amount'] = amounts[:len(embeddings_df)]
    
    return embeddings_df

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
    
    # Create DataFrame with predictions
    preds_df = pd.DataFrame()
    
    # Add txn_ids if available
    if txn_ids:
        preds_df['txn_id'] = txn_ids
    
    # Add true and predicted labels
    preds_df['true_category'] = labels
    preds_df['predicted_category'] = predictions
    
    # Add top 3 prediction confidences
    top_k = 3
    if probs is not None and len(probs) > 0:
        top_indices = np.argsort(-probs, axis=1)[:, :top_k]
        top_probas = np.take_along_axis(probs, top_indices, axis=1)
        
        for i in range(min(top_k, probs.shape[1])):
            preds_df[f'top_{i+1}_category'] = top_indices[:, i]
            preds_df[f'top_{i+1}_confidence'] = top_probas[:, i]
    
    # Add correctness flag
    preds_df['is_correct'] = preds_df['true_category'] == preds_df['predicted_category']
    
    # Save to CSV
    output_path = os.path.join(config.results_dir, config.prediction_output_file)
    preds_df.to_csv(output_path, index=False)
    
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
        
        # Extract and save embeddings
        if config.extract_features:
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