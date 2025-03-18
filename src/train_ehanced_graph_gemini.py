#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Hybrid Transaction Model Training Script

This script implements the training process for the EnhancedHybridTransactionModel.
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
import pyarrow.parquet as pq  # Import pyarrow
from typing import List, Dict, Tuple, Any, Optional  # Import typing hints
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# Add project root to path (assuming this script is in a subdirectory)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import custom modules (you'll need to provide these)
try:
    from src.models.hybrid_transaction_model import EnhancedHybridTransactionModel
    # Removed unused imports
except ImportError:
    print("Error: Could not import custom modules. Make sure src/models/hybrid_transaction_model.py exists.")
    sys.exit(1)

from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class Config:
    # Data configuration
    data_dir: str = "/path/to/parquet/files"  # Will be overridden
    output_dir: str = "../models/enhanced_model_output"
    batch_size: int = 64
    num_workers: int = 4  # Use multiprocessing
    prefetch_factor: int = 2
    max_files: int = 100

    # Model configuration
    hidden_dim: int = 256
    num_heads: int = 8
    num_graph_layers: int = 2
    num_temporal_layers: int = 2
    dropout: float = 0.2
    use_hyperbolic: bool = True
    use_neural_ode: bool = False
    use_text: bool = False
    multi_task: bool = True
    num_relations: int = 5

    # Training configuration
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 10
    patience: int = 3
    grad_clip: float = 1.0

    # GPU optimization
    use_amp: bool = True
    use_cuda_graphs: bool = True
    cuda_graph_batch_size: Optional[int] = field(default=None, metadata={'description': 'Will be set to batch_size'})

    # XGBoost integration
    extract_embeddings: bool = True
    embedding_output_file: str = "transaction_embeddings.pkl"

    # Metrics
    eval_steps: int = 100
    log_steps: int = 10

    # Added: Seed for reproducibility
    seed: int = 42



def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)  #  If you import and use 'random'

def df_collate_fn(batch: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Correct collate function to handle DataFrames.

    Args:
        batch: A list of pandas DataFrames.

    Returns:
        A list containing a single concatenated DataFrame.
    """
    return [pd.concat(batch, ignore_index=True)]


class ParquetTransactionDataset(Dataset):
    def __init__(self, parquet_files: List[str], preprocess_fn=None):
        self.parquet_files = parquet_files
        self.preprocess_fn = preprocess_fn

        # Use pyarrow to get metadata efficiently
        self.file_row_counts = []
        self.total_rows = 0
        self.metadata = []  # Store metadata for each file

        for file in tqdm(parquet_files, desc="Loading Metadata"):
            try:
                metadata = pq.read_metadata(file)
                self.file_row_counts.append(metadata.num_rows)
                self.total_rows += metadata.num_rows
                self.metadata.append(metadata)  # Store metadata
            except Exception as e:
                print(f"Warning: Could not read metadata for {file}, skipping. Error: {e}")
                #  Don't add to counts, so this file is skipped

        self.lookup = []
        for file_idx, row_count in enumerate(self.file_row_counts):
            self.lookup.extend([(file_idx, row_idx) for row_idx in range(row_count)])

        print(f"Dataset initialized with {len(parquet_files)} files and {self.total_rows} total rows")


    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> pd.DataFrame:
        if idx >= len(self.lookup):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.lookup)} items")

        file_idx, row_idx = self.lookup[idx]
        file_path = self.parquet_files[file_idx]

        try:
            # Efficiently read a single row using pyarrow.  This is MUCH faster.
            table = pq.read_table(file_path, use_threads=True) # Read as pyarrow table for efficiency
            row = table.slice(row_idx, 1).to_pandas() # Efficiently get the specific row

            if row.empty:
                raise ValueError(f"Row {row_idx} not found in {file_path}")
            
            if self.preprocess_fn:
                row = self.preprocess_fn(row)
            return row

        except Exception as e:
            raise RuntimeError(f"Error reading row {row_idx} from {file_path}: {e}") from e


    def get_batch_df(self, indices: List[int]) -> pd.DataFrame:
        """Get a batch of rows as a single DataFrame (efficiently)."""
        dfs = []
        for idx in indices:
            try:
                df = self[idx]  # Use the efficient __getitem__
                if not df.empty: # Check if the returned df is empty
                    dfs.append(df)
            except Exception as e:
                print(f"Error getting item {idx}: {e}")
                #  Decide: skip the row (as currently), or raise the exception.
                #  Raising is often better for debugging.
                continue  # Or: raise
        if not dfs:
            raise ValueError("All dataframes in batch were empty")  # Raise if all are empty
        return pd.concat(dfs, ignore_index=True)

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the transaction DataFrame.  This is now done *once*
    when the dataset is created.
    """
    # Handle missing values (impute or drop - strategy depends on your data)
    df = df.fillna(0)  # Simple imputation.  Replace with a better strategy if needed.

    # Type casting for memory efficiency (do this once, not per batch!)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        # Add other type conversions as needed (e.g., object -> category)
    return df

def configure_for_hardware():
    """Configures PyTorch based on hardware availability (GPU, MPS, CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():  # Check for MPS (Apple Silicon)
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device

def create_dataloaders(config: Config, device):
    """Creates training, validation, and test data loaders."""

    # 1. Find all Parquet files
    all_files = glob.glob(os.path.join(config.data_dir, "*.parquet"))
    all_files = sorted(all_files)[:config.max_files]  # Limit number of files
    print(f"Found {len(all_files)} parquet files.")
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {config.data_dir}")

    # 2. Split files into train/val/test sets
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=config.seed)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=config.seed)  # 0.25 * 0.8 = 0.2

    # 3. Create datasets with preprocessing
    train_dataset = ParquetTransactionDataset(train_files, preprocess_fn=preprocess_transactions)
    val_dataset = ParquetTransactionDataset(val_files, preprocess_fn=preprocess_transactions)
    test_dataset = ParquetTransactionDataset(test_files, preprocess_fn=preprocess_transactions)


    # 4. Create DataLoaders (with corrected collate_fn and num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,  # Use multiple workers
        collate_fn=df_collate_fn,  # Use the corrected collate function
        pin_memory=True if device.type == 'cuda' else False,  # Pin memory for CUDA
        prefetch_factor=config.prefetch_factor,  # Prefetch batches
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=df_collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=config.prefetch_factor,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=df_collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=config.prefetch_factor,
        drop_last=False
    )
    return train_loader, val_loader, test_loader

def prepare_model_inputs(batch: List[pd.DataFrame], model: nn.Module, device) -> Dict[str, torch.Tensor]:
    """
    Prepares model inputs from a batch of DataFrames.  Crucially, this now
    calls model.prepare_data_from_dataframe.
    """
    if not batch:
        raise ValueError("Batch is empty")

    df = batch[0]  # Access the single concatenated DataFrame
    if df.empty:
        raise ValueError("The DataFrame in the batch is empty")

    # The model is responsible for data preparation.
    prepared_data = model.prepare_data_from_dataframe(df)

    # Add labels to the prepared_data dict.  This is now done *here*,
    # and the labels are created based on the *model's* processed data,
    # ensuring proper alignment.
    prepared_data['labels'] = torch.tensor(df['category_id'].values, dtype=torch.long).to(device)
    if 'merchant_id' in df.columns:
       prepared_data['merchant_labels'] = torch.tensor(pd.factorize(df['merchant_id'])[0], dtype=torch.long).to(device)
    if 'amount' in df.columns:
       prepared_data['amount_labels'] = torch.tensor(df['amount'], dtype=torch.float32).to(device)

    # Move tensors to the correct device, only if necessary
    for key, value in prepared_data.items():
        if isinstance(value, torch.Tensor) and value.device != device:
             prepared_data[key] = value.to(device)
    return prepared_data


def create_cuda_graph(model: nn.Module, sample_data: Dict, device):
    """Creates a CUDA graph for a static part of the computation."""
    # Put sample data on the correct device.
    for k, v in sample_data.items():
        if isinstance(v, torch.Tensor):
            sample_data[k] = v.to(device, non_blocking=True)

    # Warmup, capture, and then clean up.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # Warmup iterations
            model.zero_grad(set_to_none=True)  # Use set_to_none for efficiency
            with autocast(enabled=True):
                outputs = model(**sample_data)
                if isinstance(outputs, tuple):
                    category_logits, tax_type_logits = outputs
                    loss = F.cross_entropy(category_logits, sample_data['labels'])
                    if 'tax_type' in sample_data:
                        loss += F.cross_entropy(tax_type_logits, sample_data['tax_type_labels']) # Assuming you have this
                else:
                    loss = F.cross_entropy(outputs, sample_data['labels'])
            loss.backward()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        static_outputs = model(**sample_data)
        if isinstance(static_outputs, tuple):
            static_category_logits, static_tax_type_logits = static_outputs
            static_loss = F.cross_entropy(static_category_logits, sample_data['labels'])
            if 'tax_type' in sample_data:
                static_loss += F.cross_entropy(static_tax_type_logits, sample_data['tax_type_labels'])
        else:
            static_loss = F.cross_entropy(static_outputs, sample_data['labels'])

        static_loss.backward()

    return graph, static_outputs, sample_data


def run_with_cuda_graph(model, graph, static_inputs, static_outputs, inputs, optimizer, device):
    """Runs a forward and backward pass using a pre-captured CUDA graph."""

    # Copy *new* input data into the static input tensors.
    # Ideally, avoid copies entirely by allocating static_inputs outside
    # the graph and having the model work directly with them.
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and k in static_inputs:
            static_inputs[k].copy_(v) # Copy the *new* values into the static inputs

    graph.replay()
    optimizer.step()

    return static_outputs # Return the static_outputs


def train(config: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device):
    """Trains the model."""

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.patience)

    scaler = GradScaler(enabled=config.use_amp)  # For mixed precision

    # Create CUDA graph (if enabled and model/data are suitable)
    use_cuda_graphs = config.use_cuda_graphs and device.type == 'cuda'
    static_graph = None
    static_outputs = None
    static_inputs = None

    if use_cuda_graphs:
        # Create a sample batch for CUDA graph capture *before* the training loop.
        try:
            sample_batch = next(iter(train_loader))
            sample_inputs = prepare_model_inputs(sample_batch, model, device)
            static_graph, static_outputs, static_inputs = create_cuda_graph(model, sample_inputs, device)
        except Exception as e:
            print(f"CUDA graph creation failed: {e}.  Disabling CUDA graphs.")
            use_cuda_graphs = False

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    model_path = os.path.join(config.output_dir, "best_model.pth")

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                inputs = prepare_model_inputs(batch, model, device)

                if use_cuda_graphs:
                    # Use the CUDA graph.
                    outputs = run_with_cuda_graph(model, static_graph, static_inputs, static_outputs, inputs, optimizer, device)
                    # Calculate loss *outside* the graph.
                    category_loss = F.cross_entropy(outputs['logits'], inputs['labels'])
                    merchant_loss = F.cross_entropy(outputs['merchant_logits'], inputs['merchant_labels']) # Assuming you have this
                    amount_loss = F.mse_loss(outputs['amount_pred'].squeeze(), inputs['amount_labels'])  # And this
                    loss = category_loss + merchant_loss + amount_loss

                else:
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=config.use_amp):
                        outputs = model(**inputs)
                        category_loss = F.cross_entropy(outputs['logits'], inputs['labels'])
                        merchant_loss = F.cross_entropy(outputs['merchant_logits'], inputs['merchant_labels'])
                        amount_loss = F.mse_loss(outputs['amount_pred'].squeeze(), inputs['amount_labels'])
                        loss = category_loss + merchant_loss + amount_loss

                    scaler.scale(loss).backward()
                    if config.grad_clip > 0.0:
                        scaler.unscale_(optimizer)  # Unscale before clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * inputs['labels'].size(0)  # Accumulate correctly
                total_samples += inputs['labels'].size(0)

                if batch_idx % config.log_steps == 0:
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            except ValueError as ve:
                print(f"Skipping batch {batch_idx} due to ValueError: {ve}")
                continue  # Skip to the next batch
            except Exception as e:
                print(f"Error in training loop (batch {batch_idx}): {e}")
                continue  # or: raise, depending on desired behavior

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / total_samples

        # Evaluate and get validation loss
        val_loss, val_metrics = evaluate(model, val_loader, device, multi_task=config.multi_task)

        # Step the scheduler based on validation loss *after* each epoch.
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")


        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'config': config,  # Save the configuration
                'val_loss': best_val_loss,
            }, model_path)
            print(f"Saved best model to {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")


def evaluate(model: nn.Module, data_loader: DataLoader, device, multi_task:bool):
    """Evaluates the model."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_labels = []
    all_preds = []
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch in data_loader:
            try:
                inputs = prepare_model_inputs(batch, model, device)

                outputs = model(**inputs)

                category_loss = F.cross_entropy(outputs['logits'], inputs['labels'])
                # You need to handle this logic if multi_task is True
                if multi_task:
                  merchant_loss = F.cross_entropy(outputs['merchant_logits'], inputs['merchant_labels'])
                  amount_loss = F.mse_loss(outputs['amount_pred'].squeeze(), inputs['amount_labels'])
                  loss = category_loss + merchant_loss + amount_loss
                else:
                  loss = category_loss


                total_loss += loss.item() * inputs['labels'].size(0)  # Correct accumulation
                total_samples += inputs['labels'].size(0)

                # Store predictions and labels, handling multi-task
                if multi_task:
                    _, preds = torch.max(outputs['logits'], 1)  # Use the main logits for evaluation
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(inputs['labels'].cpu().numpy()) # Make sure this is main label

            except ValueError as ve:
                print(f"Skipping batch in evaluation due to ValueError: {ve}")
                continue  # Skip to the next batch
            except Exception as e:
                print(f"Error in evaluation loop: {e}")
                continue

    avg_loss = total_loss / total_samples
    metrics = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics

def calculate_metrics(y_true, y_pred):
    """Calculates evaluation metrics."""
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    # You could also compute a classification report:
    # report = classification_report(y_true, y_pred, zero_division=0)

    return {'f1': f1, 'precision': precision, 'recall': recall}


def main(args):
    config = Config(data_dir=args.data_dir)

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Configure the device (CPU, GPU, MPS)
    device = configure_for_hardware()

     # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(config, device)

    # Get dummy batch to initialize model input dimension
    try:
        dummy_batch = next(iter(train_loader))
        dummy_inputs = prepare_model_inputs(dummy_batch, EnhancedHybridTransactionModel(input_dim=1), device)  # Dummy input dim
        input_dim = dummy_inputs['x'].shape[1]  # Now we get the correct input_dim
    except Exception as e:
        print(f"Error getting dummy batch: {e}")
        return
    
    # Instantiate the model with the correct input dimension.
    model = EnhancedHybridTransactionModel(input_dim=input_dim, multi_task=config.multi_task)
    model.to(device)

    print(model)
    # Train the model
    train(config, model, train_loader, val_loader, device)

    # Evaluate the model
    best_model_path = os.path.join(config.output_dir, "best_model.pth")
     # Check if the best model file exists
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_metrics = evaluate(model, test_loader, device, multi_task=config.multi_task)
        print(f"Test Loss: {test_loss:.4f}, Test F1: {test_metrics['f1']:.4f}")

        # Example of extracting embeddings (if enabled)
        if config.extract_embeddings:
          #  You'd need a way to get *all* data through the model
          #  to extract embeddings for all transactions.  This is a
          #  simplified example, assuming you have a function to do this:
          try:
            # Assuming you have all data prepared similarly to train_loader
            all_embeddings = model.extract_embeddings(
                x=dummy_inputs['x'],
                edge_index=dummy_inputs['edge_index'],
                edge_type=dummy_inputs['edge_type'],
                edge_attr=dummy_inputs['edge_attr'],
                seq_features=dummy_inputs['seq_features'],
                timestamps=dummy_inputs['timestamps'],
                tabular_features=dummy_inputs['tabular_features'],
                t0=dummy_inputs['t0'],
                t1=dummy_inputs['t1'],
            )
            torch.save(all_embeddings, config.embedding_output_file)
            print(f"Embeddings saved to {config.embedding_output_file}")
          except Exception as e:
             print("Could not extract embeddings", e)
    else:
        print(f"Error: Best model file not found at {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Enhanced Hybrid Transaction Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing Parquet files.")
    # You could add other configurable parameters as command-line arguments here
    args = parser.parse_args()
    main(args)