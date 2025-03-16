import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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
import random


def load_transaction_feedback_data(file_path: str) -> pd.DataFrame:
    """
    Load transaction data with user feedback from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing transaction data with business entity features
    """
    print(f"Loading transaction data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Print data statistics
    print(f"Loaded {len(df)} transactions")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if required columns exist
    required_cols = ['user_id', 'txn_id', 'is_new_user']
    target_cols = [
        'presented_category_id', 'presented_category_name', 
        'presented_tax_account_type', 'presented_tax_account_type_name', 
        'accepted_category_id', 'accepted_category_name',
        'accepted_tax_account_type', 'accepted_tax_account_type_name'
    ]
    
    # Business metadata columns (optional)
    business_cols = [
        'company_type', 'company_size'
    ]
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        # Add dummy columns
        for col in missing_cols:
            if col == 'user_id':
                df[col] = range(len(df))
            elif col == 'txn_id':
                df[col] = [f"txn_{i}" for i in range(len(df))]
            elif col == 'is_new_user':
                df[col] = 0
    
    # Check target columns
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        print(f"Warning: Missing target columns: {missing_targets}")
    
    # Check for business metadata columns
    missing_business = [col for col in business_cols if col not in df.columns]
    if missing_business:
        print(f"Info: No company data columns found: {missing_business}. Adding synthetic data for demonstration.")
        
        # Add synthetic company data for demonstration purposes that matches our schema
        
        # QBO product tiers
        qbo_products = ['Simple Start', 'Essentials', 'Plus', 'Advanced']
        
        # Industry types - based on small business categories
        industry_types = ['Retail', 'Professional Services', 'Manufacturing', 'Construction', 
                         'Healthcare', 'Food Service', 'Technology', 'Real Estate', 'Other Services']
        
        industry_codes = list(range(1, len(industry_types) + 1))
        
        # QBO signup types
        signup_types = ['Direct', 'Partner', 'Trial', 'Conversion']
        
        # Region and language
        regions = ['US_WEST', 'US_EAST', 'US_CENTRAL', 'CANADA', 'UK', 'APAC']
        languages = ['en_US', 'en_CA', 'en_UK', 'es_US', 'fr_CA']
        
        # Generate industry info
        df['industry_name'] = np.random.choice(industry_types, size=len(df))
        df['industry_code'] = df['industry_name'].apply(lambda x: industry_types.index(x) + 1)
        df['industry_standard'] = 'SIC'
        
        # Generate QBO product information
        df['qbo_current_product'] = np.random.choice(
            qbo_products, 
            size=len(df), 
            p=[0.35, 0.30, 0.25, 0.10]  # Weighted toward simpler products
        )
        
        df['qbo_signup_type_desc'] = np.random.choice(signup_types, size=len(df))
        
        # Generate binary flags
        df['qbo_accountant_attached_current_flag'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        df['qbo_accountant_attached_ever'] = np.where(
            df['qbo_accountant_attached_current_flag'] == 1, 1, np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
        )
        df['qblive_attach_flag'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        
        # Generate region and language data
        df['region_id'] = np.random.randint(1, len(regions) + 1, size=len(df))
        df['region_name'] = df['region_id'].apply(lambda x: regions[x-1])
        df['language_id'] = np.random.randint(1, len(languages) + 1, size=len(df))
        df['language_name'] = df['language_id'].apply(lambda x: languages[x-1])
        
        # Generate company ID and name
        if 'company_id' not in df.columns:
            df['company_id'] = np.random.randint(10000, 99999, size=len(df))
            df['company_name'] = df['company_id'].apply(lambda x: f"Company {x}")
        
        # Generate some date information
        today = pd.Timestamp.now()
        df['qbo_signup_date'] = [today - pd.Timedelta(days=np.random.randint(30, 1500)) for _ in range(len(df))]
        df['qbo_gns_date'] = df['qbo_signup_date'].apply(
            lambda x: x + pd.Timedelta(days=np.random.randint(1, 30))
        )
        
        # Make sure related transactions have the same company info
        if 'user_id' in df.columns:
            # Group by user_id and assign consistent company features within each user
            for user_id in df['user_id'].unique():
                user_indices = df[df['user_id'] == user_id].index
                if len(user_indices) > 0:
                    # Select random values for this user
                    industry_name = np.random.choice(industry_types)
                    industry_code = industry_types.index(industry_name) + 1
                    qbo_product = np.random.choice(qbo_products)
                    region_id = np.random.randint(1, len(regions) + 1)
                    language_id = np.random.randint(1, len(languages) + 1)
                    company_id = np.random.randint(10000, 99999)
                    accountant_attached = np.random.choice([0, 1], p=[0.7, 0.3])
                    
                    # Assign consistent values
                    df.loc[user_indices, 'industry_name'] = industry_name
                    df.loc[user_indices, 'industry_code'] = industry_code
                    df.loc[user_indices, 'qbo_current_product'] = qbo_product
                    df.loc[user_indices, 'region_id'] = region_id
                    df.loc[user_indices, 'region_name'] = regions[region_id-1]
                    df.loc[user_indices, 'language_id'] = language_id
                    df.loc[user_indices, 'language_name'] = languages[language_id-1]
                    df.loc[user_indices, 'company_id'] = company_id
                    df.loc[user_indices, 'company_name'] = f"Company {company_id}"
                    df.loc[user_indices, 'qbo_accountant_attached_current_flag'] = accountant_attached
                    df.loc[user_indices, 'qbo_accountant_attached_ever'] = accountant_attached
    else:
        print("Found business entity features: using real company data columns")
        
    return df


class TransactionFeedbackClassifier:
    """
    Classifier for transaction data with user feedback, supporting dual prediction
    for category and tax account type.
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 256, 
                 category_dim: int = 400, tax_type_dim: int = 20,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.2,
                 use_hyperbolic: bool = True, use_neural_ode: bool = False,
                 use_ensemble: bool = False, ensemble_size: int = 3,
                 max_seq_length: int = 20, lr: float = 1e-4, 
                 weight_decay: float = 1e-5, multi_task: bool = True,
                 use_text: bool = False, text_processor: str = "finbert"):
        """
        Initialize the transaction feedback classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            category_dim: Number of categories to predict
            tax_type_dim: Number of tax account types to predict
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
            multi_task: Whether to use multi-task learning for dual prediction
            use_text: Whether to use text features (transaction descriptions)
            text_processor: Type of text processor to use
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.category_dim = category_dim
        self.tax_type_dim = tax_type_dim
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
        self.multi_task = multi_task
        self.use_text = use_text
        self.text_processor = text_processor
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.graph = None
        self.graph_builder = None
    
    def prepare_data(self, transactions_df: pd.DataFrame) -> tuple:
        """
        Prepare transaction data by building a graph and processing features.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Tuple of processed data
        """
        # Initialize graph builder
        self.graph_builder = TransactionGraphBuilder(
            num_categories=self.category_dim,
            num_tax_types=self.tax_type_dim
        )
        
        # Build graph from transaction data
        self.graph = self.graph_builder.build_graph(transactions_df)
        
        # Split graph into train/val/test sets
        self.graph = create_train_val_test_split(self.graph, group_by_user=True)
        
        # Get transaction features
        transaction_features = self.graph['transaction'].x
        
        # Get user features if available
        if 'user' in self.graph.node_types:
            user_features = self.graph['user'].x
            user_indices = self.graph['transaction', 'made_by', 'user'].edge_index[1]
            batch_size = transaction_features.size(0)
            
            # Create is_new_user tensor if available
            if hasattr(self.graph['user'], 'is_new_user'):
                is_new_user = self.graph['user'].is_new_user[user_indices]
            else:
                is_new_user = torch.zeros(batch_size, dtype=torch.bool)
        else:
            user_features = None
            is_new_user = None
            
        # Get company features if available
        company_features = None
        if 'company' in self.graph.node_types:
            # Check if there are transaction-to-company edges
            if ('transaction', 'from_company', 'company') in self.graph.edge_types:
                company_features = self.graph['company'].x
                company_indices = self.graph['transaction', 'from_company', 'company'].edge_index[1]
                
                # If company features are indexed, we need to get the right ones for each transaction
                if company_indices is not None:
                    company_features = company_features[company_indices]
            else:
                # If no explicit edges, assume 1:1 mapping with transactions
                company_features = self.graph['company'].x
                
            print(f"Using company features of shape: {company_features.shape}")
        else:
            print("No company features available in the graph")
        
        # Create dummy sequence features (we may not have temporal data)
        batch_size = transaction_features.size(0)
        seq_len = min(self.max_seq_length, 5)  # Use a small sequence length for testing
        
        # Create dummy sequence features using transaction features
        seq_features = transaction_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Create dummy timestamps
        timestamps = torch.zeros(batch_size, seq_len)
        for i in range(seq_len):
            timestamps[:, i] = i  # Simple sequential timestamps
            
        # Get t0 and t1 for ODE integration
        t0, t1 = 0.0, float(seq_len)
        
        # Extract transaction descriptions if available and text processing is enabled
        transaction_descriptions = None
        if self.use_text and 'transaction_description' in transactions_df.columns:
            transaction_descriptions = transactions_df['transaction_description'].tolist()
            
        return (
            transaction_features, seq_features, timestamps, 
            user_features, is_new_user, transaction_descriptions, company_features, t0, t1
        )
        
    def initialize_model(self, input_dim: int, graph_input_dim: int = None, company_input_dim: int = None) -> None:
        """
        Initialize the transaction model and optimizer.
        
        Args:
            input_dim: Dimension of input features
            graph_input_dim: Dimension of graph features (if different)
            company_input_dim: Dimension of company features (if different)
        """
        # Set graph input dimension if provided
        if graph_input_dim is None:
            graph_input_dim = input_dim
            
        # Initialize model
        if self.use_ensemble:
            # Not implemented for multi-task yet
            self.model = HyperTemporalEnsemble(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.category_dim,
                num_models=self.ensemble_size,
                dropout=self.dropout,
                company_input_dim=company_input_dim
            )
        else:
            self.model = HyperTemporalTransactionModel(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.category_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_hyperbolic=self.use_hyperbolic,
                use_neural_ode=self.use_neural_ode,
                use_text_processor=self.use_text,
                text_processor_type=self.text_processor,
                graph_input_dim=graph_input_dim,
                company_input_dim=company_input_dim,
                tax_type_output_dim=self.tax_type_dim,
                multi_task=self.multi_task
            )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
    def train(self, transaction_features: torch.Tensor, seq_features: torch.Tensor,
              timestamps: torch.Tensor, user_features: Optional[torch.Tensor] = None,
              is_new_user: Optional[torch.Tensor] = None, 
              transaction_descriptions: Optional[List[str]] = None,
              company_features: Optional[torch.Tensor] = None,
              t0: float = 0.0, t1: float = 10.0,
              num_epochs: int = 100, patience: int = 10) -> dict:
        """
        Train the transaction model.
        
        Args:
            transaction_features: Graph transaction features
            seq_features: Sequence features (may be dummy)
            timestamps: Sequence timestamps
            user_features: User features (optional)
            is_new_user: Boolean tensor for new users (optional)
            transaction_descriptions: List of transaction descriptions (optional)
            company_features: Company features (optional)
            t0: Start time for ODE integration
            t1: End time for ODE integration
            num_epochs: Maximum training epochs
            patience: Early stopping patience
            
        Returns:
            Dictionary of training metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized before training")
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Move data to device
        transaction_features = transaction_features.to(device)
        seq_features = seq_features.to(device)
        timestamps = timestamps.to(device)
        
        if user_features is not None:
            user_features = user_features.to(device)
        
        if is_new_user is not None:
            is_new_user = is_new_user.to(device)
        
        # Get train/val masks for transactions
        train_mask = self.graph['transaction'].train_mask
        val_mask = self.graph['transaction'].val_mask
        
        # Get labels (using presented/accepted values)
        if hasattr(self.graph['transaction'], 'y_category'):
            y_category = self.graph['transaction'].y_category
        else:
            # Fallback
            y_category = self.graph['transaction'].y
            
        if hasattr(self.graph['transaction'], 'y_tax_type'):
            y_tax_type = self.graph['transaction'].y_tax_type
        else:
            # Dummy tax type
            y_tax_type = torch.zeros_like(y_category)
        
        # Initialize learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=num_epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )
        
        # Initialize metrics
        metrics = {
            'epoch': [],
            'train_loss': [],
            'train_category_acc': [],
            'train_tax_type_acc': [],
            'val_loss': [],
            'val_category_acc': [],
            'val_tax_type_acc': []
        }
        
        # Initialize early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Enable gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    # Get predictions
                    logits = self.model(
                        transaction_features, seq_features, transaction_features,
                        timestamps, t0, t1, transaction_descriptions,
                        auto_align_dims=True, user_features=user_features,
                        is_new_user=is_new_user, company_features=company_features
                    )
                    
                    # Compute loss based on multi-task or single-task
                    if self.multi_task:
                        category_logits, tax_type_logits = logits
                        category_loss = nn.functional.cross_entropy(
                            category_logits[train_mask], y_category[train_mask]
                        )
                        tax_type_loss = nn.functional.cross_entropy(
                            tax_type_logits[train_mask], y_tax_type[train_mask]
                        )
                        # Combined loss with weighting
                        train_loss = 0.7 * category_loss + 0.3 * tax_type_loss
                    else:
                        # Single task loss (category only)
                        train_loss = nn.functional.cross_entropy(
                            logits[train_mask], y_category[train_mask]
                        )
                
                # Backward pass with scaling
                scaler.scale(train_loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Standard forward pass without mixed precision
                logits = self.model(
                    transaction_features, seq_features, transaction_features,
                    timestamps, t0, t1, transaction_descriptions,
                    auto_align_dims=True, user_features=user_features,
                    is_new_user=is_new_user, company_features=company_features
                )
                
                # Compute loss
                if self.multi_task:
                    category_logits, tax_type_logits = logits
                    category_loss = nn.functional.cross_entropy(
                        category_logits[train_mask], y_category[train_mask]
                    )
                    tax_type_loss = nn.functional.cross_entropy(
                        tax_type_logits[train_mask], y_tax_type[train_mask]
                    )
                    train_loss = 0.7 * category_loss + 0.3 * tax_type_loss
                else:
                    train_loss = nn.functional.cross_entropy(
                        logits[train_mask], y_category[train_mask]
                    )
                
                # Backward and optimize
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Compute training accuracy
            if self.multi_task:
                category_logits, tax_type_logits = logits
                train_category_acc = self._compute_accuracy(
                    category_logits[train_mask], y_category[train_mask]
                )
                train_tax_type_acc = self._compute_accuracy(
                    tax_type_logits[train_mask], y_tax_type[train_mask]
                )
            else:
                train_category_acc = self._compute_accuracy(
                    logits[train_mask], y_category[train_mask]
                )
                train_tax_type_acc = 0.0  # Not applicable for single-task
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(
                    transaction_features, seq_features, transaction_features,
                    timestamps, t0, t1, transaction_descriptions,
                    auto_align_dims=True, user_features=user_features,
                    is_new_user=is_new_user, company_features=company_features
                )
                
                # Compute validation loss and accuracy
                if self.multi_task:
                    val_category_logits, val_tax_type_logits = val_logits
                    val_category_loss = nn.functional.cross_entropy(
                        val_category_logits[val_mask], y_category[val_mask]
                    )
                    val_tax_type_loss = nn.functional.cross_entropy(
                        val_tax_type_logits[val_mask], y_tax_type[val_mask]
                    )
                    val_loss = 0.7 * val_category_loss + 0.3 * val_tax_type_loss
                    
                    val_category_acc = self._compute_accuracy(
                        val_category_logits[val_mask], y_category[val_mask]
                    )
                    val_tax_type_acc = self._compute_accuracy(
                        val_tax_type_logits[val_mask], y_tax_type[val_mask]
                    )
                else:
                    val_loss = nn.functional.cross_entropy(
                        val_logits[val_mask], y_category[val_mask]
                    )
                    val_category_acc = self._compute_accuracy(
                        val_logits[val_mask], y_category[val_mask]
                    )
                    val_tax_type_acc = 0.0  # Not applicable for single-task
            
            # Update metrics
            metrics['epoch'].append(epoch)
            metrics['train_loss'].append(train_loss.item())
            metrics['train_category_acc'].append(train_category_acc)
            metrics['train_tax_type_acc'].append(train_tax_type_acc)
            metrics['val_loss'].append(val_loss.item())
            metrics['val_category_acc'].append(val_category_acc)
            metrics['val_tax_type_acc'].append(val_tax_type_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss.item():.4f} | "
                      f"Cat Acc: {train_category_acc:.4f} | "
                      f"Tax Acc: {train_tax_type_acc:.4f} | "
                      f"Val Loss: {val_loss.item():.4f} | "
                      f"Val Cat Acc: {val_category_acc:.4f} | "
                      f"Val Tax Acc: {val_tax_type_acc:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_category_acc': val_category_acc,
                    'val_tax_type_acc': val_tax_type_acc
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f"Loaded best model from epoch {best_model_state['epoch']+1} "
                  f"with validation loss {best_model_state['val_loss']:.4f}")
        
        return metrics
    
    def evaluate(self, transaction_features: torch.Tensor, seq_features: torch.Tensor,
                timestamps: torch.Tensor, user_features: Optional[torch.Tensor] = None,
                is_new_user: Optional[torch.Tensor] = None, 
                transaction_descriptions: Optional[List[str]] = None,
                company_features: Optional[torch.Tensor] = None,
                t0: float = 0.0, t1: float = 10.0) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            transaction_features: Graph transaction features
            seq_features: Sequence features
            timestamps: Sequence timestamps
            user_features: User features (optional)
            is_new_user: Boolean tensor for new users (optional)
            transaction_descriptions: List of transaction descriptions (optional)
            company_features: Company features (optional)
            t0: Start time for ODE integration
            t1: End time for ODE integration
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None or self.graph is None:
            raise ValueError("Model and graph must be initialized and trained")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get device
        device = next(self.model.parameters()).device
        
        # Move data to device if needed
        transaction_features = transaction_features.to(device)
        seq_features = seq_features.to(device)
        timestamps = timestamps.to(device)
        
        if user_features is not None:
            user_features = user_features.to(device)
        
        if is_new_user is not None:
            is_new_user = is_new_user.to(device)
        
        # Get test mask
        test_mask = self.graph['transaction'].test_mask
        
        # Get labels
        if hasattr(self.graph['transaction'], 'y_category'):
            y_category = self.graph['transaction'].y_category
        else:
            y_category = self.graph['transaction'].y
            
        if hasattr(self.graph['transaction'], 'y_tax_type'):
            y_tax_type = self.graph['transaction'].y_tax_type
        else:
            y_tax_type = torch.zeros_like(y_category)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(
                transaction_features, seq_features, transaction_features,
                timestamps, t0, t1, transaction_descriptions,
                auto_align_dims=True, user_features=user_features,
                is_new_user=is_new_user, company_features=company_features
            )
            
            # Compute metrics based on multi-task or single-task
            if self.multi_task:
                category_logits, tax_type_logits = logits
                
                # Category metrics
                category_loss = nn.functional.cross_entropy(
                    category_logits[test_mask], y_category[test_mask]
                )
                category_acc = self._compute_accuracy(
                    category_logits[test_mask], y_category[test_mask]
                )
                category_f1 = self._compute_f1_score(
                    category_logits[test_mask], y_category[test_mask]
                )
                
                # Tax type metrics
                tax_type_loss = nn.functional.cross_entropy(
                    tax_type_logits[test_mask], y_tax_type[test_mask]
                )
                tax_type_acc = self._compute_accuracy(
                    tax_type_logits[test_mask], y_tax_type[test_mask]
                )
                tax_type_f1 = self._compute_f1_score(
                    tax_type_logits[test_mask], y_tax_type[test_mask]
                )
                
                # Combined metrics
                test_loss = 0.7 * category_loss + 0.3 * tax_type_loss
                
                # Get predictions
                y_category_pred = torch.argmax(category_logits[test_mask], dim=1).cpu().numpy()
                y_tax_type_pred = torch.argmax(tax_type_logits[test_mask], dim=1).cpu().numpy()
                
                return {
                    'test_loss': test_loss.item(),
                    'category_loss': category_loss.item(),
                    'category_acc': category_acc,
                    'category_f1': category_f1,
                    'tax_type_loss': tax_type_loss.item(),
                    'tax_type_acc': tax_type_acc,
                    'tax_type_f1': tax_type_f1,
                    'y_category_true': y_category[test_mask].cpu().numpy(),
                    'y_category_pred': y_category_pred,
                    'y_tax_type_true': y_tax_type[test_mask].cpu().numpy(),
                    'y_tax_type_pred': y_tax_type_pred
                }
            else:
                # Single task metrics (category only)
                test_loss = nn.functional.cross_entropy(
                    logits[test_mask], y_category[test_mask]
                )
                category_acc = self._compute_accuracy(
                    logits[test_mask], y_category[test_mask]
                )
                category_f1 = self._compute_f1_score(
                    logits[test_mask], y_category[test_mask]
                )
                
                # Get predictions
                y_category_pred = torch.argmax(logits[test_mask], dim=1).cpu().numpy()
                
                return {
                    'test_loss': test_loss.item(),
                    'category_acc': category_acc,
                    'category_f1': category_f1,
                    'y_category_true': y_category[test_mask].cpu().numpy(),
                    'y_category_pred': y_category_pred
                }
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'category_dim': self.category_dim,
                'tax_type_dim': self.tax_type_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'use_hyperbolic': self.use_hyperbolic,
                'use_neural_ode': self.use_neural_ode,
                'use_ensemble': self.use_ensemble,
                'ensemble_size': self.ensemble_size,
                'multi_task': self.multi_task,
                'use_text': self.use_text,
                'text_processor': self.text_processor
            }
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        # Load checkpoint
        checkpoint = torch.load(path)
        
        # Load configuration
        config = checkpoint['config']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.category_dim = config['category_dim']
        self.tax_type_dim = config['tax_type_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_hyperbolic = config['use_hyperbolic']
        self.use_neural_ode = config['use_neural_ode']
        self.use_ensemble = config['use_ensemble']
        self.ensemble_size = config['ensemble_size']
        self.multi_task = config['multi_task']
        self.use_text = config['use_text']
        self.text_processor = config['text_processor']
        
        # Initialize model
        self.initialize_model(self.input_dim)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        print(f"Model loaded from {path}")
    
    def _compute_accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute classification accuracy.
        
        Args:
            logits: Model prediction logits
            y: Ground truth labels
            
        Returns:
            Classification accuracy
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        return correct / total
    
    def _compute_f1_score(self, logits: torch.Tensor, y: torch.Tensor, average: str = 'weighted') -> float:
        """
        Compute F1 score.
        
        Args:
            logits: Model prediction logits
            y: Ground truth labels
            average: Averaging method for F1 score
            
        Returns:
            F1 score
        """
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()
        return f1_score(y_true, preds, average=average)

    
def visualize_training_metrics(metrics: dict, save_path: str = None) -> None:
    """
    Visualize training metrics.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(15, 6))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot category accuracy
    plt.subplot(1, 3, 2)
    plt.plot(metrics['epoch'], metrics['train_category_acc'], label='Train Cat Acc')
    plt.plot(metrics['epoch'], metrics['val_category_acc'], label='Val Cat Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Category Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot tax type accuracy
    plt.subplot(1, 3, 3)
    plt.plot(metrics['epoch'], metrics['train_tax_type_acc'], label='Train Tax Acc')
    plt.plot(metrics['epoch'], metrics['val_tax_type_acc'], label='Val Tax Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Tax Type Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training visualization saved to {save_path}")
    
    plt.show()


def main():
    """
    Main function for training and evaluating the transaction feedback classifier.
    """
    print("\n=== Transaction Feedback Classifier with Multi-Task Learning ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Define paths
    data_path = "data/transactions_with_feedback.csv"
    model_path = "models/transaction_feedback_model.pt"
    vis_path = "plots/training_metrics.png"
    
    # Load data
    print("\nChecking if data file exists...")
    if os.path.exists(data_path):
        # Load real data
        transactions_df = load_transaction_feedback_data(data_path)
    else:
        # Generate synthetic data if real data is not available
        print(f"Data file not found. Creating synthetic data for testing.")
        
        # Create synthetic data
        num_transactions = 5000
        num_users = 100
        num_categories = 50
        num_tax_types = 10
        
        # Create user IDs
        user_ids = np.random.randint(0, num_users, num_transactions)
        
        # Create synthetic transaction data with all the features from our schema
        transactions_df = pd.DataFrame({
            'user_id': user_ids,
            'txn_id': [f"txn_{i}" for i in range(num_transactions)],
            'is_new_user': np.random.choice([0, 1], num_transactions, p=[0.9, 0.1]),
            'presented_category_id': np.random.randint(0, num_categories, num_transactions),
            'presented_tax_account_type': np.random.randint(0, num_tax_types, num_transactions),
            'conf_score': np.random.uniform(0.5, 1.0, num_transactions),
            'amount': np.random.uniform(10.0, 1000.0, num_transactions),
            'merchant_id': np.random.randint(0, 100, num_transactions),
            'merchant_name': [f"Merchant {i}" for i in range(num_transactions)],
            'merchant_city': np.random.choice(['San Francisco', 'New York', 'Chicago', 'Dallas', 'Seattle'], num_transactions),
            'merchant_state': np.random.choice(['CA', 'NY', 'IL', 'TX', 'WA'], num_transactions),
            'cleansed_description': [f"Payment for service {i}" for i in range(num_transactions)],
            'raw_description': [f"PAYMENT FOR SERVICE {i}" for i in range(num_transactions)],
            'memo': [f"Memo {i}" for i in np.random.randint(0, 1000, num_transactions)],
            'posted_date': pd.date_range(start='2023-01-01', periods=num_transactions),
            'siccode': np.random.randint(1000, 9999, num_transactions),
            'mcc': np.random.randint(1000, 9999, num_transactions),
            'mcc_name': np.random.choice(['Retail', 'Restaurant', 'Travel', 'Services', 'Utilities'], num_transactions),
            'transaction_type': np.random.choice(['debit', 'credit'], num_transactions, p=[0.8, 0.2]),
            'account_id': np.random.randint(1000, 9999, num_transactions),
            'account_name': np.random.choice(['Checking', 'Savings', 'Credit Card', 'Business Account'], num_transactions),
            'account_type_id': np.random.randint(1, 5, num_transactions),
            'is_before_cutoff_date': np.random.choice([True, False], num_transactions, p=[0.8, 0.2]),
            'locale': np.random.choice(['en_US', 'en_CA', 'es_US'], num_transactions, p=[0.8, 0.15, 0.05])
        })
        
        # Create matching accepted values (with some disagreements to simulate user feedback)
        transactions_df['accepted_category_id'] = transactions_df['presented_category_id'].copy()
        transactions_df['accepted_tax_account_type'] = transactions_df['presented_tax_account_type'].copy()
        
        # Add category names
        category_names = [f"Category_{i}" for i in range(num_categories)]
        tax_type_names = [f"Tax_Type_{i}" for i in range(num_tax_types)]
        
        transactions_df['presented_category_name'] = transactions_df['presented_category_id'].apply(lambda x: category_names[x])
        transactions_df['accepted_category_name'] = transactions_df['accepted_category_id'].apply(lambda x: category_names[x])
        transactions_df['presented_tax_account_type_name'] = transactions_df['presented_tax_account_type'].apply(lambda x: tax_type_names[x])
        transactions_df['accepted_tax_account_type_name'] = transactions_df['accepted_tax_account_type'].apply(lambda x: tax_type_names[x])
        
        # Introduce some disagreements (user corrections)
        disagreement_mask = np.random.choice([0, 1], num_transactions, p=[0.8, 0.2])
        for idx in np.where(disagreement_mask)[0]:
            # Change some categories
            new_cat_id = (transactions_df.loc[idx, 'presented_category_id'] + np.random.randint(1, 10)) % num_categories
            transactions_df.loc[idx, 'accepted_category_id'] = new_cat_id
            transactions_df.loc[idx, 'accepted_category_name'] = category_names[new_cat_id]
            
            # Change some tax types
            new_tax_id = (transactions_df.loc[idx, 'presented_tax_account_type'] + np.random.randint(1, 5)) % num_tax_types
            transactions_df.loc[idx, 'accepted_tax_account_type'] = new_tax_id
            transactions_df.loc[idx, 'accepted_tax_account_type_name'] = tax_type_names[new_tax_id]
    
    # Initialize classifier
    print("\nInitializing transaction feedback classifier...")
    classifier = TransactionFeedbackClassifier(
        hidden_dim=128,
        category_dim=100,
        tax_type_dim=20,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        use_hyperbolic=True,
        use_neural_ode=False,  # Neural ODE is computationally expensive
        use_ensemble=False,    # Ensemble is slow for testing
        max_seq_length=10,
        lr=1e-4,
        weight_decay=1e-5,
        multi_task=True,       # Enable dual prediction
        use_text=False         # Disable text processing for now
    )
    
    # Prepare data
    print("\nPreparing transaction data...")
    (transaction_features, seq_features, timestamps, 
     user_features, is_new_user, transaction_descriptions, company_features, t0, t1) = classifier.prepare_data(transactions_df)
    
    # Get input dimension from transaction features
    input_dim = transaction_features.size(1)
    
    # Get company input dimension if available
    company_input_dim = None
    if company_features is not None:
        company_input_dim = company_features.size(1)
        print(f"Company features dimension: {company_input_dim}")
    
    # Initialize model
    classifier.initialize_model(input_dim, graph_input_dim=input_dim, company_input_dim=company_input_dim)
    
    # Print model information
    num_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    print(f"\nModel initialized with {input_dim} input features")
    print(f"Model has {num_params:,} trainable parameters")
    print(f"Multi-task learning: {'Enabled' if classifier.multi_task else 'Disabled'}")
    print(f"Hyperbolic geometry: {'Enabled' if classifier.use_hyperbolic else 'Disabled'}")
    print(f"Neural ODE: {'Enabled' if classifier.use_neural_ode else 'Disabled'}")
    print(f"Text processing: {'Enabled' if classifier.use_text else 'Disabled'}")
    
    # Train model
    print("\nTraining transaction feedback classifier...")
    metrics = classifier.train(
        transaction_features, seq_features, timestamps,
        user_features, is_new_user, transaction_descriptions,
        company_features, t0, t1, num_epochs=50, patience=10
    )
    
    # Visualize training
    visualize_training_metrics(metrics, save_path=vis_path)
    
    # Evaluate model
    print("\nEvaluating transaction feedback classifier...")
    test_metrics = classifier.evaluate(
        transaction_features, seq_features, timestamps,
        user_features, is_new_user, transaction_descriptions,
        company_features, t0, t1
    )
    
    # Print evaluation results
    print("\nTest Results:")
    print(f"Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"Category Accuracy: {test_metrics['category_acc']:.4f}")
    print(f"Category F1 Score: {test_metrics['category_f1']:.4f}")
    
    if classifier.multi_task:
        print(f"Tax Type Accuracy: {test_metrics['tax_type_acc']:.4f}")
        print(f"Tax Type F1 Score: {test_metrics['tax_type_f1']:.4f}")
    
    # Save model
    classifier.save_model(model_path)
    
    print("\n=== Transaction Feedback Classifier Training Complete ===")


if __name__ == "__main__":
    main()