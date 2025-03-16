import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple, Optional

class TransactionGraphBuilder:
    """
    Class for building a heterogeneous graph from transaction data for GNN-based classification.
    The graph consists of transaction nodes, merchant nodes, and category nodes with edges between them.
    
    Modified to handle the new format with user feedback data, including tax account types
    and dual-target classification (category and tax account type).
    
    Now includes support for business entity data like company type and size as optional features.
    """
    
    def __init__(self, num_categories: int = 400, num_tax_types: int = 20):
        """
        Initialize the TransactionGraphBuilder.
        
        Args:
            num_categories: Total number of transaction categories (default: 400)
            num_tax_types: Total number of tax account types (default: 20)
        """
        self.num_categories = num_categories
        self.num_tax_types = num_tax_types
        self.transaction_features = None
        self.merchant_features = None
        self.user_features = None
        self.company_features = None  # Added for business entity data
        self.category_mapping = None
        self.tax_type_mapping = None
        self.merchant_mapping = None
        self.user_mapping = None
        self.company_type_mapping = None  # Mapping for company types
        self.company_size_mapping = None  # Mapping for company sizes
    
    def preprocess_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess transaction data by handling missing values, encoding categorical features,
        and normalizing numerical features.
        
        Modified to handle the new format with user feedback data and dual classification targets.
        Also supports optional business metadata like company type and size.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original dataframe
        df = transactions_df.copy()
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna('unknown', inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Add merchant_id if not present (use txn_id as fallback)
        if 'merchant_id' not in df.columns:
            if 'model_provider' in df.columns:
                # Use model_provider as a proxy for merchant if available
                df['merchant_id'] = df['model_provider'].astype(str)
            else:
                # Otherwise generate a random merchant_id (for testing only) as strings
                df['merchant_id'] = [f"m{i}" for i in np.random.randint(0, 100, size=len(df))]
        
        # Process company metadata if available
        # Handle company_type from industry info
        if 'industry_name' in df.columns:
            if self.company_type_mapping is None:
                unique_industry_types = df['industry_name'].unique()
                self.company_type_mapping = {t: i for i, t in enumerate(unique_industry_types)}
            df['company_type_idx'] = df['industry_name'].map(self.company_type_mapping)
        elif 'company_type' in df.columns:
            if self.company_type_mapping is None:
                unique_company_types = df['company_type'].unique()
                self.company_type_mapping = {t: i for i, t in enumerate(unique_company_types)}
            df['company_type_idx'] = df['company_type'].map(self.company_type_mapping)
        
        # Handle company_size from QBO product info
        if 'qbo_current_product' in df.columns:
            if self.company_size_mapping is None:
                unique_product_types = df['qbo_current_product'].unique()
                self.company_size_mapping = {s: i for i, s in enumerate(unique_product_types)}
            df['company_size_idx'] = df['qbo_current_product'].map(self.company_size_mapping)
        elif 'company_size' in df.columns:
            if self.company_size_mapping is None:
                unique_company_sizes = df['company_size'].unique()
                self.company_size_mapping = {s: i for i, s in enumerate(unique_company_sizes)}
            df['company_size_idx'] = df['company_size'].map(self.company_size_mapping)
        
        # Create necessary mappings for the new data format
        
        # User mapping
        if self.user_mapping is None and 'user_id' in df.columns:
            unique_users = df['user_id'].unique()
            self.user_mapping = {u: i for i, u in enumerate(unique_users)}
            df['user_idx'] = df['user_id'].map(self.user_mapping)
        elif 'user_id' in df.columns:
            df['user_idx'] = df['user_id'].map(self.user_mapping)
        else:
            # If no user_id, create a dummy one
            df['user_id'] = range(len(df))
            df['user_idx'] = df['user_id']
            self.user_mapping = {i: i for i in range(len(df))}
        
        # Merchant mapping
        if self.merchant_mapping is None:
            unique_merchants = df['merchant_id'].unique()
            self.merchant_mapping = {m: i for i, m in enumerate(unique_merchants)}
        df['merchant_idx'] = df['merchant_id'].map(self.merchant_mapping)
        
        # Category mapping - use accepted_category_id or presented_category_id if available, otherwise fallback
        cat_id_col = None
        if 'accepted_category_id' in df.columns:
            cat_id_col = 'accepted_category_id'
        elif 'presented_category_id' in df.columns:
            cat_id_col = 'presented_category_id'
        elif 'category_id' in df.columns:
            cat_id_col = 'category_id'
        
        if cat_id_col:
            if self.category_mapping is None:
                unique_categories = df[cat_id_col].unique()
                self.category_mapping = {c: i for i, c in enumerate(unique_categories)}
            df['category_idx'] = df[cat_id_col].map(self.category_mapping)
        else:
            # If no category id columns, create a dummy classification target
            # Use random integers in the appropriate range
            df['category_id'] = np.random.randint(0, self.num_categories, size=len(df))
            if self.category_mapping is None:
                unique_categories = df['category_id'].unique()
                self.category_mapping = {c: i for i, c in enumerate(unique_categories)}
            df['category_idx'] = df['category_id'].map(self.category_mapping)
        
        # Tax account type mapping - use accepted_tax_account_type or presented_tax_account_type if available
        tax_type_col = None
        if 'accepted_tax_account_type' in df.columns:
            tax_type_col = 'accepted_tax_account_type'
        elif 'presented_tax_account_type' in df.columns:
            tax_type_col = 'presented_tax_account_type'
        
        if tax_type_col:
            if self.tax_type_mapping is None:
                unique_tax_types = df[tax_type_col].unique()
                self.tax_type_mapping = {t: i for i, t in enumerate(unique_tax_types)}
            df['tax_type_idx'] = df[tax_type_col].map(self.tax_type_mapping)
        else:
            # If no tax type columns, create a dummy classification target
            # Use random integers in the appropriate range
            df['tax_account_type'] = np.random.randint(0, self.num_tax_types, size=len(df))
            if self.tax_type_mapping is None:
                unique_tax_types = df['tax_account_type'].unique()
                self.tax_type_mapping = {t: i for i, t in enumerate(unique_tax_types)}
            df['tax_type_idx'] = df['tax_account_type'].map(self.tax_type_mapping)
        
        # Add is_new_user if not present
        if 'is_new_user' not in df.columns:
            df['is_new_user'] = 0  # Default to not new
            
        return df
    
    def extract_features(self, transactions_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract and normalize features for transactions, merchants, users, and company data.
        
        Modified to handle the new data format, additional features, and business metadata.
        
        Args:
            transactions_df: Preprocessed transaction DataFrame
            
        Returns:
            Tuple of (transaction_features, merchant_features, user_features, company_features) as PyTorch tensors
        """
        # Numerical features for transactions
        # Exclude indexes and target columns from features
        exclude_cols = ['merchant_idx', 'category_idx', 'user_idx', 'tax_type_idx',
                        'company_type_idx', 'company_size_idx',
                        'merchant_id', 'category_id', 'user_id', 'txn_id', 
                        'accepted_category_id', 'presented_category_id', 
                        'accepted_tax_account_type', 'presented_tax_account_type',
                        'company_type', 'company_size']
        
        numerical_cols = [col for col in transactions_df.columns 
                         if transactions_df[col].dtype in ['int64', 'float64'] 
                         and col not in exclude_cols]
        
        # Handle specific required fields
        required_features = []
        
        # Add is_new_user as a feature if available
        if 'is_new_user' in transactions_df.columns:
            required_features.append('is_new_user')
            
        # Add confidence score if available
        if 'conf_score' in transactions_df.columns:
            required_features.append('conf_score')
        
        # Make sure required features are included
        for feat in required_features:
            if feat in numerical_cols:
                continue
            if feat in transactions_df.columns:
                numerical_cols.append(feat)
        
        # If no numerical columns, create a dummy feature
        if not numerical_cols:
            transactions_df['dummy_feature'] = 1.0
            numerical_cols = ['dummy_feature']
            
        # Normalize numerical features
        scaler = StandardScaler()
        transaction_num_features = scaler.fit_transform(transactions_df[numerical_cols])
        
        # One-hot encode categorical features
        categorical_cols = [col for col in transactions_df.columns 
                           if transactions_df[col].dtype == 'object' 
                           and col not in exclude_cols + ['merchant_id', 'category_id', 'user_id', 'txn_id']]
        
        # Process model_provider and model_version if available
        if 'model_provider' in transactions_df.columns and 'model_provider' not in categorical_cols:
            categorical_cols.append('model_provider')
            
        if 'model_version' in transactions_df.columns and 'model_version' not in categorical_cols:
            categorical_cols.append('model_version')
        
        # Process categorical features
        if categorical_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            transaction_cat_features = encoder.fit_transform(transactions_df[categorical_cols])
            # Combine numerical and categorical features
            transaction_features = np.hstack([transaction_num_features, transaction_cat_features])
        else:
            transaction_features = transaction_num_features
        
        # Create merchant features (aggregated transaction stats per merchant)
        merchant_features = self._create_merchant_features(transactions_df)
        
        # Create user features if user_id is available
        if 'user_id' in transactions_df.columns:
            user_features = self._create_user_features(transactions_df)
        else:
            # Create dummy user features
            num_users = len(self.user_mapping) if self.user_mapping else 1
            user_features = np.zeros((num_users, 1))
            
        # Create company features if available - check for any company-related columns
        company_features = None
        company_cols = ['company_id', 'company_name', 'industry_name', 'industry_code', 
                        'qbo_current_product', 'qbo_signup_type_desc', 'region_id', 'language_id',
                        'company_type', 'company_size']
        
        found_cols = [col for col in company_cols if col in transactions_df.columns]
        if found_cols:
            print(f"Found company-related columns: {', '.join(found_cols)}")
            company_features = self._create_company_features(transactions_df)
        else:
            print("No company features available in the graph")
        
        # Convert to PyTorch tensors
        self.transaction_features = torch.FloatTensor(transaction_features)
        self.merchant_features = torch.FloatTensor(merchant_features)
        self.user_features = torch.FloatTensor(user_features)
        
        if company_features is not None:
            self.company_features = torch.FloatTensor(company_features)
        
        return self.transaction_features, self.merchant_features, self.user_features, self.company_features
        
    def _create_user_features(self, transactions_df: pd.DataFrame) -> np.ndarray:
        """
        Create features for user nodes by aggregating transaction data.
        
        Args:
            transactions_df: Preprocessed transaction DataFrame
            
        Returns:
            User features as numpy array
        """
        # Get list of possible aggregation columns
        potential_agg_cols = [col for col in transactions_df.columns 
                             if transactions_df[col].dtype in ['int64', 'float64'] 
                             and col not in ['user_idx', 'merchant_idx', 'category_idx', 'tax_type_idx']]
        
        # Define actual aggregation columns (use what's available)
        agg_cols = []
        for col in ['conf_score', 'is_new_user']:
            if col in potential_agg_cols:
                agg_cols.append(col)
                
        # If no columns to aggregate, add a dummy column
        if not agg_cols:
            transactions_df['dummy_user_col'] = 1
            agg_cols = ['dummy_user_col']
        
        # Group by user and compute aggregated statistics
        agg_dict = {col: ['mean', 'count'] for col in agg_cols}
        user_stats = transactions_df.groupby('user_idx').agg(agg_dict)
        
        # Flatten multi-index columns
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        
        # Fill NaN values with 0
        user_stats.fillna(0, inplace=True)
        
        # Add transaction count per user
        user_stats['txn_count'] = transactions_df.groupby('user_idx').size()
        
        # Handle empty user_stats (if it happens)
        if len(user_stats) == 0:
            num_users = len(self.user_mapping) if self.user_mapping else 1
            return np.zeros((num_users, 1))
        
        # Normalize user features if we have more than 1 feature
        if user_stats.shape[1] > 1:
            scaler = StandardScaler()
            user_features = scaler.fit_transform(user_stats)
        else:
            user_features = user_stats.values
            
        return user_features
    
    def _create_company_features(self, transactions_df: pd.DataFrame) -> np.ndarray:
        """
        Create features for company entities based on company metadata.
        
        Args:
            transactions_df: Preprocessed transaction DataFrame
            
        Returns:
            Company features as numpy array
        """
        # Get company metadata features
        company_features = []
        
        # Process company type if available (one-hot encode)
        if 'company_type_idx' in transactions_df.columns:
            unique_company_types = len(self.company_type_mapping) if self.company_type_mapping else 1
            company_type_features = np.eye(unique_company_types)[transactions_df['company_type_idx'].values]
            company_features.append(company_type_features)
        
        # Process company size if available (one-hot encode)
        if 'company_size_idx' in transactions_df.columns:
            unique_company_sizes = len(self.company_size_mapping) if self.company_size_mapping else 1
            company_size_features = np.eye(unique_company_sizes)[transactions_df['company_size_idx'].values]
            company_features.append(company_size_features)
        
        # Process QBO-specific data
        qbo_status_features = []
        
        # QBO signup variables
        qbo_numeric_cols = [
            col for col in transactions_df.columns 
            if col in ['qbo_signup_date', 'qbo_gns_date'] and
            transactions_df[col].dtype != 'object'
        ]
        
        if qbo_numeric_cols:
            # Convert dates to days since epoch if they're date objects
            for col in ['qbo_signup_date', 'qbo_gns_date']:
                if col in transactions_df.columns and transactions_df[col].dtype == 'object':
                    try:
                        transactions_df[f'{col}_days'] = pd.to_datetime(transactions_df[col]).dt.days
                        qbo_numeric_cols.append(f'{col}_days')
                    except:
                        pass
            
            # Normalize QBO numeric features
            scaler = StandardScaler()
            qbo_numeric_features = scaler.fit_transform(transactions_df[qbo_numeric_cols])
            qbo_status_features.append(qbo_numeric_features)
        
        # Categorical QBO features
        for col in ['qbo_signup_type_desc', 'qbo_current_product']:
            if col in transactions_df.columns:
                # Create mapping if not exists
                if not hasattr(self, f'{col}_mapping'):
                    unique_values = transactions_df[col].unique()
                    setattr(self, f'{col}_mapping', {v: i for i, v in enumerate(unique_values)})
                
                # Get mapping
                col_mapping = getattr(self, f'{col}_mapping')
                
                # One-hot encode
                transactions_df[f'{col}_idx'] = transactions_df[col].map(col_mapping)
                unique_values = len(col_mapping)
                col_features = np.eye(unique_values)[transactions_df[f'{col}_idx'].values]
                qbo_status_features.append(col_features)
        
        # Binary QBO features
        binary_cols = [
            'qbo_accountant_attached_current_flag', 
            'qbo_accountant_attached_ever',
            'qblive_attach_flag',
            'is_before_cutoff_date'
        ]
        
        for col in binary_cols:
            if col in transactions_df.columns:
                # Convert to binary
                try:
                    binary_feature = transactions_df[col].astype(float).values.reshape(-1, 1)
                    qbo_status_features.append(binary_feature)
                except:
                    # If conversion fails, try to handle as boolean or string
                    if transactions_df[col].dtype == 'bool':
                        binary_feature = transactions_df[col].astype(int).values.reshape(-1, 1)
                        qbo_status_features.append(binary_feature)
                    elif transactions_df[col].dtype == 'object':
                        # Try to convert common string representations of boolean
                        try:
                            bool_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0, 't': 1, 'f': 0}
                            binary_feature = transactions_df[col].str.lower().map(bool_map).fillna(0).values.reshape(-1, 1)
                            qbo_status_features.append(binary_feature)
                        except:
                            print(f"Could not convert column {col} to binary feature")
        
        # Add QBO status features if any
        if qbo_status_features:
            combined_qbo = np.hstack(qbo_status_features)
            company_features.append(combined_qbo)
        
        # Process industry data
        industry_features = []
        
        # One-hot encode industry info if available
        if 'industry_code' in transactions_df.columns:
            # Create mapping if not exists
            if not hasattr(self, 'industry_code_mapping'):
                unique_industries = transactions_df['industry_code'].unique()
                self.industry_code_mapping = {v: i for i, v in enumerate(unique_industries)}
            
            # One-hot encode
            transactions_df['industry_code_idx'] = transactions_df['industry_code'].map(self.industry_code_mapping)
            unique_industries = len(self.industry_code_mapping)
            industry_code_features = np.eye(unique_industries)[transactions_df['industry_code_idx'].values]
            industry_features.append(industry_code_features)
        
        # Process region and language data
        if 'region_id' in transactions_df.columns:
            # Create mapping if not exists
            if not hasattr(self, 'region_mapping'):
                unique_regions = transactions_df['region_id'].unique()
                self.region_mapping = {v: i for i, v in enumerate(unique_regions)}
            
            # One-hot encode
            transactions_df['region_idx'] = transactions_df['region_id'].map(self.region_mapping)
            unique_regions = len(self.region_mapping)
            region_features = np.eye(unique_regions)[transactions_df['region_idx'].values]
            industry_features.append(region_features)
        
        if 'language_id' in transactions_df.columns:
            # Create mapping if not exists
            if not hasattr(self, 'language_mapping'):
                unique_languages = transactions_df['language_id'].unique()
                self.language_mapping = {v: i for i, v in enumerate(unique_languages)}
            
            # One-hot encode
            transactions_df['language_idx'] = transactions_df['language_id'].map(self.language_mapping)
            unique_languages = len(self.language_mapping)
            language_features = np.eye(unique_languages)[transactions_df['language_idx'].values]
            industry_features.append(language_features)
            
        # Add industry features if any
        if industry_features:
            combined_industry = np.hstack(industry_features)
            company_features.append(combined_industry)
            
        # Add any additional company features that might be available
        additional_company_features = [
            col for col in transactions_df.columns
            if (col.startswith('company_') and 
               col not in ['company_type', 'company_size', 'company_type_idx', 'company_size_idx', 'company_id', 'company_name']) and
            transactions_df[col].dtype in ['int64', 'float64']
        ]
        
        if additional_company_features:
            # Normalize additional numerical features
            scaler = StandardScaler()
            additional_features = scaler.fit_transform(transactions_df[additional_company_features])
            company_features.append(additional_features)
        
        # Combine all company features
        if company_features:
            print(f"Combining {len(company_features)} company feature sets")
            # Check all features for valid shapes
            valid_features = []
            for i, feat in enumerate(company_features):
                if feat.shape[0] == len(transactions_df):
                    valid_features.append(feat)
                else:
                    print(f"Warning: Feature set {i} has wrong shape {feat.shape}, expected first dim {len(transactions_df)}")
                    
            if valid_features:
                combined_features = np.hstack(valid_features)
                print(f"Created combined company features with shape {combined_features.shape}")
            else:
                print("No valid company features found, creating dummy features")
                combined_features = np.zeros((len(transactions_df), 1))
        else:
            # Create dummy features if no company data is available
            print("No company feature sets found, creating dummy features")
            combined_features = np.zeros((len(transactions_df), 1))
            
        return combined_features
        
    def _create_merchant_features(self, transactions_df: pd.DataFrame) -> np.ndarray:
        """
        Create features for merchant nodes by aggregating transaction data.
        
        Args:
            transactions_df: Preprocessed transaction DataFrame
            
        Returns:
            Merchant features as numpy array
        """
        # Group by merchant and compute aggregated statistics
        merchant_stats = transactions_df.groupby('merchant_idx').agg({
            'amount': ['mean', 'std', 'min', 'max', 'count'],
            # Add more aggregations as needed
        })
        
        # Flatten multi-index columns
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns.values]
        
        # Add company-related features to merchant features if available
        if 'company_type_idx' in transactions_df.columns:
            # Get most common company type per merchant
            merchant_company_type = transactions_df.groupby('merchant_idx')['company_type_idx'].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 0
            )
            merchant_stats['company_type_idx'] = merchant_company_type
            
        if 'company_size_idx' in transactions_df.columns:
            # Get most common company size per merchant
            merchant_company_size = transactions_df.groupby('merchant_idx')['company_size_idx'].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 0
            )
            merchant_stats['company_size_idx'] = merchant_company_size
        
        # Fill NaN values with 0
        merchant_stats.fillna(0, inplace=True)
        
        # Normalize merchant features
        scaler = StandardScaler()
        merchant_features = scaler.fit_transform(merchant_stats)
        
        return merchant_features
    
    def build_graph(self, transactions_df: pd.DataFrame) -> HeteroData:
        """
        Build a heterogeneous graph from transaction data.
        
        Modified to include user nodes, company metadata, and support dual classification targets
        (category and tax account type).
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            PyTorch Geometric HeteroData object representing the transaction graph
        """
        # Preprocess transactions
        processed_df = self.preprocess_transactions(transactions_df)
        
        # Extract features
        transaction_features, merchant_features, user_features, company_features = self.extract_features(processed_df)
        
        # Create heterogeneous graph
        graph = HeteroData()
        
        # Add node features
        graph['transaction'].x = transaction_features
        graph['merchant'].x = merchant_features
        graph['user'].x = user_features
        
        # Add company nodes if available - using a broader condition
        company_cols = ['company_id', 'company_name', 'industry_name', 'industry_code', 
                        'qbo_current_product', 'qbo_signup_type_desc', 'region_id', 'language_id',
                        'company_type', 'company_size']
        has_company_data = any(col in processed_df.columns for col in company_cols)
        
        if company_features is not None:
            print(f"Adding company features of shape {company_features.shape} to graph")
            graph['company'].x = company_features
        else:
            print("No company features to add to graph nodes")
        
        # Add category nodes (one-hot encoded)
        num_categories = len(self.category_mapping)
        graph['category'].x = torch.eye(num_categories)
        
        # Add tax account type nodes (one-hot encoded)
        num_tax_types = len(self.tax_type_mapping)
        graph['tax_type'].x = torch.eye(num_tax_types)
        
        # Create transaction indices
        src_nodes = torch.tensor(range(len(processed_df)), dtype=torch.long)
        
        # Add edges: transaction -> merchant
        dst_nodes = torch.tensor(processed_df['merchant_idx'].values, dtype=torch.long)
        graph['transaction', 'belongs_to', 'merchant'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Add edges: transaction -> category
        dst_nodes = torch.tensor(processed_df['category_idx'].values, dtype=torch.long)
        graph['transaction', 'has_category', 'category'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Add edges: transaction -> tax_type
        dst_nodes = torch.tensor(processed_df['tax_type_idx'].values, dtype=torch.long)
        graph['transaction', 'has_tax_type', 'tax_type'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Add edges: transaction -> user
        dst_nodes = torch.tensor(processed_df['user_idx'].values, dtype=torch.long)
        graph['transaction', 'made_by', 'user'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Add edges for company data if available
        if has_company_data and company_features is not None:
            # For now, each transaction can be linked to the company node with the same index
            # This is a simplification; in a real implementation, you might need a more complex mapping
            print("Creating company-transaction edges")
            dst_nodes = torch.tensor(range(len(processed_df)), dtype=torch.long)
            graph['transaction', 'from_company', 'company'].edge_index = torch.stack([src_nodes, dst_nodes])
            
            # Add reverse edges for bidirectional message passing
            graph['company', 'has_transaction', 'transaction'].edge_index = torch.stack([dst_nodes, src_nodes])
        
        # Add reverse edges for bidirectional message passing
        # user -> transaction edges
        src_nodes = torch.tensor(processed_df['user_idx'].values, dtype=torch.long)
        dst_nodes = torch.tensor(range(len(processed_df)), dtype=torch.long)
        graph['user', 'makes', 'transaction'].edge_index = torch.stack([src_nodes, dst_nodes])
        
        # Add primary labels for transactions (category)
        graph['transaction'].y_category = torch.tensor(processed_df['category_idx'].values, dtype=torch.long)
        
        # Add secondary labels (tax account type)
        graph['transaction'].y_tax_type = torch.tensor(processed_df['tax_type_idx'].values, dtype=torch.long)
        
        # Store original IDs for reference
        if 'txn_id' in processed_df.columns:
            # Handle string or object dtype by storing as a list attribute
            if processed_df['txn_id'].dtype == 'object':
                graph['transaction'].txn_id_list = processed_df['txn_id'].tolist()
            else:
                # For numeric IDs, use tensor
                graph['transaction'].txn_id = torch.tensor(processed_df['txn_id'].values)
            
        if 'user_id' in processed_df.columns:
            # Handle different types of user_id
            if processed_df['user_id'].dtype == 'object':
                graph['user'].user_id_list = processed_df['user_id'].tolist()
            else:
                graph['user'].user_id = torch.tensor(processed_df['user_id'].values)
            
        # Add business metadata as node attributes
        if 'company_type' in processed_df.columns:
            graph['company'].company_type = torch.tensor(
                processed_df['company_type_idx'].values if 'company_type_idx' in processed_df.columns else []
            )
            
        if 'company_size' in processed_df.columns:
            graph['company'].company_size = torch.tensor(
                processed_df['company_size_idx'].values if 'company_size_idx' in processed_df.columns else []
            )
            
        # Add is_new_user as a separate node attribute if available
        if 'is_new_user' in processed_df.columns:
            graph['user'].is_new_user = torch.tensor(processed_df.groupby('user_idx')['is_new_user'].first().values)
        
        return graph
    
    def save_graph(self, graph: HeteroData, output_path: str) -> None:
        """
        Save the graph to disk.
        
        Args:
            graph: PyTorch Geometric HeteroData object
            output_path: Path to save the graph
        """
        torch.save(graph, output_path)
    
    def load_graph(self, input_path: str) -> HeteroData:
        """
        Load a graph from disk.
        
        Args:
            input_path: Path to the saved graph
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        return torch.load(input_path)


def create_train_val_test_split(graph: HeteroData, train_ratio: float = 0.7, 
                               val_ratio: float = 0.15, 
                               group_by_user: bool = True) -> HeteroData:
    """
    Split a graph into training, validation, and test sets.
    
    Modified to support user-based splitting to prevent data leakage.
    
    Args:
        graph: PyTorch Geometric HeteroData object
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        group_by_user: Whether to ensure all transactions from the same user 
                      stay in the same split (prevents data leakage)
        
    Returns:
        Graph with train/val/test masks added
    """
    num_transactions = graph['transaction'].x.size(0)
    
    if group_by_user and 'user' in graph.node_types and hasattr(graph['transaction'], 'made_by'):
        # Get user IDs for each transaction
        src_indices = torch.arange(num_transactions)
        user_indices = graph['transaction', 'made_by', 'user'].edge_index[1]
        
        # Group transactions by user
        user_to_txn = {}
        for txn_idx, user_idx in zip(src_indices.tolist(), user_indices.tolist()):
            if user_idx not in user_to_txn:
                user_to_txn[user_idx] = []
            user_to_txn[user_idx].append(txn_idx)
        
        # Split users
        num_users = len(user_to_txn)
        user_indices = torch.randperm(num_users).tolist()
        
        train_size = int(train_ratio * num_users)
        val_size = int(val_ratio * num_users)
        
        train_user_indices = user_indices[:train_size]
        val_user_indices = user_indices[train_size:train_size + val_size]
        test_user_indices = user_indices[train_size + val_size:]
        
        # Assign transactions to splits based on user
        train_indices = []
        for user_idx in train_user_indices:
            if user_idx in user_to_txn:
                train_indices.extend(user_to_txn[user_idx])
                
        val_indices = []
        for user_idx in val_user_indices:
            if user_idx in user_to_txn:
                val_indices.extend(user_to_txn[user_idx])
                
        test_indices = []
        for user_idx in test_user_indices:
            if user_idx in user_to_txn:
                test_indices.extend(user_to_txn[user_idx])
    else:
        # If not grouping by user or user data is not available, use random split
        indices = torch.randperm(num_transactions).tolist()
        
        train_size = int(train_ratio * num_transactions)
        val_size = int(val_ratio * num_transactions)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    
    # Create masks
    train_mask = torch.zeros(num_transactions, dtype=torch.bool)
    val_mask = torch.zeros(num_transactions, dtype=torch.bool)
    test_mask = torch.zeros(num_transactions, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # Add masks to graph
    graph['transaction'].train_mask = train_mask
    graph['transaction'].val_mask = val_mask
    graph['transaction'].test_mask = test_mask
    
    return graph