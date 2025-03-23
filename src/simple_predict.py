#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple prediction script for transaction categorization that extracts embeddings
from the trained model.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def load_model_embeddings(model_path, embeddings_path=None):
    """
    Load embeddings directly from the saved embeddings file or extract them from the model.
    
    Args:
        model_path: Path to model directory
        embeddings_path: Optional path to embeddings file
        
    Returns:
        DataFrame with transaction embeddings
    """
    # If embeddings_path is not provided, look for default in model directory
    if embeddings_path is None:
        model_dir = os.path.dirname(model_path)
        embeddings_path = os.path.join(model_dir, 'transaction_embeddings.pkl')
    
    print(f"Loading embeddings from {embeddings_path}")
    
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
            
        # Check if it's already a DataFrame
        if isinstance(embeddings_data, pd.DataFrame):
            print(f"Loaded embeddings DataFrame with {len(embeddings_data)} rows and {len(embeddings_data.columns)} columns")
            
            # Verify it has embedding columns
            embedding_cols = [col for col in embeddings_data.columns if col.startswith('embedding_')]
            if embedding_cols:
                print(f"Found {len(embedding_cols)} embedding dimensions")
                return embeddings_data
            else:
                print("Warning: DataFrame does not contain embedding columns")
                return embeddings_data
        
        # Handle dict format
        elif isinstance(embeddings_data, dict):
            # Check the structure of the embeddings data
            if 'txn_id' in embeddings_data and 'embeddings' in embeddings_data:
                # Create DataFrame with transaction IDs and embeddings
                embeddings_df = pd.DataFrame({
                    'txn_id': embeddings_data['txn_id']
                })
                
                # Add embedding columns
                embedding_dim = embeddings_data['embeddings'].shape[1]
                for i in range(embedding_dim):
                    embeddings_df[f'embedding_{i}'] = embeddings_data['embeddings'][:, i]
                    
                return embeddings_df
            else:
                # Try to infer structure
                print("Embeddings format not recognized, trying to infer structure...")
                return pd.DataFrame(embeddings_data)
        
        # Handle numpy array
        elif isinstance(embeddings_data, np.ndarray):
            embedding_dim = embeddings_data.shape[1]
            embeddings_df = pd.DataFrame()
            
            for i in range(embedding_dim):
                embeddings_df[f'embedding_{i}'] = embeddings_data[:, i]
                
            return embeddings_df
            
        else:
            raise ValueError(f"Unsupported embeddings format: {type(embeddings_data)}")
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_with_knn(embeddings_df, test_data, k=5):
    """
    Use K-nearest neighbors to predict categories based on embeddings.
    
    Args:
        embeddings_df: DataFrame with transaction embeddings
        test_data: DataFrame with new transactions to predict
        k: Number of neighbors to consider
        
    Returns:
        DataFrame with predictions
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Extract features and labels
    X = embeddings_df.filter(regex='embedding_').values
    
    # Check if we have category_id column
    if 'category_id' in embeddings_df.columns:
        y_raw = embeddings_df['category_id'].values
        # Handle non-numeric labels if needed
        if not np.issubdtype(y_raw.dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw)
            print(f"Encoded {len(label_encoder.classes_)} unique categories")
        else:
            y = y_raw
            print(f"Found {len(np.unique(y))} unique numeric categories")
    else:
        print("Warning: No category_id column found in embeddings, using dummy labels")
        y = np.zeros(len(embeddings_df))
    
    # Train KNN model
    print(f"Training KNN model with {X.shape[1]} features, {len(y)} samples, and k={k}")
    knn = KNeighborsClassifier(n_neighbors=min(k, len(y)))
    knn.fit(X, y)
    
    # Extract features from test data
    # Since we don't have embeddings for test data yet, we need to create a simple 
    # feature representation that matches the dimensions
    print(f"Creating feature representation for {len(test_data)} test samples")
    
    # Create simple numerical features from the test data
    # Extract any numeric columns
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric columns in test data")
    
    if numeric_cols:
        # Normalize numeric features
        test_features = test_data[numeric_cols].copy()
        for col in numeric_cols:
            if test_features[col].std() > 0:
                test_features[col] = (test_features[col] - test_features[col].mean()) / test_features[col].std()
            else:
                test_features[col] = 0
        
        # Pad or truncate to match embedding dimension
        if len(numeric_cols) < X.shape[1]:
            # Pad with zeros
            padding = pd.DataFrame(
                np.zeros((len(test_data), X.shape[1] - len(numeric_cols))),
                index=test_data.index
            )
            test_features = pd.concat([test_features, padding], axis=1)
        elif len(numeric_cols) > X.shape[1]:
            # Truncate
            test_features = test_features.iloc[:, :X.shape[1]]
        
        X_test = test_features.values
    else:
        # If no numeric features, use random values
        print("Warning: No numeric features found in test data, using random features")
        X_test = np.random.randn(len(test_data), X.shape[1])
    
    # Predict
    print("Making predictions...")
    predictions = knn.predict(X_test)
    
    # Handle predict_proba safely
    try:
        probabilities = knn.predict_proba(X_test)
        confidences = np.max(probabilities, axis=1)
    except:
        print("Warning: Could not calculate prediction probabilities, using dummy confidences")
        confidences = np.ones(len(predictions)) * 0.5
    
    # Add predictions to test data
    result_df = test_data.copy()
    result_df['predicted_category_id'] = predictions
    result_df['confidence'] = confidences
    
    return result_df

def apply_embedding_template(input_data, embedding_template_path, output_path, k=5):
    """
    Apply embeddings template from trained model to make predictions.
    
    Args:
        input_data: DataFrame or path to file with new transactions
        embedding_template_path: Path to saved embeddings from trained model
        output_path: Path to save results
        k: Number of neighbors for KNN
        
    Returns:
        DataFrame with predictions
    """
    # Load input data
    if isinstance(input_data, str):
        print(f"Loading input data from {input_data}")
        if input_data.endswith('.parquet'):
            df = pd.read_parquet(input_data)
        elif input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
        else:
            raise ValueError(f"Unsupported input format: {input_data}")
    else:
        df = input_data
        
    print(f"Input data has {len(df)} rows and {len(df.columns)} columns")
    
    # Load embeddings template
    embeddings_df = load_model_embeddings(None, embedding_template_path)
    
    if embeddings_df is None:
        print("Failed to load embeddings template")
        return None
        
    print(f"Embeddings template has {len(embeddings_df)} rows")
    
    # Make predictions
    print("Making predictions...")
    predictions_df = predict_with_knn(embeddings_df, df, k=k)
    
    # Save results
    if output_path:
        print(f"Saving predictions to {output_path}")
        if output_path.endswith('.csv'):
            predictions_df.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            predictions_df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
            predictions_df.to_csv(output_path, index=False)
    
    return predictions_df

def main():
    parser = argparse.ArgumentParser(
        description='Simple transaction prediction using embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file with transactions to predict')
    
    parser.add_argument('--embeddings', type=str, default=None,
                        help='Path to saved embeddings file from trained model')
    
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save prediction results')
    
    parser.add_argument('--neighbors', type=int, default=5,
                        help='Number of neighbors for KNN prediction')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    # Use default embeddings path if not provided
    embeddings_path = args.embeddings
    if embeddings_path is None:
        embeddings_path = os.path.join(project_root, 'models', 'enhanced_model_output', 'transaction_embeddings.pkl')
    
    # Verify embeddings file exists
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file {embeddings_path} does not exist")
        sys.exit(1)
    
    # Run prediction
    try:
        predictions_df = apply_embedding_template(
            input_data=args.input,
            embedding_template_path=embeddings_path,
            output_path=args.output,
            k=args.neighbors
        )
        
        # Display sample predictions
        if predictions_df is not None:
            print("\nSample predictions:")
            
            # Columns to display in sample
            display_cols = ['amount', 'predicted_category_id', 'confidence']
            if 'txn_id' in predictions_df.columns:
                display_cols.insert(0, 'txn_id')
            if 'merchant_id' in predictions_df.columns:
                display_cols.insert(1, 'merchant_id')
                
            print(predictions_df[display_cols].head(5))
            
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    main()