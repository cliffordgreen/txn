#!/usr/bin/env python3
"""
Test script for running the transaction classifier with business entity features
using synthetic data that includes all variables from the provided schema.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Import our modules
from src.train_with_feedback_data import TransactionFeedbackClassifier, load_transaction_feedback_data
from src.data_processing.transaction_graph import TransactionGraphBuilder

def generate_complete_synthetic_data(num_transactions=1000, num_users=20):
    """
    Generate fully synthetic transaction data including all variables from the schema.
    
    Args:
        num_transactions: Number of transactions to generate
        num_users: Number of unique users to simulate
        
    Returns:
        DataFrame with synthetic transaction data
    """
    print(f"Generating {num_transactions} synthetic transactions for {num_users} users...")
    
    # Create user IDs (group users into companies)
    user_ids = np.random.randint(0, num_users, num_transactions)
    
    # Set up basic categories for classification
    num_categories = 30
    num_tax_types = 10
    
    # Define various categorical variables
    qbo_products = ['Simple Start', 'Essentials', 'Plus', 'Advanced']
    industry_types = ['Retail', 'Professional Services', 'Manufacturing', 'Construction', 
                     'Healthcare', 'Food Service', 'Technology', 'Real Estate', 'Other Services']
    signup_types = ['Direct', 'Partner', 'Trial', 'Conversion']
    regions = ['US_WEST', 'US_EAST', 'US_CENTRAL', 'CANADA', 'UK', 'APAC']
    languages = ['en_US', 'en_CA', 'en_UK', 'es_US', 'fr_CA']
    locales = ['en_US', 'en_CA', 'es_US', 'fr_CA', 'en_UK']
    transaction_types = ['debit', 'credit', 'transfer', 'deposit', 'withdrawal']
    scheduleC_types = ['Schedule C-1', 'Schedule C-2', 'Schedule C-3', 'None']
    
    # Generate timestamps
    now = datetime.now()
    base_date = now - timedelta(days=365)  # 1 year ago
    
    # Create base transaction data with all string IDs to avoid mixed type issues
    transactions_df = pd.DataFrame({
        # Core transaction identifiers
        'user_id': [str(id) for id in user_ids],  # Convert to strings
        'txn_id': [f"txn_{i}" for i in range(num_transactions)],
        'cat_txn_id': [f"cat_{i}" for i in range(num_transactions)],
        'is_new_user': np.random.choice([0, 1], num_transactions, p=[0.85, 0.15]),
        
        # Transaction timestamps
        'books_create_timestamp': [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_transactions)],
        'generated_timestamp': [now - timedelta(days=np.random.randint(0, 30)) for _ in range(num_transactions)],
        
        # Transaction details
        'amount': np.random.uniform(10.0, 5000.0, num_transactions),
        'posted_date': [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_transactions)],
        'transaction_type': np.random.choice(transaction_types, num_transactions, p=[0.6, 0.25, 0.05, 0.05, 0.05]),
        'is_before_cutoff_date': np.random.choice([True, False], num_transactions, p=[0.8, 0.2]),
        'locale': np.random.choice(locales, num_transactions, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        
        # Merchant information
        'merchant_id': [f"m{id}" for id in np.random.randint(1000, 9999, num_transactions)],  # String merchant IDs
        'merchant_name': [f"Merchant {i % 100}" for i in range(num_transactions)],
        'merchant_city': np.random.choice(['San Francisco', 'New York', 'Chicago', 'Dallas', 'Seattle'], num_transactions),
        'merchant_state': np.random.choice(['CA', 'NY', 'IL', 'TX', 'WA'], num_transactions),
        'merchant_phone': [f"555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(num_transactions)],
        
        # Description fields
        'cleansed_description': [f"Payment for service {i}" for i in range(num_transactions)],
        'raw_description': [f"PAYMENT FOR SERVICE {i}" for i in range(num_transactions)],
        'description': [f"Transaction {i} for goods and services" for i in range(num_transactions)],
        'memo': [f"Memo {i}" if np.random.random() > 0.5 else "" for i in range(num_transactions)],
        
        # Classification codes as strings to avoid mixed types
        'siccode': [str(code) for code in np.random.randint(1000, 9999, num_transactions)],
        'mcc': [str(code) for code in np.random.randint(1000, 9999, num_transactions)],
        'mcc_name': np.random.choice(['Retail', 'Restaurant', 'Travel', 'Services', 'Utilities'], num_transactions),
        'scheduleC_id': [str(id) for id in np.random.randint(1, 5, num_transactions)],
        'scheduleC': np.random.choice(scheduleC_types, num_transactions),
        
        # Account information
        'account_id': [f"acc{id}" for id in np.random.randint(1000, 9999, num_transactions)],
        'account_name': np.random.choice(['Checking', 'Savings', 'Credit Card', 'Business Account'], num_transactions),
        'account_type_id': [str(id) for id in np.random.randint(1, 5, num_transactions)],
        'tax_account_type': np.random.choice(['Business', 'Personal', 'Mixed', 'Unknown'], num_transactions),
        'parent_id': [str(i) if i is not None else "" for i in np.random.choice([None, 1, 2, 3, 4, 5], num_transactions)],
        'account_create_date': [base_date - timedelta(days=np.random.randint(0, 730)) for _ in range(num_transactions)],
        
        # Target fields - category will be generated later
        'profile_methodid': [str(id) for id in np.random.randint(1, 10, num_transactions)],
        'user_category_id': np.random.randint(0, num_categories, num_transactions),
        
        # Header information
        'header_offering_id': [str(id) for id in np.random.randint(100, 999, num_transactions)],
        'books_create_date': [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_transactions)],
        
        # Machine learning metadata
        'company_model_bucket_name': np.random.choice(['model_a', 'model_b', 'model_c'], num_transactions),
    })
    
    # Extract dates as strings to match typical API format
    transactions_df['books_create_date'] = transactions_df['books_create_date'].dt.strftime('%Y-%m-%d')
    transactions_df['posted_date'] = transactions_df['posted_date'].dt.strftime('%Y-%m-%d')
    transactions_df['account_create_date'] = transactions_df['account_create_date'].dt.strftime('%Y-%m-%d')
    transactions_df['books_create_timestamp'] = transactions_df['books_create_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    transactions_df['generated_timestamp'] = transactions_df['generated_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add company ID and company data using string keys
    company_data = {}
    for user_id in np.unique(user_ids):
        # Generate company identifier
        company_id = np.random.randint(10000, 99999)
        
        # Generate QBO-specific information
        qbo_signup_date = base_date - timedelta(days=np.random.randint(30, 730))
        qbo_gns_date = qbo_signup_date + timedelta(days=np.random.randint(1, 30))
        qbo_product = np.random.choice(qbo_products)
        industry_name = np.random.choice(industry_types)
        signup_type = np.random.choice(signup_types)
        
        # Generate accountant relationship
        accountant_attached = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Generate QuickBooks Live information
        qblive_attach = np.random.choice([0, 1], p=[0.8, 0.2])
        if qblive_attach:
            qblive_gns_datetime = qbo_signup_date + timedelta(days=np.random.randint(5, 60))
            if np.random.random() < 0.2:  # 20% chance of cancellation
                qblive_cancel_datetime = qblive_gns_datetime + timedelta(days=np.random.randint(10, 180))
            else:
                qblive_cancel_datetime = None
        else:
            qblive_gns_datetime = None
            qblive_cancel_datetime = None
        
        # Generate region and language information
        region_id = np.random.randint(1, len(regions) + 1)
        language_id = np.random.randint(1, len(languages) + 1)
        
        # Store all company information - use string key to match the string user_id
        company_data[str(user_id)] = {
            'company_id': str(company_id),
            'company_name': f"Company {company_id}",
            'qbo_signup_date': qbo_signup_date.strftime('%Y-%m-%d'),
            'qbo_gns_date': qbo_gns_date.strftime('%Y-%m-%d'),
            'qbo_signup_type_desc': signup_type,
            'qbo_current_product': qbo_product,
            'qbo_accountant_attached_current_flag': accountant_attached,
            'qbo_accountant_attached_ever': accountant_attached,
            'qblive_attach_flag': qblive_attach,
            'qblive_gns_datetime': qblive_gns_datetime.strftime('%Y-%m-%d %H:%M:%S') if qblive_gns_datetime else None,
            'qblive_cancel_datetime': qblive_cancel_datetime.strftime('%Y-%m-%d %H:%M:%S') if qblive_cancel_datetime else None,
            'industry_name': industry_name,
            'industry_code': industry_types.index(industry_name) + 1,
            'industry_standard': 'SIC',
            'region_id': region_id,
            'region_name': regions[region_id - 1],
            'language_id': language_id,
            'language_name': languages[language_id - 1],
            'full_name': f"Company {company_id} Inc."
        }
    
    # Apply company data to transactions
    # Use string keys to avoid KeyError
    for column in list(company_data.values())[0].keys():
        transactions_df[column] = transactions_df['user_id'].apply(lambda x: company_data[x][column])
    
    # Generate target variables (category)
    category_names = [f"Category_{i}" for i in range(num_categories)]
    tax_type_names = [f"Tax_Type_{i}" for i in range(num_tax_types)]
    
    # Generate presented predictions (simulating model outputs)
    transactions_df['presented_category_id'] = np.random.randint(0, num_categories, num_transactions)
    transactions_df['presented_tax_account_type'] = np.random.randint(0, num_tax_types, num_transactions)
    
    transactions_df['presented_category_name'] = transactions_df['presented_category_id'].apply(lambda x: category_names[x])
    transactions_df['presented_tax_account_type_name'] = transactions_df['presented_tax_account_type'].apply(
        lambda x: tax_type_names[x]
    )
    
    # Generate "accepted" values (simulating user feedback)
    transactions_df['accepted_category_id'] = transactions_df['presented_category_id'].copy()
    transactions_df['accepted_tax_account_type'] = transactions_df['presented_tax_account_type'].copy()
    transactions_df['accepted_category_name'] = transactions_df['presented_category_name'].copy()
    transactions_df['accepted_tax_account_type_name'] = transactions_df['presented_tax_account_type_name'].copy()
    
    # Add some disagreements to simulate user corrections
    correction_mask = np.random.choice([0, 1], num_transactions, p=[0.8, 0.2])
    for idx in np.where(correction_mask)[0]:
        # For some transactions, user disagreed with the category
        new_cat_id = (transactions_df.loc[idx, 'presented_category_id'] + np.random.randint(1, 5)) % num_categories
        transactions_df.loc[idx, 'accepted_category_id'] = new_cat_id
        transactions_df.loc[idx, 'accepted_category_name'] = category_names[new_cat_id]
        
        # Sometimes tax type also changes
        if np.random.random() < 0.5:
            new_tax_id = (transactions_df.loc[idx, 'presented_tax_account_type'] + np.random.randint(1, 3)) % num_tax_types
            transactions_df.loc[idx, 'accepted_tax_account_type'] = new_tax_id
            transactions_df.loc[idx, 'accepted_tax_account_type_name'] = tax_type_names[new_tax_id]
    
    # Set the target variable as the accepted category
    transactions_df['category_name'] = transactions_df['accepted_category_name']
    
    # Convert None values to empty strings/NaNs where necessary
    for col in transactions_df.columns:
        # If column is object type, convert None to empty string
        if transactions_df[col].dtype == 'object':
            transactions_df[col] = transactions_df[col].fillna('')
            
    print(f"Generated synthetic data with {len(transactions_df.columns)} features")
    print(f"Features include: {', '.join(transactions_df.columns[:10])}... and more")
    
    return transactions_df

def run_with_synthetic_data():
    """
    Run the transaction classifier model using synthetic data with business entity features.
    """
    # Generate synthetic data
    transactions_df = generate_complete_synthetic_data(num_transactions=2000, num_users=50)
    
    # Save sample data for inspection
    os.makedirs('data', exist_ok=True)
    sample_path = 'data/sample_business_data.csv'
    transactions_df.sample(10).to_csv(sample_path, index=False)
    print(f"Saved sample data to {sample_path}")
    
    # Initialize classifier with appropriate parameters
    classifier = TransactionFeedbackClassifier(
        hidden_dim=128,
        category_dim=30,  # Match our number of categories
        tax_type_dim=10,  # Match our number of tax types
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        use_hyperbolic=True,
        use_neural_ode=False,  # Disable for faster training
        use_ensemble=False,    # Disable for faster training
        max_seq_length=5,      # Small for demo
        lr=0.001,
        weight_decay=1e-5,
        multi_task=True,       # Enable dual prediction
        use_text=False         # Disable text processing for demo
    )
    
    # Print what we're doing
    print("\nPreparing transaction data with all business entity features...")
    
    # Check specific company-related columns
    print("Checking company-related columns in the data:")
    company_columns = [col for col in transactions_df.columns 
                     if any(keyword in col.lower() for keyword in 
                           ['company', 'industry', 'qbo', 'qblive', 'region'])]
    
    for col in ['company_id', 'company_name', 'industry_name', 'qbo_current_product']:
        if col in transactions_df.columns:
            print(f"Column '{col}' found with example: {transactions_df[col].iloc[0]}")
        else:
            print(f"Column '{col}' NOT found")
            
    print(f"Found company-related columns: {', '.join(company_columns[:8])}")
    
    # Prepare data - this builds the graph and processes features
    data_tuple = classifier.prepare_data(transactions_df)
    (transaction_features, seq_features, timestamps, 
     user_features, is_new_user, transaction_descriptions, company_features, t0, t1) = data_tuple
     
    # Print debug info about company features
    print("\nCompany feature detection results:")
    print(f"Company features returned: {'Yes' if company_features is not None else 'No'}")
    if company_features is not None:
        print(f"Company features shape: {company_features.shape}")
    else:
        # Check which condition failed in extract_features method
        if hasattr(classifier.graph_builder, 'company_type_mapping'):
            print(f"company_type_mapping exists with {len(classifier.graph_builder.company_type_mapping)} entries")
        else:
            print("company_type_mapping not created")
    
    # Print feature dimensions
    if transaction_features is not None:
        print(f"\nTransaction features shape: {transaction_features.shape}")
    if seq_features is not None:
        print(f"Sequence features shape: {seq_features.shape}")
    if user_features is not None:
        print(f"User features shape: {user_features.shape}")
    if company_features is not None:
        print(f"Company features shape: {company_features.shape}")
    
    # Get dimensions
    input_dim = transaction_features.size(1)
    company_input_dim = company_features.size(1) if company_features is not None else None
    
    # Initialize model with correct dimensions
    print("\nInitializing model with appropriate dimensions...")
    print(f"Input dimension: {input_dim}, Company input dimension: {company_input_dim}")
    classifier.initialize_model(
        input_dim, 
        graph_input_dim=input_dim, 
        company_input_dim=company_input_dim
    )
    
    # Print model stats
    num_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters")
    
    # Train model with a few epochs for demonstration
    print("\nTraining model with business features (sample run)...")
    metrics = classifier.train(
        transaction_features, seq_features, timestamps,
        user_features, is_new_user, transaction_descriptions,
        company_features, t0, t1,
        num_epochs=10,  # Just a sample run
        patience=3      # Stop early if no improvement
    )
    
    # Evaluate model performance
    print("\nEvaluating model with business features...")
    test_metrics = classifier.evaluate(
        transaction_features, seq_features, timestamps,
        user_features, is_new_user, transaction_descriptions,
        company_features, t0, t1
    )
    
    # Print results
    print("\nTest Results:")
    print(f"Category Accuracy: {test_metrics['category_acc']:.4f}")
    if 'category_f1' in test_metrics:
        print(f"Category F1 Score: {test_metrics['category_f1']:.4f}")
    if 'tax_type_acc' in test_metrics:
        print(f"Tax Type Accuracy: {test_metrics['tax_type_acc']:.4f}")
        print(f"Tax Type F1 Score: {test_metrics['tax_type_f1']:.4f}")
    
    # Compare with and without business features (simple ablation study)
    print("\nRunning ablation study to compare with and without business features...")
    
    # Get test mask for ablation study
    test_mask = classifier.graph['transaction'].test_mask
    
    # Evaluate without company features by setting them to None
    with torch.no_grad():
        classifier.model.eval()
        if classifier.multi_task:
            # For multi-task mode
            category_logits, _ = classifier.model(
                transaction_features, seq_features, transaction_features,
                timestamps, t0, t1,
                user_features=user_features,
                is_new_user=is_new_user,
                company_features=None  # Explicitly set to None
            )
            # Only use test mask to get predictions for test samples
            predictions_without_company = torch.argmax(category_logits[test_mask], dim=1).cpu().numpy()
        else:
            # For single-task mode
            logits = classifier.model(
                transaction_features, seq_features, transaction_features,
                timestamps, t0, t1,
                user_features=user_features,
                is_new_user=is_new_user,
                company_features=None
            )
            # Only use test mask to get predictions for test samples
            predictions_without_company = torch.argmax(logits[test_mask], dim=1).cpu().numpy()
    
    # Get predictions with company features and ground truth
    predictions_with_company = test_metrics['y_category_pred']
    y_true = test_metrics['y_category_true']
    
    # Verify shapes match
    print(f"Shape check - predictions_with_company: {predictions_with_company.shape}, " +
          f"predictions_without_company: {predictions_without_company.shape}, " +
          f"y_true: {y_true.shape}")
    
    # Calculate accuracy
    accuracy_with_company = np.mean(predictions_with_company == y_true)
    accuracy_without_company = np.mean(predictions_without_company == y_true)
    
    print(f"Accuracy with company features: {accuracy_with_company:.4f}")
    print(f"Accuracy without company features: {accuracy_without_company:.4f}")
    print(f"Improvement from company features: {(accuracy_with_company - accuracy_without_company) * 100:.2f}%")
    
    # Save model for future use
    os.makedirs('models', exist_ok=True)
    model_path = 'models/transaction_with_business_features.pt'
    classifier.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Return the results
    return {
        'classifier': classifier,
        'metrics': metrics,
        'test_metrics': test_metrics,
        'accuracy_with_company': accuracy_with_company,
        'accuracy_without_company': accuracy_without_company
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the test
    results = run_with_synthetic_data()
    
    # Plot training metrics at the end
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(results['metrics']['epoch'], results['metrics']['train_loss'], label='Train')
    plt.plot(results['metrics']['epoch'], results['metrics']['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot category accuracy
    plt.subplot(1, 3, 2)
    plt.plot(results['metrics']['epoch'], results['metrics']['train_category_acc'], label='Train')
    plt.plot(results['metrics']['epoch'], results['metrics']['val_category_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Category Classification Accuracy')
    plt.legend()
    
    # Plot tax type accuracy
    plt.subplot(1, 3, 3)
    plt.plot(results['metrics']['epoch'], results['metrics']['train_tax_type_acc'], label='Train')
    plt.plot(results['metrics']['epoch'], results['metrics']['val_tax_type_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Tax Type Classification Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/business_features_training.png')
    print("Saved training metrics plot to plots/business_features_training.png")
    
    print("\nTest completed successfully!")