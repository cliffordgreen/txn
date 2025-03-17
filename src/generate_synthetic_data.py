import pandas as pd
import numpy as np
import random
import os
import string
import uuid
from datetime import datetime, timedelta

def generate_synthetic_transaction_data(num_transactions=10000, num_merchants=200, num_categories=400, 
                                       output_dir='data', num_files=5, save_csv=True):
    """
    Generate synthetic transaction data and save it to multiple parquet files and CSV.
    The data format is based on the example.csv format.
    
    Args:
        num_transactions: Number of transactions to generate
        num_merchants: Number of merchants
        num_categories: Number of categories
        output_dir: Directory to save data files
        num_files: Number of parquet files to create (data will be split across these)
        save_csv: Whether to also save as CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'parquet_files'), exist_ok=True)
    
    # Calculate transactions per file
    transactions_per_file = num_transactions // num_files
    
    # Generate category names and mappings
    category_names = [f"Category_{i}" for i in range(1, num_categories + 1)]
    category_name_map = {i+1: name for i, name in enumerate(category_names)}
    
    # Create SIC codes and MCC codes for merchants
    sic_codes = [f"{random.randint(1000, 9999)}" for _ in range(num_merchants)]
    mcc_codes = [f"{random.randint(1000, 9999)}" for _ in range(num_merchants)]
    mcc_names = ["grocery store", "restaurant", "retail", "utilities", "travel", "services", 
                "entertainment", "education", "healthcare", "technology"]
    
    # Generate merchant locations
    cities = ["San Francisco", "New York", "Chicago", "Dallas", "Seattle", "Boston", "Miami", "Austin", "Los Angeles", "Denver"]
    states = ["CA", "NY", "IL", "TX", "WA", "MA", "FL", "TX", "CA", "CO"]
    
    # Generate merchant phone numbers
    phone_numbers = [f"{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(1000, 9999)}" for _ in range(num_merchants)]
    
    # Generate languages and regions
    languages = [("en_US", "en", 1), ("en_UK", "en", 2), ("es_US", "es", 3), ("fr_CA", "fr", 4)]
    regions = [("US_WEST", "US", 1), ("US_EAST", "US", 2), ("US_CENTRAL", "US", 3), ("CANADA", "US", 4), ("UK", "UK", 5), ("APAC", "APAC", 6)]
    
    # Generate industries
    industries = [
        ("Retail", 1, "SIC"), 
        ("Professional Services", 2, "SIC"), 
        ("Manufacturing", 3, "SIC"), 
        ("Construction", 4, "SIC"), 
        ("Healthcare", 5, "SIC"), 
        ("Food Service", 6, "SIC"), 
        ("Technology", 7, "SIC"), 
        ("Real Estate", 8, "SIC"), 
        ("Other Services", 9, "SIC")
    ]
    
    # Product tiers
    product_tiers = ['Simple Start', 'Essentials', 'Plus', 'Advanced']
    
    # All dataframes to be concatenated at the end
    all_dfs = []
    all_company_dfs = []
    
    for file_idx in range(num_files):
        file_transactions = transactions_per_file
        if file_idx == num_files - 1:  # Add remaining transactions to last file
            file_transactions += num_transactions % num_files
            
        # Generate transaction IDs
        transaction_ids = [f"txn_{i}" for i in range(file_idx * transactions_per_file + 1, 
                                                  file_idx * transactions_per_file + file_transactions + 1)]
        cat_transaction_ids = [f"cat_{i}" for i in range(file_idx * transactions_per_file + 1, 
                                                        file_idx * transactions_per_file + file_transactions + 1)]
        
        # Generate merchant IDs and data
        merchant_indices = np.random.randint(0, num_merchants, file_transactions)
        merchant_ids = [f"m{random.randint(1000, 9999)}" for _ in range(num_merchants)]
        selected_merchant_ids = [merchant_ids[idx] for idx in merchant_indices]
        
        # Generate merchant names for the selected merchants
        retail_merchants = ["Walmart", "Target", "Amazon", "Best Buy", "Costco", "Home Depot", "Macy's",
                          "Kroger", "Safeway", "Publix", "Whole Foods", "IKEA", "Walgreens", "CVS", 
                          "Apple Store", "GameStop", "Lowes", "Trader Joe's", "Starbucks", "McDonalds"]
        
        tech_merchants = ["Apple", "Microsoft", "Google", "Samsung", "Dell", "HP", "Adobe", "Netflix",
                         "Spotify", "Hulu", "Dropbox", "Slack", "Zoom", "AWS", "GitHub"]
        
        finance_merchants = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One",
                            "Fidelity", "Vanguard", "PayPal", "Venmo", "Square", "Robinhood"]
        
        all_merchants = retail_merchants + tech_merchants + finance_merchants
        
        merchant_names = []
        for idx in merchant_indices:
            if idx < len(all_merchants):
                merchant_names.append(all_merchants[idx])
            else:
                merchant_names.append(f"Merchant {idx % len(all_merchants) + 1}")
        
        # Generate merchant cities, states, and phones
        merchant_cities = [cities[idx % len(cities)] for idx in merchant_indices]
        merchant_states = [states[idx % len(states)] for idx in merchant_indices]
        merchant_phones = [phone_numbers[idx % len(phone_numbers)] for idx in merchant_indices]
        
        # Generate SIC and MCC data
        selected_sic_codes = [sic_codes[idx % len(sic_codes)] for idx in merchant_indices]
        selected_mcc_codes = [mcc_codes[idx % len(mcc_codes)] for idx in merchant_indices]
        selected_mcc_names = [mcc_names[idx % len(mcc_names)] for idx in merchant_indices]
        
        # Generate realistic amounts
        amount_types = np.random.choice(['small', 'medium', 'large', 'very_large'], 
                                      file_transactions, p=[0.4, 0.4, 0.15, 0.05])
        amounts = []
        for amount_type in amount_types:
            if amount_type == 'small':
                amounts.append(round(random.uniform(1, 50), 2))
            elif amount_type == 'medium':
                amounts.append(round(random.uniform(50, 500), 2))
            elif amount_type == 'large':
                amounts.append(round(random.uniform(500, 2000), 2))
            else:  # very_large
                amounts.append(round(random.uniform(2000, 10000), 2))
        
        # Generate timestamps in the last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        timestamps = []
        dates = []
        posted_dates = []
        
        for _ in range(file_transactions):
            tx_date = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
            timestamps.append(int(tx_date.timestamp()))
            dates.append(tx_date.date())
            
            # Posted date is usually 1-3 days after transaction date
            posted_date = tx_date + timedelta(days=random.randint(0, 3))
            posted_dates.append(posted_date.date())
        
        # Generate review dates and update timestamps
        review_dates = []
        update_timestamps = []
        update_dates = []
        books_create_timestamps = []
        books_create_dates = []
        
        for tx_date in dates:
            # Review date is usually 0-30 days after transaction date
            review_date = tx_date + timedelta(days=random.randint(0, 30))
            review_dates.append(review_date)
            
            # Update date & timestamp is usually 1-2 days after review date
            update_date = review_date + timedelta(days=random.randint(1, 2))
            update_dates.append(update_date)
            update_time = datetime.combine(update_date, datetime.min.time()) + timedelta(hours=random.randint(8, 17), 
                                                                                        minutes=random.randint(0, 59), 
                                                                                        seconds=random.randint(0, 59))
            update_timestamps.append(update_time)
            
            # Books create date & timestamp is usually same as update date
            books_create_date = update_date
            books_create_dates.append(books_create_date)
            books_create_time = update_time
            books_create_timestamps.append(books_create_time)
        
        # Generate transaction types
        transaction_types = np.random.choice(['debit', 'credit', 'fee', 'transfer'], file_transactions, p=[0.6, 0.25, 0.1, 0.05])
        
        # Generate online/international flags
        is_online = np.random.choice([True, False], file_transactions, p=[0.3, 0.7])
        is_international = np.random.choice([True, False], file_transactions, p=[0.1, 0.9])
        
        # Generate is_before_cutoff flags
        is_before_cutoff = np.random.choice([True, False], file_transactions, p=[0.8, 0.2])
        
        # Add user_ids (for sequence building)
        num_users = max(50, num_transactions // 30)  # Average 30 transactions per user
        user_ids = np.random.randint(1, num_users + 1, file_transactions)
        
        # Generate transaction descriptions
        purchase_prefixes = ["Purchase at", "Payment to", "Charge from", "Debit Card Purchase at"]
        transaction_descriptions = []
        raw_descriptions = []
        cleansed_descriptions = []
        memo_texts = []
        
        for i in range(file_transactions):
            prefix = random.choice(purchase_prefixes)
            merchant = merchant_names[i]
            amount = amounts[i]
            online_text = "Online" if is_online[i] else ""
            
            # Unique transaction identifier
            txn_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            
            # Raw description (uppercase, more cryptic)
            raw = f"{prefix.upper()} {merchant.upper()} {online_text.upper()} REF#{txn_ref} ${amount:.2f}".strip()
            raw_descriptions.append(raw)
            
            # Cleansed description (more readable)
            cleansed = f"{prefix} {merchant} {online_text} ${amount:.2f}".strip()
            cleansed_descriptions.append(cleansed)
            
            # Standard description for the model
            description = f"Transaction {i} for goods and services"
            transaction_descriptions.append(description)
            
            # Memo text (may be empty)
            if random.random() < 0.7:
                memo_texts.append(f"Memo {i}")
            else:
                memo_texts.append("")
        
        # Generate locale
        locales = [l[0] for l in languages]
        selected_locales = np.random.choice(locales, file_transactions)
        
        # Generate transaction validation IDs (TXN)
        txn_refs = [''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) for _ in range(file_transactions)]
        
        # Generate category IDs
        category_ids = np.random.randint(1, num_categories + 1, file_transactions)
        category_names = [category_name_map[cid] for cid in category_ids]
        
        # Generate presented categories (simulating model predictions)
        presented_category_ids = []
        for cid in category_ids:
            # 80% chance of predicting correctly, 20% chance of predicting something else
            if random.random() < 0.8:
                presented_category_ids.append(cid)
            else:
                # Pick a random category that's different from the true one
                other_categories = list(range(1, num_categories + 1))
                other_categories.remove(cid)
                presented_category_ids.append(random.choice(other_categories))
        
        presented_category_names = [category_name_map[cid] for cid in presented_category_ids]
        
        # Generate user-assigned categories
        user_category_ids = []
        for i, cid in enumerate(category_ids):
            # 90% chance of accepting the true category, 10% chance of picking something else
            if random.random() < 0.9:
                user_category_ids.append(cid)
            else:
                # Pick a random category that's different from the true one
                other_categories = list(range(1, num_categories + 1))
                other_categories.remove(cid)
                user_category_ids.append(random.choice(other_categories))
        
        # Generate company information (business entity features)
        company_ids = np.random.randint(10000, 99999, file_transactions)
        company_names = []
        for cid in company_ids:
            suffix = random.choice(["Inc.", "LLC", "Corp", "Co."])
            company_names.append(f"Company {cid} {suffix}")
        
        # Make company features consistent for the same user
        user_company_map = {}
        company_industry_names = []
        company_industry_codes = []
        company_industry_standards = []
        company_sizes = []
        company_tiers = []
        company_region_ids = []
        company_region_names = []
        company_language_ids = []
        company_language_names = []
        company_features = []  # High dimensional company features
        company_qbo_signup_dates = []
        company_qbo_gns_dates = []
        company_qbo_signup_types = []
        company_qbo_current_products = []
        company_accountant_attached_current = []
        company_accountant_attached_ever = []
        company_qblive_attach_flags = []
        company_qblive_gns_datetimes = []
        company_qblive_cancel_datetimes = []
        company_full_names = []
        company_model_buckets = []
        
        for uid in user_ids:
            if uid not in user_company_map:
                # Generate company profile
                industry_idx = random.randint(0, len(industries) - 1)
                industry_name, industry_code, industry_standard = industries[industry_idx]
                
                # Pick a size
                size = random.choice(['Small', 'Medium', 'Large'])
                
                # Pick a tier
                tier = random.choice(product_tiers)
                
                # Pick a region
                region_idx = random.randint(0, len(regions) - 1)
                region_name, region_short, region_id = regions[region_idx]
                
                # Pick a language
                language_idx = random.randint(0, len(languages) - 1)
                locale, language_name, language_id = languages[language_idx]
                
                # Generate signup date (1-3 years ago)
                years_ago = random.randint(1, 3)
                qbo_signup_date = datetime.now() - timedelta(days=365 * years_ago + random.randint(0, 30))
                qbo_signup_date = qbo_signup_date.date()
                
                # Generate GNS date (0-30 days after signup)
                qbo_gns_date = qbo_signup_date + timedelta(days=random.randint(0, 30))
                
                # Signup type
                qbo_signup_type = random.choice(["Direct", "Partner", "Conversion"])
                
                # Current product
                qbo_current_product = random.choice(product_tiers)
                
                # Accountant flags
                accountant_attached_current = random.choice([0, 1])
                accountant_attached_ever = 1 if accountant_attached_current == 1 else random.choice([0, 1])
                
                # QBLive flags
                qblive_attach_flag = random.choice([0, 1])
                
                # QBLive dates (if attached)
                if qblive_attach_flag == 1:
                    # GNS date is 0-60 days after QBO signup
                    qblive_gns_datetime = datetime.combine(qbo_signup_date, datetime.min.time()) + timedelta(days=random.randint(0, 60), 
                                                                                                          hours=random.randint(8, 17))
                    
                    # Cancel date is 30-180 days after GNS (if user cancelled)
                    if random.random() < 0.3:  # 30% chance of cancellation
                        qblive_cancel_datetime = qblive_gns_datetime + timedelta(days=random.randint(30, 180),
                                                                                hours=random.randint(8, 17))
                    else:
                        qblive_cancel_datetime = None
                else:
                    qblive_gns_datetime = None
                    qblive_cancel_datetime = None
                
                # Full company name
                full_name = f"Company {uid} {random.choice(['Inc.', 'LLC', 'Corp', 'Co.'])}"
                
                # Model bucket
                model_bucket = random.choice(['model_a', 'model_b', 'model_c'])
                
                # Create reduced company features (12-dimensional instead of 40K+)
                # This matches what's needed by the notebook
                company_vector = np.random.randn(12) * 0.2
                
                # Store all company information in the map
                user_company_map[uid] = {
                    'industry_name': industry_name,
                    'industry_code': industry_code,
                    'industry_standard': industry_standard,
                    'size': size,
                    'tier': tier,
                    'region_name': region_name,
                    'region_id': region_id,
                    'language_name': language_name,
                    'language_id': language_id,
                    'qbo_signup_date': qbo_signup_date,
                    'qbo_gns_date': qbo_gns_date,
                    'qbo_signup_type': qbo_signup_type,
                    'qbo_current_product': qbo_current_product,
                    'accountant_attached_current': accountant_attached_current,
                    'accountant_attached_ever': accountant_attached_ever,
                    'qblive_attach_flag': qblive_attach_flag,
                    'qblive_gns_datetime': qblive_gns_datetime,
                    'qblive_cancel_datetime': qblive_cancel_datetime,
                    'full_name': full_name,
                    'model_bucket': model_bucket,
                    'features': company_vector
                }
        
        # Assign company features based on user_id
        for uid in user_ids:
            company_profile = user_company_map[uid]
            company_industry_names.append(company_profile['industry_name'])
            company_industry_codes.append(company_profile['industry_code'])
            company_industry_standards.append(company_profile['industry_standard'])
            company_sizes.append(company_profile['size'])
            company_tiers.append(company_profile['tier'])
            company_region_names.append(company_profile['region_name'])
            company_region_ids.append(company_profile['region_id'])
            company_language_names.append(company_profile['language_name'])
            company_language_ids.append(company_profile['language_id'])
            company_qbo_signup_dates.append(company_profile['qbo_signup_date'])
            company_qbo_gns_dates.append(company_profile['qbo_gns_date'])
            company_qbo_signup_types.append(company_profile['qbo_signup_type'])
            company_qbo_current_products.append(company_profile['qbo_current_product'])
            company_accountant_attached_current.append(company_profile['accountant_attached_current'])
            company_accountant_attached_ever.append(company_profile['accountant_attached_ever'])
            company_qblive_attach_flags.append(company_profile['qblive_attach_flag'])
            company_qblive_gns_datetimes.append(company_profile['qblive_gns_datetime'])
            company_qblive_cancel_datetimes.append(company_profile['qblive_cancel_datetime'])
            company_full_names.append(company_profile['full_name'])
            company_model_buckets.append(company_profile['model_bucket'])
            company_features.append(company_profile['features'])
        
        # Generate ScheduleC data
        scheduleC_ids = np.random.randint(1, 5, file_transactions)
        scheduleC_names = [f"Schedule C-{id}" if id < 4 else "None" for id in scheduleC_ids]
        
        # Generate account data
        account_ids = [f"acc{random.randint(1000, 9999)}" for _ in range(file_transactions)]
        account_names = np.random.choice(["Checking", "Savings", "Credit Card", "Business Account"], file_transactions)
        account_type_ids = np.random.randint(1, 5, file_transactions)
        tax_account_types = np.random.choice(["Business", "Personal", "Mixed", "Unknown"], file_transactions)
        parent_ids = np.random.randint(1, 5, file_transactions)
        account_create_dates = [(datetime.now() - timedelta(days=random.randint(30, 365*2))).date() for _ in range(file_transactions)]
        
        # Generate more user features
        profile_methodids = np.random.randint(1, 10, file_transactions)
        header_offering_ids = np.random.randint(200, 900, file_transactions)
        is_new_users = np.random.choice([0, 1], file_transactions, p=[0.9, 0.1])
        
        # Generate tax details
        presented_tax_types = np.random.randint(0, 10, file_transactions)
        presented_tax_type_names = [f"Tax_Type_{id}" for id in presented_tax_types]
        accepted_tax_types = presented_tax_types.copy()
        
        # Sometimes accepted tax type differs from presented
        for i in range(file_transactions):
            if random.random() < 0.2:  # 20% chance of correction
                other_tax_types = list(range(10))
                other_tax_types.remove(presented_tax_types[i])
                accepted_tax_types[i] = random.choice(other_tax_types)
        
        accepted_tax_type_names = [f"Tax_Type_{id}" for id in accepted_tax_types]
        
        # Create DataFrame for this batch with all required columns from example.csv
        df = pd.DataFrame({
            # User and transaction identifiers
            'user_id': user_ids,
            'txn_id': transaction_ids,
            'cat_txn_id': cat_transaction_ids,
            'is_new_user': is_new_users,
            
            # Timestamps
            'books_create_timestamp': books_create_timestamps,
            'generated_timestamp': [datetime.now() for _ in range(file_transactions)],
            'books_create_date': books_create_dates,
            'review_date': review_dates,
            'update_timestamp': update_timestamps,
            'update_date': update_dates,
            
            # Locale
            'locale': selected_locales,
            'TXN': txn_refs,
            
            # Transaction details
            'amount': amounts,
            'posted_date': posted_dates,
            'transaction_type': transaction_types,
            'is_before_cutoff_date': is_before_cutoff,
            
            # Merchant information
            'merchant_id': selected_merchant_ids,
            'merchant_name': merchant_names,
            'merchant_city': merchant_cities,
            'merchant_state': merchant_states,
            'merchant_phone': merchant_phones,
            
            # Descriptions
            'cleansed_description': cleansed_descriptions,
            'raw_description': raw_descriptions,
            'description': transaction_descriptions,
            'memo': memo_texts,
            
            # Merchant classification
            'siccode': selected_sic_codes,
            'mcc': selected_mcc_codes,
            'mcc_name': selected_mcc_names,
            
            # ScheduleC data
            'scheduleC_id': scheduleC_ids,
            'scheduleC': scheduleC_names,
            
            # Account information
            'account_id': account_ids,
            'account_name': account_names,
            'account_type_id': account_type_ids,
            'tax_account_type': tax_account_types,
            'parent_id': parent_ids,
            'account_create_date': account_create_dates,
            
            # User profile information
            'profile_methodid': profile_methodids,
            'user_category_id': user_category_ids,
            'header_offering_id': header_offering_ids,
            
            # Company information
            'company_model_bucket_name': company_model_buckets,
            'company_id': company_ids,
            'company_name': company_names,
            'qbo_signup_date': company_qbo_signup_dates,
            'qbo_gns_date': company_qbo_gns_dates,
            'qbo_signup_type_desc': company_qbo_signup_types,
            'qbo_current_product': company_qbo_current_products,
            'qbo_accountant_attached_current_flag': company_accountant_attached_current,
            'qbo_accountant_attached_ever': company_accountant_attached_ever,
            'qblive_attach_flag': company_qblive_attach_flags,
            'qblive_gns_datetime': company_qblive_gns_datetimes,
            'qblive_cancel_datetime': company_qblive_cancel_datetimes,
            
            # Industry and region information
            'industry_name': company_industry_names,
            'industry_code': company_industry_codes,
            'industry_standard': company_industry_standards,
            'region_id': company_region_ids,
            'region_name': company_region_names,
            'language_id': company_language_ids,
            'language_name': company_language_names,
            'full_name': company_full_names,
            
            # Category information
            'presented_category_id': presented_category_ids,
            'presented_tax_account_type': presented_tax_types,
            'presented_category_name': presented_category_names,
            'presented_tax_account_type_name': presented_tax_type_names,
            'accepted_category_id': category_ids,
            'accepted_tax_account_type': accepted_tax_types,
            'accepted_category_name': category_names,
            'accepted_tax_account_type_name': accepted_tax_type_names,
            'category_name': category_names,
            'category_id': category_ids,
        })
        
        # Convert company features to numpy array
        company_features_np = np.array(company_features)
        
        # Save to parquet file - explicitly include "transaction" in the name
        output_file = os.path.join(output_dir, 'parquet_files', f"transaction_data_batch_{file_idx + 1}.parquet")
        df.to_parquet(output_file, index=False)
        
        # Store the DataFrame for CSV concatenation if needed
        all_dfs.append(df)
        
        # Save company features to a separate parquet file (reduced dimensions)
        # Create a dictionary of columns first to avoid fragmentation warnings
        feature_dict = {'user_id': user_ids, 'company_id': company_ids}
        
        # Add company features to the dictionary
        for i in range(company_features_np.shape[1]):  # Only 12 dimensions now
            feature_dict[f'feature_{i}'] = company_features_np[:, i]
            
        # Create DataFrame all at once to avoid fragmentation
        company_df = pd.DataFrame(feature_dict)
        all_company_dfs.append(company_df)
            
        company_file = os.path.join(output_dir, 'parquet_files', f"company_features_batch_{file_idx + 1}.parquet")
        company_df.to_parquet(company_file, index=False)
        
        print(f"Generated synthetic data batch {file_idx + 1}/{num_files}: {file_transactions} transactions")
    
    # Save a combined CSV file if requested
    if save_csv:
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Save to CSV
        csv_file = os.path.join(output_dir, "synthetic_transactions.csv")
        combined_df.to_csv(csv_file, index=False)
        print(f"Combined data saved to CSV: {csv_file}")
        
        # Save a smaller sample for business data
        sample_size = min(500, len(combined_df))
        sample_df = combined_df.sample(sample_size)
        sample_csv = os.path.join(output_dir, "sample_business_data.csv")
        sample_df.to_csv(sample_csv, index=False)
        print(f"Sample business data saved to CSV: {sample_csv}")
        
        # Save the combined company features
        combined_company_df = pd.concat(all_company_dfs, ignore_index=True)
        company_csv = os.path.join(output_dir, "company_features.csv")
        combined_company_df.to_csv(company_csv, index=False)
        print(f"Company features saved to CSV: {company_csv}")
    
    print(f"Synthetic transaction data saved to {output_dir}/parquet_files")
    print(f"Created {num_files} parquet files with a total of {num_transactions} transactions")
    
    return combined_df

if __name__ == "__main__":
    # Generate synthetic data (smaller set for faster processing)
    generate_synthetic_transaction_data(
        num_transactions=1000,  # Smaller number for faster processing
        num_merchants=100,
        num_categories=200,
        output_dir='data',
        num_files=2,  # Fewer files for quicker generation
        save_csv=True
    )