import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_transaction_data(num_transactions=10000, num_merchants=200, num_categories=400, output_file='data/synthetic_transactions.csv'):
    """
    Generate synthetic transaction data and save it to a CSV file.
    
    Args:
        num_transactions: Number of transactions to generate
        num_merchants: Number of merchants
        num_categories: Number of categories
        output_file: Path to output CSV file
    """
    # Generate random transaction data
    transaction_ids = list(range(1, num_transactions + 1))
    merchant_ids = np.random.randint(1, num_merchants + 1, num_transactions)
    category_ids = np.random.randint(1, num_categories + 1, num_transactions)
    
    # Generate realistic amounts
    amount_types = np.random.choice(['small', 'medium', 'large', 'very_large'], num_transactions, p=[0.4, 0.4, 0.15, 0.05])
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
    timestamps = [int((start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))).timestamp()) for _ in range(num_transactions)]
    
    # Generate online/international flags
    is_online = np.random.choice([True, False], num_transactions, p=[0.3, 0.7])
    is_international = np.random.choice([True, False], num_transactions, p=[0.1, 0.9])
    
    # Add user_ids (for sequence building)
    num_users = max(100, num_transactions // 30)  # Average 30 transactions per user
    user_ids = np.random.randint(1, num_users + 1, num_transactions)
    
    # Generate sample merchant names
    retail_merchants = ["Walmart", "Target", "Amazon", "Best Buy", "Costco", "Home Depot", "Macy's",
                      "Kroger", "Safeway", "Publix", "Whole Foods", "IKEA", "Walgreens", "CVS", 
                      "Apple Store", "GameStop", "Lowes", "Trader Joe's", "Starbucks", "McDonalds"]
    
    tech_merchants = ["Apple", "Microsoft", "Google", "Samsung", "Dell", "HP", "Adobe", "Netflix",
                     "Spotify", "Hulu", "Dropbox", "Slack", "Zoom", "AWS", "GitHub"]
    
    finance_merchants = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One",
                        "Fidelity", "Vanguard", "PayPal", "Venmo", "Square", "Robinhood"]
    
    merchant_names = []
    for mid in merchant_ids:
        if mid <= len(retail_merchants):
            merchant_names.append(retail_merchants[mid % len(retail_merchants)])
        elif mid <= len(retail_merchants) + len(tech_merchants):
            merchant_names.append(tech_merchants[mid % len(tech_merchants)])
        else:
            merchant_names.append(finance_merchants[mid % len(finance_merchants)])
    
    # Generate transaction descriptions
    purchase_prefixes = ["Purchase at", "Payment to", "Charge from", "Debit Card Purchase at"]
    transaction_descriptions = []
    
    for i in range(num_transactions):
        prefix = random.choice(purchase_prefixes)
        merchant = merchant_names[i]
        amount = amounts[i]
        online_text = "Online" if is_online[i] else ""
        transaction_descriptions.append(f"{prefix} {merchant} {online_text} ${amount:.2f}".strip())
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'merchant_id': merchant_ids,
        'category_id': category_ids,
        'amount': amounts,
        'timestamp': timestamps,
        'is_online': is_online,
        'is_international': is_international,
        'user_id': user_ids,
        'merchant_name': merchant_names,
        'transaction_description': transaction_descriptions
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Synthetic transaction data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate synthetic data
    generate_synthetic_transaction_data(
        num_transactions=10000,
        num_merchants=200,
        num_categories=400,
        output_file='data/synthetic_transactions.csv'
    )