import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.modern_text_processor import FinBERTProcessor

class SimpleTextClassifier(nn.Module):
    """Simple text classifier that uses advanced text processors."""
    
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=400, text_processor_type="finbert"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize text processor
        if text_processor_type == "finbert":
            self.text_processor = FinBERTProcessor(
                output_dim=input_dim,
                pooling_strategy="mean",
                test_mode=False  # Using test mode for quicker execution
            )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, transaction_descriptions):
        # Process text descriptions
        text_embeddings = self.text_processor.process_batch(transaction_descriptions)
        
        # Apply classification layers
        logits = self.classifier(text_embeddings)
        
        return logits

def main():
    # Load data from CSV
    print("Loading transaction data...")
    csv_path = 'data/synthetic_transactions.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions")
    
    # Extract features and target
    descriptions = df['transaction_description'].tolist()
    categories = df['category_id'].astype(int).values
    
    # Create label encoder for categories
    label_encoder = LabelEncoder()
    encoded_categories = label_encoder.fit_transform(categories)
    num_classes = len(label_encoder.classes_)
    
    # Convert to tensors
    y = torch.tensor(encoded_categories)
    
    # Create train and test splits
    indices = list(range(len(descriptions)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_descriptions = [descriptions[i] for i in train_indices]
    test_descriptions = [descriptions[i] for i in test_indices]
    
    train_labels = y[train_indices]
    test_labels = y[test_indices]
    
    # Initialize model
    print("\nInitializing text classifier model...")
    model = SimpleTextClassifier(
        input_dim=384,
        hidden_dim=128,
        output_dim=num_classes,
        text_processor_type="finbert"
    )
    
    # Set up training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining model...")
    num_epochs = 1  # Just one epoch for testing
    batch_size = 16  # Small batch for testing
    
    # Use a very small subset for quick testing
    subset_size = min(100, len(train_descriptions))
    train_descriptions = train_descriptions[:subset_size]
    train_labels = train_labels[:subset_size]
    
    # Train model
    model.train()
    for epoch in range(num_epochs):
        # Process in small batches
        for i in range(0, len(train_descriptions), batch_size):
            batch_descriptions = train_descriptions[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size].to(device)
            
            # Forward pass
            outputs = model(batch_descriptions)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Batch {i//batch_size + 1}/{len(train_descriptions)//batch_size + 1}, Loss: {loss.item():.4f}")
    
    print("\nTraining complete!")
    
    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    
    # Use small subset for testing
    subset_size = min(20, len(test_descriptions))
    test_subset_descriptions = test_descriptions[:subset_size]
    test_subset_labels = test_labels[:subset_size].to(device)
    
    with torch.no_grad():
        outputs = model(test_subset_descriptions)
        _, predicted = torch.max(outputs, 1)
        
        accuracy = (predicted == test_subset_labels).sum().item() / test_subset_labels.size(0)
        print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nText processing model successfully tested!")

if __name__ == "__main__":
    main()