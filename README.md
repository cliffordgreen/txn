# Transaction Classification with User Feedback and Business Entity Integration

This project implements an advanced machine learning system to classify financial transactions into categories and tax account types, using a multi-task learning approach. The system incorporates user feedback and business entity features, allowing it to improve based on corrections made by users and to personalize predictions for specific business contexts. It leverages hyperbolic geometry, multi-modal fusion, and temporal modeling to achieve high performance.

## Project Structure

- `data/`: Directory for storing transaction datasets
- `src/`: Source code directory
  - `data_processing/`: Scripts for data preprocessing and graph construction
  - `models/`: Model implementations (GNN, Hybrid, Temporal, Hyper-Temporal variants)
  - `utils/`: Utility functions
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `requirements.txt`: Project dependencies

## Key Features

- **Business Entity Integration** for context-aware transaction classification 
- **Company-Aware Attention Mechanisms** that incorporate business metadata
- **Industry-Sensitive Representations** that adapt to different business types
- **Hyperbolic Geometry** for modeling hierarchical transaction relationships
- **Neural Ordinary Differential Equations** for continuous-time transaction dynamics
- **Multi-modal Cross-Attention Fusion** with adaptive gating mechanisms
- **Dynamic Contextual Temporal Layers** that model multiple time scales simultaneously
- **Mixture-of-Experts Ensemble** with specialized model components
- **Self-supervised Auxiliary Tasks** for improved representation learning
- **Advanced Sequence Augmentation** techniques for robust training

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

1. Basic Transaction Classification:
```bash
python src/train_transaction_classifier.py
```

2. Hyper-Temporal Model with Text Processing:
```bash
python src/train_hyper_temporal_model.py
```

3. Classification with User Feedback Data:
```bash
python src/train_with_feedback_data.py
```

### Making Predictions with User Feedback and Business Entity Model

```python
from src.train_with_feedback_data import TransactionFeedbackClassifier

# Initialize classifier
classifier = TransactionFeedbackClassifier(
    hidden_dim=128,
    category_dim=100,
    tax_type_dim=20,
    multi_task=True
)

# Load pre-trained model
classifier.load_model('models/transaction_feedback_model.pt')

# Prepare data
transaction_features, seq_features, timestamps, user_features, is_new_user, _, company_features, t0, t1 = classifier.prepare_data(df)

# Make predictions with business entity features
predictions = classifier.model(
    transaction_features, seq_features, transaction_features, 
    timestamps, t0, t1, 
    user_features=user_features, 
    is_new_user=is_new_user,
    company_features=company_features
)

# For multi-task model, predictions is a tuple of (category_logits, tax_type_logits)
```

## Key Components

### Multi-Task Learning for Dual Classification
The model performs simultaneous prediction of both category and tax account type through multi-task learning, sharing representations between tasks while maintaining task-specific heads.

### User Feedback Integration
- Uses both presented (model predictions) and accepted (user corrections) labels
- Learns from user feedback by treating accepted values as ground truth 
- Automatically adapts to user preferences over time

### Hyper-Temporal Model Architecture
- **Heterogeneous Graph Neural Network**: Models relationships between transactions, users, merchants, companies, and categories
- **Company-Aware Context Layer**: Enriches transaction representations with business entity metadata
- **Hyperbolic Encoders**: Efficiently represents hierarchical transaction patterns
- **Multi-modal Fusion**: Integrates multiple feature types (graph, sequence, tabular, text, company)
- **Dynamic Temporal Layers**: Captures evolving user and business transaction patterns
- **Business-Sensitive Attention**: Adjusts model focus based on business characteristics

### Text Processing (Optional)
The model supports various text processing approaches for transaction descriptions:
- **FinBERT**: Financial domain-specific language model
- **General LLMs**: Support for using external large language models

### Feature Support
The model accepts the following data fields:
- Required: `user_id`, `txn_id`, `is_new_user`
- Target fields: `presented_category_id/name`, `presented_tax_account_type/name`, `accepted_category_id/name`, `accepted_tax_account_type/name`
- Optional features: `conf_score`, `model_provider`, `model_version`
- Optional text: transaction descriptions (when available)
- Business entity features:
  - Company identification: `company_id`, `company_name`
  - QBO information: `qbo_signup_date`, `qbo_gns_date`, `qbo_signup_type_desc`, `qbo_current_product`
  - QBO usage flags: `qbo_accountant_attached_current_flag`, `qbo_accountant_attached_ever`, `qblive_attach_flag`
  - Industry information: `industry_name`, `industry_code`, `industry_standard`
  - Regional information: `region_id`, `region_name`, `language_id`, `language_name`
  - Account information: `account_id`, `account_name`, `account_type_id`

## Implementation Details

### Data Processing
The data processing pipeline handles:
- Missing fields with sensible defaults
- Conversion between IDs and human-readable names
- User-based train/validation/test splitting to prevent data leakage
- Business metadata extraction and normalization
- Integration of industry, region, and QBO product information
- Company-specific feature engineering and embedding

### Model Training
- Mixed precision training for efficiency
- Early stopping based on validation performance
- Learning rate scheduling with warm-up
- Support for both single-task and multi-task objectives

### Visualization
The training pipeline includes visualization tools for:
- Training and validation metrics
- Category and tax type accuracy tracking
- Learning curves and model diagnostics

## Getting Started

1. Prepare your transaction data with the required fields
2. Install dependencies with `pip install -r requirements.txt`
3. Run the training script with `python src/train_with_feedback_data.py`
4. Evaluate the model performance with the generated metrics and plots

## Repository Structure

```
transaction-classification/
├── data/                      # Data directory (CSV files)
├── models/                    # Saved model checkpoints
├── plots/                     # Visualizations and training plots
├── src/
│   ├── data_processing/       # Data processing utilities
│   │   └── transaction_graph.py # Graph construction from transactions
│   ├── models/                # Model implementations
│   │   ├── hyper_temporal_model.py # Main model architecture
│   │   └── modern_text_processor.py # Text processing modules
│   └── train_with_feedback_data.py # Main training script
└── requirements.txt           # Project dependencies
```