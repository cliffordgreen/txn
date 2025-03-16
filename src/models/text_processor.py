import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re

class TransactionTextProcessor:
    """
    Processes transaction descriptions to extract meaningful features for transaction classification.
    Uses a combination of pattern matching, keyword extraction, and text embedding techniques.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 max_length: int = 32, use_pretrained: bool = True):
        """
        Initialize the transaction text processor.
        
        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings
            max_length: Maximum sequence length
            use_pretrained: Whether to use pretrained embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.use_pretrained = use_pretrained
        
        # Common payment-related stopwords that should be removed
        self.stopwords = {
            'payment', 'purchase', 'transaction', 'pos', 'debit', 'credit', 'card',
            'online', 'web', 'ach', 'transfer', 'fee', 'charge', 'autopay', 'auto',
            'recurring', 'subscription', 'bill', 'paid', 'pay', 'payment', 'pymt'
        }
        
        # Common merchant prefixes/suffixes to normalize
        self.normalize_patterns = [
            (r'\binc\b\.?', ''),
            (r'\bllc\b\.?', ''),
            (r'\bltd\b\.?', ''),
            (r'\bcorp\b\.?', ''),
            (r'\bcorporation\b', ''),
            (r'\bco\b\.?', ''),
            (r'\b(usa|us|uk|ca)\b', ''),
            (r'[^a-z0-9\s]', ' '),  # Replace non-alphanumeric with space
            (r'\s+', ' ')           # Collapse multiple spaces
        ]
        
        # Regular expressions for extracting structured information
        self.patterns = {
            'date': r'\b\d{1,2}[\/\-\.]\d{1,2}(?:[\/\-\.]\d{2,4})?\b',
            'amount': r'\$\s*\d+(?:\.\d{2})?',
            'reference_number': r'ref\s*(?:#|num|number)?\s*[\w\d]+',
            'location': r'(?:in|at)\s+([a-z]+(?:\s+[a-z]+){0,3})',
        }
        
        # Category-specific keywords (expanded with subcategories)
        self.category_keywords = {
            'food': [
                'restaurant', 'cafe', 'diner', 'eatery', 'bistro', 'grill', 'steakhouse',
                'pizzeria', 'bakery', 'coffeeshop', 'deli', 'taco', 'burger', 'sushi',
                'thai', 'chinese', 'italian', 'mexican', 'breakfast', 'lunch', 'dinner',
                'brunch', 'donut', 'bagel', 'sandwich', 'salad', 'seafood', 'buffet',
                'takeout', 'delivery', 'uber eats', 'doordash', 'grubhub', 'seamless'
            ],
            'grocery': [
                'supermarket', 'grocery', 'market', 'food', 'produce', 'organic', 'farm',
                'walmart', 'target', 'costco', 'safeway', 'kroger', 'trader', 'whole foods',
                'aldi', 'publix', 'wegmans', 'shop', 'store', 'mart', 'fresh', 'natural',
                'butcher', 'bakery', 'deli'
            ],
            'transportation': [
                'airline', 'flight', 'airport', 'airfare', 'ticket', 'travel', 'uber', 'lyft',
                'taxi', 'cab', 'rideshare', 'transit', 'bus', 'train', 'subway', 'metro', 'rail',
                'rental', 'car', 'gas', 'fuel', 'station', 'parking', 'toll', 'transport',
                'amtrak', 'greyhound', 'megabus', 'turo', 'zipcar'
            ],
            'shopping': [
                'store', 'shop', 'mall', 'outlet', 'boutique', 'retail', 'merchandise', 'mart',
                'amazon', 'ebay', 'etsy', 'online', 'ecommerce', 'department', 'warehouse',
                'clothing', 'apparel', 'fashion', 'shoes', 'accessories', 'jewelry', 'watch',
                'electronics', 'computer', 'phone', 'appliance', 'furniture', 'home', 'kitchen',
                'beauty', 'cosmetic', 'makeup', 'skincare', 'fragrance', 'perfume'
            ],
            'entertainment': [
                'movie', 'theater', 'cinema', 'film', 'concert', 'show', 'event', 'ticket',
                'netflix', 'hulu', 'disney', 'spotify', 'apple', 'music', 'subscription',
                'game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam', 'twitch',
                'book', 'ebook', 'kindle', 'audible', 'podcast', 'magazine', 'newspaper',
                'sport', 'fitness', 'gym', 'club', 'recreation', 'hobby', 'leisure'
            ],
            'utilities': [
                'electric', 'electricity', 'power', 'utility', 'gas', 'water', 'sewer', 'waste',
                'phone', 'mobile', 'cell', 'wireless', 'telecom', 'internet', 'cable', 'tv',
                'broadband', 'network', 'isp', 'provider', 'service', 'bill', 'monthly', 'usage'
            ],
            'housing': [
                'rent', 'lease', 'mortgage', 'home', 'apartment', 'condo', 'house', 'housing',
                'property', 'real estate', 'hoa', 'maintenance', 'repair', 'improvement',
                'furniture', 'appliance', 'decor', 'renovation', 'remodel', 'construction',
                'garden', 'lawn', 'yard', 'pool', 'cleaning', 'security', 'insurance'
            ],
            'health': [
                'medical', 'health', 'doctor', 'physician', 'hospital', 'clinic', 'care',
                'dental', 'dentist', 'vision', 'optometrist', 'eye', 'specialist', 'therapy',
                'pharmacy', 'drug', 'prescription', 'medication', 'medicine', 'supplement',
                'vitamin', 'fitness', 'wellness', 'gym', 'exercise', 'workout', 'training',
                'insurance', 'premium', 'copay', 'deductible', 'coverage'
            ],
            'education': [
                'school', 'college', 'university', 'education', 'academic', 'tuition', 'fee',
                'course', 'class', 'program', 'degree', 'certificate', 'training', 'learning',
                'book', 'textbook', 'supply', 'material', 'software', 'tool', 'equipment',
                'loan', 'student', 'scholarship', 'grant', 'financial aid', 'study', 'research'
            ],
            'personal': [
                'salon', 'spa', 'hair', 'nail', 'beauty', 'barber', 'stylist', 'massage',
                'cosmetic', 'makeup', 'skincare', 'treatment', 'therapy', 'service',
                'clothing', 'apparel', 'fashion', 'tailor', 'alterations', 'laundry', 'dry clean',
                'personal', 'care', 'hygiene', 'grooming', 'wellness', 'self', 'improvement'
            ],
            'financial': [
                'bank', 'credit union', 'financial', 'institution', 'invest', 'investment',
                'broker', 'trading', 'stock', 'bond', 'mutual fund', 'etf', 'retirement',
                'ira', '401k', 'pension', 'annuity', 'insurance', 'tax', 'accounting',
                'loan', 'mortgage', 'debt', 'credit', 'interest', 'fee', 'service', 'charge'
            ],
            'charity': [
                'donation', 'donate', 'charity', 'charitable', 'nonprofit', 'non-profit',
                'organization', 'foundation', 'fund', 'fundraiser', 'support', 'cause',
                'volunteer', 'community', 'service', 'humanitarian', 'relief', 'development',
                'social', 'welfare', 'aid', 'assistance', 'help', 'give', 'giving'
            ],
            'subscription': [
                'subscription', 'recurring', 'monthly', 'annual', 'yearly', 'service',
                'membership', 'plan', 'access', 'premium', 'account', 'platform', 'site',
                'streaming', 'media', 'content', 'digital', 'online', 'cloud', 'storage',
                'software', 'app', 'application', 'tool', 'solution', 'product'
            ]
        }
        
        # Transaction types
        self.transaction_types = {
            'purchase': ['purchase', 'buy', 'payment', 'paid', 'pos', 'debit'],
            'refund': ['refund', 'return', 'credit', 'rebate', 'cashback', 'reimbursement'],
            'withdrawal': ['withdrawal', 'atm', 'cash'],
            'deposit': ['deposit', 'direct deposit', 'payment received', 'transfer from'],
            'transfer': ['transfer', 'zelle', 'venmo', 'paypal', 'sent', 'received'],
            'fee': ['fee', 'charge', 'service charge', 'maintenance', 'overdraft'],
            'interest': ['interest', 'dividend', 'yield', 'earned']
        }
        
        # Initialize vocabulary and embeddings
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        # Initialize embedding model
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        
        # Load pretrained embeddings if enabled
        if self.use_pretrained:
            self._init_pretrained_embeddings()
    
    def _init_pretrained_embeddings(self):
        """
        Initialize pretrained word embeddings.
        This is a placeholder method. In a real implementation, it would load
        pretrained word vectors (e.g., GloVe, Word2Vec) for financial/transaction text.
        """
        # In a real implementation, this would load pretrained embeddings
        # For demonstration, we just initialize with random values
        nn.init.xavier_uniform_(self.embedding.weight.data)
    
    def _expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        contractions = {
            "n't": " not",
            "'s": " is",
            "'m": " am",
            "'re": " are",
            "'ll": " will",
            "'ve": " have",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def preprocess_description(self, description: str) -> str:
        """
        Preprocess transaction description.
        
        Args:
            description: Raw transaction description
            
        Returns:
            Preprocessed description
        """
        if not description:
            return ""
        
        # Convert to lowercase
        text = description.lower()
        
        # Expand contractions
        text = self._expand_contractions(text)
        
        # Apply normalization patterns
        for pattern, replacement in self.normalize_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Remove specific transaction codes and numbers
        text = re.sub(r'#\d+', '', text)         # Remove reference numbers like #12345
        text = re.sub(r'\b\d{4,}\b', '', text)   # Remove long numbers
        
        # Remove specific transaction patterns
        text = re.sub(r'pos purchase', '', text)
        text = re.sub(r'purchase authorized on', '', text)
        text = re.sub(r'card \d+', '', text)
        text = re.sub(r'tran\s*(?:saction)?\s*(?:id|date)?', '', text)
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        
        # Remove extra whitespace
        text = ' '.join(words).strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_structured_info(self, description: str) -> Dict[str, str]:
        """
        Extract structured information from transaction description.
        
        Args:
            description: Raw transaction description
            
        Returns:
            Dictionary of extracted structured information
        """
        structured_info = {}
        
        for info_type, pattern in self.patterns.items():
            matches = re.findall(pattern, description.lower())
            if matches:
                structured_info[info_type] = matches[0]
        
        return structured_info
    
    def extract_category_features(self, description: str) -> Dict[str, float]:
        """
        Extract category-related features from transaction description.
        
        Args:
            description: Preprocessed transaction description
            
        Returns:
            Dictionary mapping categories to their relevance scores
        """
        text = description.lower()
        words = set(text.split())
        
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            # Calculate how many category keywords are present in the description
            matches = sum(1 for keyword in keywords if keyword in text)
            
            # Assign a score based on the number of matches and their specificity
            if matches > 0:
                # More matches = higher score
                # Normalize by the size of the keyword list
                category_scores[category] = matches / (0.5 * len(keywords))
        
        # Normalize scores
        if category_scores:
            total_score = sum(category_scores.values())
            for category in category_scores:
                category_scores[category] /= total_score
        
        return category_scores
    
    def extract_transaction_type(self, description: str) -> Dict[str, float]:
        """
        Extract transaction type features from description.
        
        Args:
            description: Preprocessed transaction description
            
        Returns:
            Dictionary mapping transaction types to their probabilities
        """
        text = description.lower()
        
        type_scores = {}
        
        for tx_type, keywords in self.transaction_types.items():
            # Calculate how many transaction type keywords are present
            matches = sum(1 for keyword in keywords if keyword in text)
            
            # Assign a score based on the number of matches
            if matches > 0:
                type_scores[tx_type] = matches / len(keywords)
        
        # If no matches found, default to 'purchase'
        if not type_scores:
            type_scores['purchase'] = 0.5
        
        # Normalize scores
        total_score = sum(type_scores.values())
        for tx_type in type_scores:
            type_scores[tx_type] /= total_score
        
        return type_scores
    
    def extract_merchant_name(self, description: str) -> str:
        """
        Extract merchant name from transaction description.
        
        Args:
            description: Raw transaction description
            
        Returns:
            Extracted merchant name
        """
        # Remove common prefixes that appear before merchant name
        prefixes = [
            r'purchase authorized on \d{2}/\d{2} ',
            r'(pos )?purchase ',
            r'payment to ',
            r'payment for ',
            r'recurring payment to ',
            r'automatic payment to ',
            r'withdrawal from ',
            r'deposit( from)? ',
            r'(online )?transfer( to| from)? ',
            r'(online )?bill payment to ',
            r'check( card)? (purchase|payment) '
        ]
        
        text = description.lower()
        
        for prefix in prefixes:
            text = re.sub(prefix, '', text)
        
        # Remove location information (city, state)
        text = re.sub(r'\s+in\s+[\w\s]+$', '', text)
        text = re.sub(r'\s+at\s+[\w\s]+$', '', text)
        text = re.sub(r'\s+\d{5}(\-\d{4})?$', '', text)  # Remove ZIP code
        
        # Remove date information
        text = re.sub(r'\b\d{1,2}/\d{1,2}(/\d{2,4})?\b', '', text)
        
        # Remove amount information
        text = re.sub(r'\$\s*\d+(\.\d{2})?', '', text)
        
        # Remove card information
        text = re.sub(r'card \d+', '', text)
        
        # Clean up and get the first part as merchant name
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Get first part as merchant name (typically the first few words)
        parts = text.split()
        if len(parts) <= 4:
            merchant_name = text
        else:
            merchant_name = ' '.join(parts[:3])
        
        return merchant_name.strip()
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into a sequence of word indices.
        
        Args:
            text: Input text
            
        Returns:
            Sequence of word indices
        """
        # Split text into words
        words = text.split()
        
        # Convert words to indices
        indices = []
        for word in words[:self.max_length]:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                # Add new word to vocabulary if space available
                if len(self.word_to_idx) < self.vocab_size:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    indices.append(idx)
                else:
                    # Use unknown token if vocabulary is full
                    indices.append(self.word_to_idx['<UNK>'])
        
        # Pad sequence to max length
        while len(indices) < self.max_length:
            indices.append(self.word_to_idx['<PAD>'])
        
        return indices[:self.max_length]
    
    def get_embeddings(self, indices: List[int]) -> torch.Tensor:
        """
        Get word embeddings for a sequence of word indices.
        
        Args:
            indices: Sequence of word indices
            
        Returns:
            Word embeddings tensor
        """
        # Convert indices to tensor
        indices_tensor = torch.tensor(indices)
        
        # Get embeddings
        embeddings = self.embedding(indices_tensor)
        
        return embeddings
    
    def process_batch(self, descriptions: List[str]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of transaction descriptions.
        
        Args:
            descriptions: List of raw transaction descriptions
            
        Returns:
            Dictionary of processed features
        """
        # Preprocess descriptions
        preprocessed = [self.preprocess_description(desc) for desc in descriptions]
        
        # Extract structured information
        structured_info = [self.extract_structured_info(desc) for desc in descriptions]
        
        # Extract category features
        category_features = [self.extract_category_features(desc) for desc in preprocessed]
        
        # Extract transaction type features
        transaction_types = [self.extract_transaction_type(desc) for desc in preprocessed]
        
        # Extract merchant names
        merchant_names = [self.extract_merchant_name(desc) for desc in descriptions]
        
        # Tokenize preprocessed descriptions
        tokenized = [self.tokenize(desc) for desc in preprocessed]
        
        # Convert to tensors
        token_tensors = torch.tensor(tokenized)
        
        # Get embeddings
        embeddings = self.embedding(token_tensors)
        
        # Create category feature tensors (one-hot encoding)
        all_categories = sorted(list(set().union(*[set(f.keys()) for f in category_features])))
        category_tensors = torch.zeros(len(descriptions), len(all_categories))
        
        for i, features in enumerate(category_features):
            for j, category in enumerate(all_categories):
                if category in features:
                    category_tensors[i, j] = features[category]
        
        # Create transaction type tensors (one-hot encoding)
        all_types = sorted(list(set().union(*[set(t.keys()) for t in transaction_types])))
        type_tensors = torch.zeros(len(descriptions), len(all_types))
        
        for i, types in enumerate(transaction_types):
            for j, tx_type in enumerate(all_types):
                if tx_type in types:
                    type_tensors[i, j] = types[tx_type]
        
        # Return all features
        return {
            'embeddings': embeddings,
            'category_features': category_tensors,
            'transaction_types': type_tensors,
            'merchant_names': merchant_names,
            'structured_info': structured_info,
            'preprocessed_text': preprocessed
        }
    
    def process_single(self, description: str) -> Dict[str, torch.Tensor]:
        """
        Process a single transaction description.
        
        Args:
            description: Raw transaction description
            
        Returns:
            Dictionary of processed features
        """
        return self.process_batch([description])


class TransactionDescriptionCNN(nn.Module):
    """
    CNN model for processing transaction descriptions.
    Uses multiple convolutional filters of different sizes to capture
    different n-gram patterns in transaction descriptions.
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128, 
                 filter_sizes: List[int] = [2, 3, 4, 5], num_filters: int = 32,
                 dropout: float = 0.5, num_classes: int = 400):
        """
        Initialize the transaction description CNN.
        
        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden features
            filter_sizes: List of convolutional filter sizes
            num_filters: Number of filters for each size
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size
            )
            for filter_size in filter_sizes
        ])
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Transpose for convolutional layer [batch_size, embedding_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply convolutions and max-pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution
            conv_out = conv(x)
            
            # Apply ReLU
            conv_out = torch.relu(conv_out)
            
            # Apply max-pooling
            pool_out = torch.max_pool1d(conv_out, kernel_size=conv_out.shape[2])
            
            # Add to outputs
            conv_outputs.append(pool_out.squeeze(2))
        
        # Concatenate all conv outputs
        out = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        out = self.dropout_layer(out)
        
        # Apply fully connected layer
        out = self.fc(out)
        
        # Apply batch normalization
        out = self.bn(out)
        
        # Apply ReLU
        out = torch.relu(out)
        
        # Apply dropout
        out = self.dropout_layer(out)
        
        # Apply output layer
        out = self.output(out)
        
        return out


class TransactionTextTransformer(nn.Module):
    """
    Transformer model for processing transaction descriptions.
    Uses a transformer encoder to capture contextual dependencies
    in transaction descriptions.
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1,
                 num_classes: int = 400):
        """
        Initialize the transaction text transformer.
        
        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            mask: Attention mask for padding
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Apply fully connected layers
        x = self.fc(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 32):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Embedding dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridTextProcessor(nn.Module):
    """
    Hybrid model combining CNN and transformer approaches for
    processing transaction descriptions.
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128,
                 filter_sizes: List[int] = [2, 3, 4, 5], num_filters: int = 32,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.2,
                 num_classes: int = 400):
        """
        Initialize the hybrid text processor.
        
        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden features
            filter_sizes: List of convolutional filter sizes
            num_filters: Number of filters for each size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super().__init__()
        
        # CNN model
        self.cnn = TransactionDescriptionCNN(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            dropout=dropout,
            num_classes=hidden_dim  # Output to hidden layer instead of classes
        )
        
        # Transformer model
        self.transformer = TransactionTextTransformer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=hidden_dim  # Output to hidden layer instead of classes
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            mask: Attention mask for padding
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # CNN branch
        cnn_features = self.cnn(x)
        
        # Transformer branch
        transformer_features = self.transformer(x, mask)
        
        # Concatenate features
        combined = torch.cat([cnn_features, transformer_features], dim=1)
        
        # Apply fusion layer
        output = self.fusion(combined)
        
        return output