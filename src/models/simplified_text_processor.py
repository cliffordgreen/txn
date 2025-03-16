import torch
import torch.nn as nn
from typing import List, Dict, Union
import numpy as np

class SimplifiedTextProcessor:
    """
    Simplified text processor for transaction descriptions that uses 
    basic embeddings instead of transformers for faster training and testing.
    """
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 64, output_dim: int = 64):
        """
        Initialize the simplified text processor.
        
        Args:
            vocab_size: Size of vocabulary for embedding layer
            embedding_dim: Dimension of token embeddings
            output_dim: Dimension of output embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # Create a simple vocabulary mapping
        self.word_to_idx = {}
        self.next_idx = 0
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )
    
    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into word indices.
        
        Args:
            text: Input text
            
        Returns:
            List of word indices
        """
        words = text.lower().split()
        indices = []
        
        for word in words:
            if word not in self.word_to_idx:
                if self.next_idx < self.vocab_size - 1:
                    self.word_to_idx[word] = self.next_idx
                    self.next_idx += 1
                else:
                    # Use last index as UNK token
                    self.word_to_idx[word] = self.vocab_size - 1
            
            indices.append(self.word_to_idx[word])
        
        return indices
    
    def process_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Process a batch of transaction descriptions.
        
        Args:
            texts: List of transaction descriptions
            
        Returns:
            Embedding tensor [batch_size, output_dim]
        """
        batch_embeddings = []
        
        for text in texts:
            # Tokenize text
            indices = self._tokenize(text)
            
            if len(indices) == 0:
                # Empty text, use zeros
                avg_embedding = torch.zeros(self.embedding_dim)
            else:
                # Convert to tensor
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                
                # Get embeddings
                token_embeddings = self.embedding(indices_tensor)
                
                # Average pooling
                avg_embedding = torch.mean(token_embeddings, dim=0)
            
            batch_embeddings.append(avg_embedding)
        
        # Stack embeddings
        embeddings = torch.stack(batch_embeddings)
        
        # Project to output dimension
        output_embeddings = self.projection(embeddings)
        
        return output_embeddings