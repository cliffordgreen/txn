"""
Configuration file for enhanced GNN model training.
Contains all hyperparameters and settings for training the enhanced graph model.
"""

import os
from dataclasses import dataclass

@dataclass
class EnhancedGraphModelConfig:
    """Configuration class for the Enhanced Graph Transaction Model"""
    
    # Data configuration
    data_dir: str = "data/parquet_files"  # Directory containing parquet files
    output_dir: str = "models/enhanced_model_output"  # Directory to save model outputs
    batch_size: int = 64  # Batch size for training
    num_workers: int = 4  # Number of workers for data loading
    prefetch_factor: int = 2  # Prefetch factor for data loading
    max_files: int = 100  # Maximum number of parquet files to process
    
    # Model configuration
    hidden_dim: int = 256  # Hidden dimension size
    num_heads: int = 8  # Number of attention heads
    num_graph_layers: int = 2  # Number of graph layers
    num_temporal_layers: int = 2  # Number of temporal layers
    dropout: float = 0.2  # Dropout rate
    use_hyperbolic: bool = True  # Whether to use hyperbolic encoding
    use_neural_ode: bool = False  # Whether to use neural ODE (set to False for faster training)
    use_text: bool = True  # Whether to use text processing
    multi_task: bool = True  # Whether to enable multi-task learning
    num_relations: int = 5  # Number of relation types (company, merchant, industry, price, temporal)
    
    # Training configuration
    learning_rate: float = 3e-4  # Learning rate
    weight_decay: float = 1e-5  # Weight decay
    num_epochs: int = 10  # Number of training epochs
    patience: int = 3  # Patience for early stopping
    grad_clip: float = 1.0  # Gradient clipping value
    
    # GPU optimization
    use_amp: bool = True  # Use mixed precision training
    use_cuda_graphs: bool = True  # Use CUDA graphs for optimization
    cuda_graph_batch_size: int = 64  # Fixed batch size for CUDA graphs (same as batch_size)
    
    # XGBoost integration
    extract_embeddings: bool = True  # Whether to extract embeddings for XGBoost
    embedding_output_file: str = "transaction_embeddings.pkl"  # Output file for embeddings
    
    # Metrics
    eval_steps: int = 100  # Number of steps between evaluations
    log_steps: int = 10  # Number of steps between logging
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set CUDA graph batch size to match batch size
        self.cuda_graph_batch_size = self.batch_size

# Default configuration instance
config = EnhancedGraphModelConfig()

# Configurations for different environments

def get_p3_2xlarge_config() -> EnhancedGraphModelConfig:
    """Get configuration optimized for p3.2xlarge instances with V100 GPU"""
    p3_config = EnhancedGraphModelConfig()
    p3_config.batch_size = 512  # Larger batch size for 16GB V100
    p3_config.cuda_graph_batch_size = 512
    p3_config.num_workers = 8  # More workers for faster data loading
    p3_config.learning_rate = 5e-4  # Higher learning rate for larger batches
    p3_config.use_amp = True  # Always use mixed precision on V100
    return p3_config

def get_g4dn_xlarge_config() -> EnhancedGraphModelConfig:
    """Get configuration optimized for g4dn.xlarge instances with T4 GPU"""
    g4_config = EnhancedGraphModelConfig()
    g4_config.batch_size = 256  # Smaller batch size for 16GB T4
    g4_config.cuda_graph_batch_size = 256
    g4_config.num_workers = 4  # Fewer workers for instance with fewer vCPUs
    g4_config.use_neural_ode = False  # Disable neural ODE for faster training
    return g4_config

def get_local_config() -> EnhancedGraphModelConfig:
    """Get configuration optimized for local development"""
    local_config = EnhancedGraphModelConfig()
    local_config.batch_size = 32  # Small batch size for local GPUs
    local_config.cuda_graph_batch_size = 32
    local_config.num_workers = 2  # Fewer workers for local machine
    local_config.max_files = 5  # Process fewer files for testing
    local_config.use_neural_ode = False  # Disable neural ODE for faster training
    local_config.use_cuda_graphs = False  # Disable CUDA graphs for local testing
    return local_config