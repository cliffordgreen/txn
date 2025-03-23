import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, JumpingKnowledge, RGCNConv, GatedGraphConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Import our custom models
from src.models.hyper_temporal_model import (
    HyperTemporalTransactionModel, 
    DynamicContextualTemporal,
    MultiModalFusion
)

try:
    # Import graph model if available
    from src.models.graph_enhanced_model import GraphEnhancedTemporalModel
    HAS_GRAPH_MODEL = True
except ImportError:
    HAS_GRAPH_MODEL = False
    
# Import graph processing utilities
try:
    from src.data_processing.transaction_graph import build_transaction_relationship_graph
    HAS_GRAPH_PROCESSING = True
except ImportError:
    HAS_GRAPH_PROCESSING = False

# --- FIXED FEATURE SCHEMA (Define at the top level) ---
# This is *crucial*. It defines the *exact* features the model expects.
FEATURE_SCHEMA = {
    'numerical': [
        'amount', 'log_amount', 'user_id', 'merchant_id', 'company_id',
        'industry_code', 'region_id', 'language_id', 'account_type_id',
        'tax_account_type', 'scheduleC_id'
    ],
    'boolean': [
        'is_new_user', 'is_before_cutoff_date',
        'qbo_accountant_attached_current_flag', 'qbo_accountant_attached_ever',
        'qblive_attach_flag', 'is_test_data', 'is_validated'
    ],
    'categorical': [
        'transaction_type', 'qbo_current_product', 'qbo_signup_type_desc',
        'company_model_bucket_name', 'industry_name', 'tax_type', 'merchant_category'
    ],
    'timestamp': [  # Prioritize these
        'generated_timestamp', 'timestamp', 'books_create_timestamp', 'update_timestamp', 'transaction_date'
    ]
}

# Define fixed categories for each categorical feature
# If a category is not in this list, it will be mapped to 'UNKNOWN'
FIXED_CATEGORIES = {
    'transaction_type': ['DEBIT', 'CREDIT', 'CHECK', 'TRANSFER', 'UNKNOWN'],
    'qbo_current_product': ['QBSE', 'QBO', 'QBOA', 'OTHER', 'UNKNOWN'],
    'industry_name': ['RETAIL', 'PROFESSIONAL_SERVICES', 'CONSTRUCTION', 'MANUFACTURING', 'FINANCE', 'HEALTHCARE', 'UNKNOWN'],
    'company_type': ['LLC', 'CORPORATION', 'SOLE_PROPRIETORSHIP', 'PARTNERSHIP', 'UNKNOWN'],
    'company_size': ['SMALL', 'MEDIUM', 'LARGE', 'UNKNOWN'],
    'qbo_signup_type_desc': ['DIRECT', 'PARTNER', 'TRIAL', 'CONVERSION', 'UNKNOWN'],
    'company_model_bucket_name': ['BUCKET_1', 'BUCKET_2', 'BUCKET_3', 'BUCKET_4', 'UNKNOWN'],
    'tax_type': ['BUSINESS', 'PERSONAL', 'MIXED', 'UNKNOWN'],
    'merchant_category': ['RETAIL', 'SERVICES', 'UTILITIES', 'TRAVEL', 'FOOD', 'UNKNOWN']
}

MAX_SEQ_LEN = 5  # Fixed sequence length
INPUT_FEATURE_DIM = 128  # Base feature dimension

class HybridTransactionModel(torch.nn.Module):
    """
    Hybrid Transaction Classification Model that combines:
    1. Enhanced GNN for graph structure learning
    2. Tabular MLP for direct feature learning
    3. Attention mechanism for feature fusion
    4. Self-supervised auxiliary tasks
    5. Multi-task learning for improved generalization
    
    This hybrid approach leverages both the graph structure of transactions
    and the raw tabular features for improved classification performance.
    """
    
    def __init__(self, hidden_channels: int = 128, num_layers: int = 3, 
                 dropout: float = 0.4, conv_type: str = 'sage',
                 heads: int = 2, use_jumping_knowledge: bool = True,
                 use_batch_norm: bool = True, 
                 use_self_supervision: bool = True,
                 use_tabular_model: bool = True,
                 use_graph_transformers: bool = True,
                 metadata: Optional[Tuple] = None):
        """
        Initialize the Hybrid Transaction Model.
        
        Args:
            hidden_channels: Dimension of hidden node features
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat')
            heads: Number of attention heads for GAT
            use_jumping_knowledge: Whether to use jumping knowledge
            use_batch_norm: Whether to use batch normalization
            use_self_supervision: Whether to use self-supervised auxiliary tasks
            use_tabular_model: Whether to include tabular MLP model
            use_graph_transformers: Whether to use graph transformer layers
            metadata: Graph metadata (node types and edge types)
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.heads = heads
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_batch_norm = use_batch_norm
        self.use_self_supervision = use_self_supervision
        self.use_tabular_model = use_tabular_model
        self.use_graph_transformers = use_graph_transformers
        
        # If metadata is not provided, use default for transaction graph
        if metadata is None:
            self.metadata = (
                ['transaction', 'merchant', 'category'],  # Node types
                [('transaction', 'belongs_to', 'merchant'),  # Edge types
                 ('transaction', 'has_category', 'category'),
                 # Add reverse edges for better message passing
                 ('merchant', 'rev_belongs_to', 'transaction'),
                 ('category', 'rev_has_category', 'transaction'),
                 # Add self-loops for better information propagation
                 ('transaction', 'self', 'transaction'),
                 ('merchant', 'self', 'merchant'),
                 ('category', 'self', 'category')]
            )
        else:
            self.metadata = metadata
        
        # Input linear layers for each node type
        self.node_encoders = nn.ModuleDict()
        
        # Convolution layers
        self.convs = nn.ModuleList()
        
        # Graph transformer layers (if used)
        if self.use_graph_transformers:
            self.transformer_layers = nn.ModuleList()
        
        # Batch normalization layers if used
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList()
        
        # Tabular MLP model (if used)
        if self.use_tabular_model:
            self.tabular_mlp = nn.Sequential(
                nn.Linear(-1, hidden_channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Initialize node encoders and convolution layers
        self._init_layers()
        
        # Jumping knowledge if used
        if self.use_jumping_knowledge:
            self.jumping_knowledge = JumpingKnowledge('lstm', hidden_channels, num_layers)
        
        # Attention mechanism for fusion of GNN and tabular outputs
        if self.use_tabular_model:
            self.fusion_attention = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, 2),
                nn.Softmax(dim=1)
            )
        
        # Output layers
        self.pre_classifier = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 400)  # 400 categories
        
        # Self-supervised auxiliary task heads (if used)
        if self.use_self_supervision:
            # Merchant prediction head
            self.merchant_predictor = nn.Linear(hidden_channels, 200)  # Assuming 200 merchants
            # Transaction amount prediction head
            self.amount_predictor = nn.Linear(hidden_channels, 1)
    
    def _init_layers(self):
        """
        Initialize node encoders, convolution layers, and other model components.
        """
        # Node encoders (linear transformation of input features)
        for node_type in self.metadata[0]:
            self.node_encoders[node_type] = nn.Sequential(
                Linear(-1, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        
        # Transaction node outputs from each layer for jumping knowledge
        self.transaction_xs = []
        
        # Convolution layers
        for i in range(self.num_layers):
            conv_dict = {}
            
            # For each edge type, create a convolution
            for edge_type in self.metadata[1]:
                # Choose convolution type
                if self.conv_type == 'gcn':
                    conv = GCNConv(-1, self.hidden_channels)
                elif self.conv_type == 'sage':
                    conv = SAGEConv((-1, -1), self.hidden_channels)
                elif self.conv_type == 'gat':
                    conv = GATConv((-1, -1), self.hidden_channels // self.heads, heads=self.heads)
                else:
                    raise ValueError(f"Unsupported convolution type: {self.conv_type}")
                
                conv_dict[edge_type] = conv
            
            # Create heterogeneous convolution
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
            # Add graph transformer layer (if used)
            if self.use_graph_transformers:
                self.transformer_layers.append(nn.MultiheadAttention(
                    embed_dim=self.hidden_channels,
                    num_heads=self.heads,
                    dropout=self.dropout,
                    batch_first=True
                ))
            
            # Add batch normalization if used
            if self.use_batch_norm:
                batch_norm_dict = {}
                for node_type in self.metadata[0]:
                    batch_norm_dict[node_type] = nn.BatchNorm1d(self.hidden_channels)
                self.batch_norms.append(nn.ModuleDict(batch_norm_dict))
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                raw_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            raw_features: Raw tabular features for transactions (if using tabular model)
            
        Returns:
            Dictionary containing model outputs, including classification logits
            and self-supervised task outputs if enabled
        """
        outputs = {}
        
        # Encode node features
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_encoders[node_type](x)
        
        # Store transaction node outputs for jumping knowledge
        if self.use_jumping_knowledge:
            self.transaction_xs = [x_dict['transaction']]
        
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply graph transformer (if used)
            if self.use_graph_transformers:
                for node_type in x_dict_new.keys():
                    # Reshape for attention mechanism
                    x_reshaped = x_dict_new[node_type].unsqueeze(0)
                    
                    # Apply transformer layer
                    x_transformed, _ = self.transformer_layers[i](
                        x_reshaped, x_reshaped, x_reshaped
                    )
                    
                    # Update features
                    x_dict_new[node_type] = x_transformed.squeeze(0)
            
            # Apply batch normalization, residual connections, and non-linearity
            for node_type in x_dict_new.keys():
                # Apply batch normalization if used
                if self.use_batch_norm:
                    x_dict_new[node_type] = self.batch_norms[i][node_type](x_dict_new[node_type])
                
                # Add residual connection (except for first layer)
                if i > 0:
                    x_dict_new[node_type] += x_dict[node_type]
                
                # Apply non-linearity and dropout
                x_dict_new[node_type] = F.relu(x_dict_new[node_type])
                x_dict_new[node_type] = F.dropout(x_dict_new[node_type], p=self.dropout, training=self.training)
            
            # Update node features
            x_dict = x_dict_new
            
            # Store transaction node outputs for jumping knowledge
            if self.use_jumping_knowledge:
                self.transaction_xs.append(x_dict['transaction'])
        
        # Apply jumping knowledge to transaction nodes if used
        if self.use_jumping_knowledge:
            x_transaction_gnn = self.jumping_knowledge(self.transaction_xs)
        else:
            x_transaction_gnn = x_dict['transaction']
        
        # Process tabular data if provided and tabular model is enabled
        if self.use_tabular_model and raw_features is not None:
            x_transaction_tabular = self.tabular_mlp(raw_features)
            
            # Fusion of GNN and tabular outputs using attention
            combined_features = torch.cat([x_transaction_gnn, x_transaction_tabular], dim=1)
            attention_weights = self.fusion_attention(combined_features)
            
            # Apply attention weights
            x_transaction = (
                attention_weights[:, 0].unsqueeze(1) * x_transaction_gnn +
                attention_weights[:, 1].unsqueeze(1) * x_transaction_tabular
            )
        else:
            x_transaction = x_transaction_gnn
        
        # Apply classifier to transaction nodes
        x_transaction = self.pre_classifier(x_transaction)
        x_transaction = F.relu(x_transaction)
        x_transaction = F.dropout(x_transaction, p=self.dropout, training=self.training)
        logits = self.classifier(x_transaction)
        
        # Store main output
        outputs['logits'] = logits
        
        # Apply self-supervised auxiliary task heads if enabled
        if self.use_self_supervision:
            # Merchant prediction
            merchant_logits = self.merchant_predictor(x_transaction)
            outputs['merchant_logits'] = merchant_logits
            
            # Transaction amount prediction
            amount_pred = self.amount_predictor(x_transaction)
            outputs['amount_pred'] = amount_pred
        
        return outputs
    
    def predict(self, graph: HeteroData, raw_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions on a heterogeneous graph.
        
        Args:
            graph: PyTorch Geometric HeteroData object
            raw_features: Raw tabular features for transactions (if using tabular model)
            
        Returns:
            Predicted category probabilities for each transaction
        """
        self.eval()
        with torch.no_grad():
            # Extract node features and edge indices from graph
            x_dict = {node_type: graph[node_type].x for node_type in self.metadata[0]}
            edge_index_dict = {edge_type: graph[edge_type].edge_index 
                               for edge_type in self.metadata[1] if edge_type in graph}
            
            # Forward pass
            outputs = self(x_dict, edge_index_dict, raw_features)
            
            # Apply softmax to get probabilities
            probs = F.softmax(outputs['logits'], dim=1)
            
            return probs


class HybridTransactionEnsemble(torch.nn.Module):
    """
    Ensemble model that combines multiple hybrid transaction models for improved
    performance through model averaging and specialization.
    """
    
    def __init__(self, num_models: int = 3, hidden_channels: int = 128, 
                 num_layers: List[int] = [2, 3, 4], 
                 dropout: float = 0.4, 
                 conv_types: List[str] = ['gcn', 'sage', 'gat'],
                 use_bagging: bool = True):
        """
        Initialize the ensemble model.
        
        Args:
            num_models: Number of models in the ensemble
            hidden_channels: Base dimension of hidden node features
            num_layers: List of number of layers for each model
            dropout: Dropout probability
            conv_types: List of convolution types for each model
            use_bagging: Whether to use bagging (bootstrap aggregating)
        """
        super().__init__()
        
        self.num_models = num_models
        self.use_bagging = use_bagging
        
        # Create ensemble of models
        self.models = nn.ModuleList()
        
        for i in range(num_models):
            # Vary model configurations
            model = HybridTransactionModel(
                hidden_channels=hidden_channels + (i * 16),  # Vary hidden channels
                num_layers=num_layers[i % len(num_layers)],  # Vary number of layers
                dropout=dropout,
                conv_type=conv_types[i % len(conv_types)],  # Vary convolution type
                use_jumping_knowledge=(i % 2 == 0),  # Vary JK usage
                use_batch_norm=True,
                use_self_supervision=(i % 2 == 0),  # Vary self-supervision
                use_tabular_model=True,
                use_graph_transformers=(i % 3 == 0)  # Vary transformer usage
            )
            
            self.models.append(model)
        
        # Meta-learner for weighted ensemble combination
        self.meta_learner = nn.Sequential(
            nn.Linear(num_models * 400, 256),  # 400 categories per model
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 400)  # Final output for 400 categories
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                raw_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the ensemble model.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            raw_features: Raw tabular features for transactions
            
        Returns:
            Final logits from the ensemble
        """
        all_logits = []
        
        # Get predictions from each model
        for model in self.models:
            outputs = model(x_dict, edge_index_dict, raw_features)
            all_logits.append(outputs['logits'])
        
        # Simple ensemble: average predictions
        if self.meta_learner is None:
            ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        else:
            # Meta-learner for weighted combination
            concatenated_logits = torch.cat(all_logits, dim=1)
            ensemble_logits = self.meta_learner(concatenated_logits)
        
        return ensemble_logits
    
    def predict(self, graph: HeteroData, raw_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions with the ensemble model.
        
        Args:
            graph: PyTorch Geometric HeteroData object
            raw_features: Raw tabular features for transactions
            
        Returns:
            Predicted category probabilities for each transaction
        """
        self.eval()
        with torch.no_grad():
            # Extract node features and edge indices from graph
            x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
            edge_index_dict = {edge_type: graph[edge_type].edge_index 
                              for edge_type in graph.edge_types}
            
            # Forward pass
            logits = self(x_dict, edge_index_dict, raw_features)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            return probs


class EnhancedHybridTransactionModel(nn.Module):
    """
    Enhanced hybrid model that integrates:
    1. Graph-based relationships (merchant, company, industry, and price)
    2. Temporal patterns with company-based grouping
    3. Hyperbolic encoding for hierarchical relationships
    
    This model combines the strengths of graph neural networks for capturing
    relationship structures between transactions and temporal models for capturing
    sequential patterns in transaction data.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 400,
                 num_heads: int = 8, num_graph_layers: int = 2, num_temporal_layers: int = 2, 
                 dropout: float = 0.2, use_hyperbolic: bool = True, use_neural_ode: bool = False,
                 use_text: bool = False, multi_task: bool = True, tax_type_dim: int = 20,
                 company_input_dim: Optional[int] = None, num_relations: int = 5,
                 graph_weight: float = 0.6, temporal_weight: float = 0.4,
                 use_dynamic_weighting: bool = True):
        """
        Initialize the enhanced hybrid transaction model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden features
            output_dim: Dimension of output features (num categories)
            num_heads: Number of attention heads
            num_graph_layers: Number of graph layers
            num_temporal_layers: Number of temporal layers
            dropout: Dropout probability
            use_hyperbolic: Whether to use hyperbolic encoding
            use_neural_ode: Whether to use neural ODE layers
            use_text: Whether to use text processing
            multi_task: Whether to use multi-task learning
            tax_type_dim: Dimension of tax type output
            company_input_dim: Dimension of company input features
            num_relations: Number of edge types in the graph
            graph_weight: Weight for graph component in the ensemble
            temporal_weight: Weight for temporal component in the ensemble
            use_dynamic_weighting: Whether to learn the weights dynamically
        """
        super().__init__()
        
        if not HAS_GRAPH_MODEL:
            raise ImportError("GraphEnhancedTemporalModel is required but not available")
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multi_task = multi_task
        self.tax_type_dim = tax_type_dim
        self.graph_weight = graph_weight
        self.temporal_weight = temporal_weight
        self.use_dynamic_weighting = use_dynamic_weighting
        
        # Normalize weights to sum to 1
        total_weight = graph_weight + temporal_weight
        self.graph_weight = graph_weight / total_weight
        self.temporal_weight = temporal_weight / total_weight
        
        # Graph-enhanced model
        self.graph_model = GraphEnhancedTemporalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_graph_layers=num_graph_layers,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout,
            use_hyperbolic=use_hyperbolic,
            use_neural_ode=use_neural_ode,
            multi_task=multi_task,
            tax_type_output_dim=tax_type_dim,
            company_input_dim=company_input_dim,
            num_relations=num_relations
        )
        
        # Temporal model
        self.temporal_model = HyperTemporalTransactionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout,
            use_hyperbolic=use_hyperbolic,
            use_neural_ode=use_neural_ode,
            use_text_processor=use_text,
            graph_input_dim=hidden_dim,  # Set to hidden_dim since we'll use pre-processed features
            company_input_dim=company_input_dim,
            tax_type_output_dim=tax_type_dim,
            multi_task=multi_task
        )
        
        # Dynamic weighting module (if used)
        if use_dynamic_weighting:
            self.weight_module = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),  # 2 weights: graph and temporal
                nn.Softmax(dim=1)
            )
            
            # Feature extractors for weight calculation
            self.graph_feature_extractor = nn.Linear(output_dim, hidden_dim)
            self.temporal_feature_extractor = nn.Linear(output_dim, hidden_dim)
        
        # Output projections
        self.category_output = nn.Linear(output_dim, output_dim)
        if multi_task:
            self.tax_type_output = nn.Linear(tax_type_dim, tax_type_dim)
        
    def forward(self, x, edge_index, edge_type, edge_attr, seq_features, 
                timestamps, tabular_features, t0, t1, descriptions=None,
                user_features=None, is_new_user=None, company_features=None,
                company_ids=None, batch_size=None, seq_len=None):
        """
        Forward pass through the enhanced hybrid model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]
            edge_attr: Edge attributes [num_edges, 1]
            seq_features: Sequential features [batch_size, seq_len, input_dim]
            timestamps: Timestamps [batch_size, seq_len]
            tabular_features: Tabular features [batch_size, input_dim]
            t0: Start time for ODE integration
            t1: End time for ODE integration
            descriptions: List of transaction descriptions (optional)
            user_features: User features (optional)
            is_new_user: Boolean tensor for new users (optional)
            company_features: Company features (optional)
            company_ids: Company IDs (optional)
            batch_size: Batch size (optional)
            seq_len: Sequence length (optional)
            
        Returns:
            If multi_task=True: Tuple of (category_logits, tax_type_logits)
            If multi_task=False: Category logits
        """
        # Get batch size and sequence length if not provided
        if batch_size is None and seq_features is not None:
            batch_size = seq_features.shape[0]
        if seq_len is None and seq_features is not None:
            seq_len = seq_features.shape[1]
        
        # Check dimensions of tensors for debugging
        print(f"Input shapes - x: {x.shape}, edge_index: {edge_index.shape}, edge_type: {edge_type.shape}")
        print(f"seq_features shape: {seq_features.shape}, timestamps shape: {timestamps.shape}")
        
        # Ensure input dimensions match the model's expectations
        expected_input_dim = self.hidden_dim  # Use the model's configured hidden_dim
        
        # Dynamically adjust x if needed to match expected dimension
        if x.shape[1] != expected_input_dim:
            print(f"Reshaping x from {x.shape} to match expected input dimension {expected_input_dim}")
            # If x has too few dimensions, pad it
            if x.shape[1] < expected_input_dim:
                padding = torch.zeros(x.shape[0], expected_input_dim - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
            # If x has too many dimensions, truncate it
            else:
                x = x[:, :expected_input_dim]
            print(f"New x shape: {x.shape}")
        
        # Handle company features dimension alignment
        if company_features is not None:
            # Print shape information for debugging
            print(f"Company features shape: {company_features.shape}, company_input_dim: {self.hidden_dim}")
            
            # Check if dimensions need to be aligned
            if company_features.shape[1] != self.hidden_dim:
                # Create alignment layer if needed
                if not hasattr(self, 'company_align') or self.company_align.in_features != company_features.shape[1]:
                    print(f"INFO: Aligning company feature dimension from {company_features.shape[1]} to {self.hidden_dim}")
                    self.company_align = nn.Linear(
                        company_features.shape[1],
                        self.hidden_dim
                    ).to(company_features.device)
                
                # Apply alignment
                company_features = self.company_align(company_features)
        
        # Forward pass through graph model
        graph_output = self.graph_model(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            seq_features=seq_features,
            timestamps=timestamps,
            company_features=company_features,  # Aligned company features
            company_ids=company_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            t0=t0,
            t1=t1
        )
        
        # Process tabular features through a projection layer to match the expected dimensions
        graph_input_dim = self.temporal_model.graph_input_dim
        
        # Add projection if the dimensions don't match
        if not hasattr(self, 'tabular_projection') or self.tabular_projection.in_features != tabular_features.size(-1):
            self.tabular_projection = nn.Linear(
                tabular_features.size(-1), 
                graph_input_dim
            ).to(tabular_features.device)
        
        # Project the tabular features to match expected dimensions
        projected_tabular_features = self.tabular_projection(tabular_features)
        
        # Forward pass through temporal model
        temporal_output = self.temporal_model(
            graph_features=projected_tabular_features,
            seq_features=seq_features,
            tabular_features=projected_tabular_features,
            timestamps=timestamps,
            t0=t0,
            t1=t1,
            descriptions=descriptions,
            user_features=user_features,
            is_new_user=is_new_user,
            company_features=company_features,  # Use the aligned company features
            company_ids=company_ids
        )
        
        # Calculate weights if using dynamic weighting
        if self.use_dynamic_weighting:
            # Extract features from outputs for weight calculation
            if self.multi_task:
                graph_cat, _ = graph_output
                temporal_cat, _ = temporal_output
            else:
                graph_cat = graph_output
                temporal_cat = temporal_output
                
            graph_features = self.graph_feature_extractor(graph_cat)
            temporal_features = self.temporal_feature_extractor(temporal_cat)
            
            # Calculate weights
            combined_features = torch.cat([graph_features, temporal_features], dim=1)
            weights = self.weight_module(combined_features)
            
            graph_weight = weights[:, 0].unsqueeze(1)
            temporal_weight = weights[:, 1].unsqueeze(1)
        else:
            # Use fixed weights
            graph_weight = self.graph_weight
            temporal_weight = self.temporal_weight
        
        # Combine the outputs based on weights
        if self.multi_task:
            # Unpack the outputs
            graph_category, graph_tax = graph_output
            temporal_category, temporal_tax = temporal_output
            
            # Weight and combine the outputs
            combined_category = graph_weight * graph_category + temporal_weight * temporal_category
            combined_tax = graph_weight * graph_tax + temporal_weight * temporal_tax
            
            # Apply output projections
            category_logits = self.category_output(combined_category)
            tax_type_logits = self.tax_type_output(combined_tax)
            
            return category_logits, tax_type_logits
        else:
            # Single task - combine directly
            combined_output = graph_weight * graph_output + temporal_weight * temporal_output
            category_logits = self.category_output(combined_output)
            
            return category_logits
    
    # --- Modified Graph Building Function (Handles Missing Data) ---
    
    def build_transaction_relationship_graph(self, df):
        """Builds a transaction relationship graph, handling missing IDs."""
        try:
            # Prioritize company_id, then user_id, then merchant_id for edges.
            if 'company_id' in df.columns and df['company_id'].notna().any():
                ids = df['company_id'].dropna().unique()
                # Create edges based on shared company_id
                edges = []
                for id_ in ids:
                    indices = df[df['company_id'] == id_].index.tolist()
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            edges.append((indices[i], indices[j]))
                            edges.append((indices[j], indices[i]))  # Bidirectional
    
            elif 'user_id' in df.columns and df['user_id'].notna().any():
                ids = df['user_id'].dropna().unique()
                edges = []
                for id_ in ids:
                    indices = df[df['user_id'] == id_].index.tolist()
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            edges.append((indices[i], indices[j]))
                            edges.append((indices[j], indices[i]))
    
            elif 'merchant_id' in df.columns and df['merchant_id'].notna().any():
                ids = df['merchant_id'].dropna().unique()
                edges = []
                for id_ in ids:
                    indices = df[df['merchant_id'] == id_].index.tolist()
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            edges.append((indices[i], indices[j]))
                            edges.append((indices[j], indices[i]))
    
            else:
                # No suitable ID columns found. Return empty graph.
                print("Warning: No suitable ID columns for graph construction. Returning an empty graph.")
                return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)), torch.empty(0, dtype=torch.long)
    
            if not edges: #no edges made
                return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)), torch.empty(0, dtype=torch.long)
    
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
            # Basic edge attributes
            edge_attr = torch.ones(edge_index.shape[1], 1)  # Single feature: edge existence
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)  # Single edge type
    
            return edge_index, edge_attr, edge_type
    
        except Exception as e:
            print(f"Error building graph: {e}. Returning an empty graph.")
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)), torch.empty(0, dtype=torch.long)
    
    def prepare_data_from_dataframe(self, df, batch_size=128, seq_len=5):
        """
        Prepare data for the model from a DataFrame using the FIXED feature schema.
        
        Args:
            df: DataFrame containing transaction data
            batch_size: Size of batch for training (default: 128)
            seq_len: Length of sequences for temporal data (default: 5)
            
        Returns:
            Dictionary with prepared data ready for model input
        """
        if not HAS_GRAPH_PROCESSING:
            # Use our built-in graph builder instead
            print(f"Building transaction relationship graph from {len(df)} records")
            edge_index, edge_attr, edge_type = self.build_transaction_relationship_graph(df)
        else:
            print(f"Building transaction relationship graph using imported function")
            edge_index, edge_attr, edge_type = build_transaction_relationship_graph(df)
        
        # --- 2. Feature Extraction (FIXED SCHEMA - This is the Core) ---
        features = []
    
        # --- 2.a Numerical Features ---
        for col in FEATURE_SCHEMA['numerical']:
            if col == 'log_amount':
                # Handle 'log_amount' (depends on 'amount')
                if 'amount' in df.columns:
                    amount = df['amount'].fillna(0).values  # Fill missing with 0
                    log_amount = np.log1p(np.abs(amount)) * np.sign(amount)
                    features.append(log_amount)
                else:
                    features.append(np.zeros(len(df)))  # Fill with 0 if 'amount' is missing
            elif col in df.columns:
                # Handle the column appropriately based on its name and content
                if col in ['merchant_id', 'user_id']:
                    # For IDs that might be strings, use factorization
                    factorized_values, _ = pd.factorize(df[col])
                    normalized = factorized_values / max(1, factorized_values.max())
                    features.append(normalized)
                else:
                    # For true numeric columns
                    try:
                        values = df[col].fillna(0).values.astype(float)  # Fill missing with 0
                        if values.std() > 0:
                            values = (values - values.mean()) / values.std()
                        features.append(values)
                    except (ValueError, TypeError):
                        # Fallback to factorization for non-numeric columns
                        factorized_values, _ = pd.factorize(df[col])
                        normalized = factorized_values / max(1, factorized_values.max())
                        features.append(normalized)
            else:
                features.append(np.zeros(len(df)))  # Fill with 0 if missing
    
        # --- 2.b Boolean Features ---
        for col in FEATURE_SCHEMA['boolean']:
            if col in df.columns:
                # Convert to float (0.0 and 1.0)
                features.append(df[col].fillna(False).astype(float).values)
            else:
                features.append(np.zeros(len(df)))  # Fill missing with 0.0 (False)
    
        # --- 2.c Extract company_ids first for proper temporal grouping ---
        company_ids_tensor = None  # Initialize for later use
        if 'company_id' in df.columns:
            # Extract company_ids properly for temporal grouping
            factorized_values, _ = pd.factorize(df['company_id'])
            company_ids_tensor = torch.tensor(factorized_values, dtype=torch.long)
            print(f"Extracted company_ids_tensor with shape: {company_ids_tensor.shape}, {len(torch.unique(company_ids_tensor))} unique companies")
        
        # --- 2.d Categorical Features (One-Hot with FIXED CATEGORIES) ---
        for col in FEATURE_SCHEMA['categorical']:
            if col in df.columns:
                # Skip company_id as we already processed it
                if col == 'company_id':
                    continue
                
                # Use pd.Categorical with pre-defined categories
                if col in FIXED_CATEGORIES:
                    # Add a dummy category for values not in our predefined list
                    cat_values = df[col].fillna('UNKNOWN').astype(str).str.upper()
                    cat_values = cat_values.apply(lambda x: x if x in FIXED_CATEGORIES[col] else 'UNKNOWN')
                    
                    # Create categorical with fixed categories
                    cat_series = pd.Categorical(cat_values, categories=FIXED_CATEGORIES[col])
                    dummies = pd.get_dummies(cat_series, prefix=col)
                    
                    # Append all one-hot encoded columns (in the correct order)
                    for cat in FIXED_CATEGORIES[col]:
                        col_name = f"{col}_{cat}"
                        if col_name in dummies.columns:
                            features.append(dummies[col_name].values)
                        else:
                            # Category not present in this batch
                            features.append(np.zeros(len(df)))
                else:
                    # Fallback for any categorical features not in our fixed schema
                    print(f"Warning: {col} found in data but not in FIXED_CATEGORIES. Using dynamic categories.")
                    dummies = pd.get_dummies(df[col].fillna('UNKNOWN'), prefix=col)
                    
                    # Limit to top 10 categories if there are too many
                    if len(dummies.columns) > 10:
                        top_cats = df[col].value_counts().nlargest(10).index
                        keep_cols = [f"{col}_{c}" for c in top_cats]
                        dummies = dummies[keep_cols]
                    
                    for dummy_col in dummies.columns:
                        features.append(dummies[dummy_col].values)
            else:
                # If the column is missing, create all-zero dummies (one for each fixed category)
                if col in FIXED_CATEGORIES:
                    for _ in FIXED_CATEGORIES[col]:
                        features.append(np.zeros(len(df)))
    
        # --- 3. Timestamp Features ---
        timestamps_tensor = None
        
        # Process the first valid timestamp column according to priority
        for ts_col in FEATURE_SCHEMA['timestamp']:
            if ts_col in df.columns and timestamps_tensor is None:
                try:
                    # Convert to datetime and extract features
                    timestamps = pd.to_datetime(df[ts_col], errors='coerce')
                    
                    if timestamps.isna().all():
                        print(f"Warning: All values in {ts_col} are NaT. Trying next column.")
                        continue
                        
                    # Fill NaT with median
                    timestamps = timestamps.fillna(timestamps.median())
                    
                    # Cyclical encoding for month (1-12)
                    month = timestamps.dt.month.values.astype(float)
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    features.append(month_sin)
                    features.append(month_cos)
                    
                    # Cyclical encoding for day of week (0-6)
                    day_of_week = timestamps.dt.dayofweek.values.astype(float)
                    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
                    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
                    features.append(dow_sin)
                    features.append(dow_cos)
                    
                    # Cyclical encoding for hour (if available)
                    if timestamps.dt.hour.max() > 0:
                        hour = timestamps.dt.hour.values.astype(float)
                        hour_sin = np.sin(2 * np.pi * hour / 24)
                        hour_cos = np.cos(2 * np.pi * hour / 24)
                        features.append(hour_sin)
                        features.append(hour_cos)
                    
                    # Store timestamp values for sequence creation
                    timestamps_int = timestamps.astype('int64') // 10**9
                    timestamps_tensor = torch.tensor(timestamps_int.values, dtype=torch.float32)
                    
                    print(f"Successfully processed timestamp features from {ts_col}")
                    break
                except Exception as e:
                    print(f"Error processing {ts_col}: {e}. Trying next column.")
                    continue
    
        # --- 4. Create Feature Matrix ---
        if not features:
            print("Warning: No features extracted. Creating dummy features.")
            feature_matrix = np.zeros((len(df), 1))
        else:
            feature_matrix = np.column_stack(features)  # Stack features horizontally
        
        print(f"Extracted {feature_matrix.shape[1]} features from transaction data")
    
        # --- 5. Padding (After Fixed Feature Extraction) ---
        if feature_matrix.shape[1] < INPUT_FEATURE_DIM:
            print(f"Padding feature matrix from {feature_matrix.shape[1]} to {INPUT_FEATURE_DIM} dimensions")
            padded_matrix = np.zeros((len(df), INPUT_FEATURE_DIM))
            padded_matrix[:, :feature_matrix.shape[1]] = feature_matrix
            feature_matrix = padded_matrix
        elif feature_matrix.shape[1] > INPUT_FEATURE_DIM:  # Also truncate if it's too large
            print(f"Truncating feature matrix from {feature_matrix.shape[1]} to {INPUT_FEATURE_DIM} dimensions")
            feature_matrix = feature_matrix[:, :INPUT_FEATURE_DIM]
    
        node_features = torch.tensor(feature_matrix, dtype=torch.float32)
        
        # Adjust batch size if needed
        batch_size = min(len(df), batch_size)
    
        # --- 6. Timestamp Handling (if not already created) ---
        if timestamps_tensor is None:
            print("Warning: No valid timestamp columns found. Generating synthetic timestamps.")
            # Fallback: Synthetic Timestamps
            if 'company_id' in df.columns:
                # Group by company and generate sequential timestamps within each group
                company_ids = pd.factorize(df['company_id'])[0]
                company_ids_tensor = torch.tensor(company_ids, dtype=torch.long)
                
                timestamps_tensor = torch.zeros(len(df), dtype=torch.float32)
                for company_id in torch.unique(company_ids_tensor):
                    company_indices = (company_ids_tensor == company_id).nonzero(as_tuple=True)[0]
                    timestamps_tensor[company_indices] = torch.arange(len(company_indices), dtype=torch.float32)
            else:
                # Completely synthetic, sequential timestamps
                timestamps_tensor = torch.arange(len(df), dtype=torch.float32)
    
        # --- 7. Create sequence features with aligned company_ids ---
        seq_features = []
        seq_company_ids = []
        
        # Create sequences of features with matching company_ids
        for start_idx in range(0, len(df), seq_len):
            end_idx = min(start_idx + seq_len, len(df))
            seq = node_features[start_idx:end_idx]
    
            # Padding for sequence length (if needed)
            if len(seq) < seq_len:
                padding = torch.zeros((seq_len - len(seq), node_features.shape[1]), dtype=torch.float32)
                seq = torch.cat([seq, padding], dim=0)
    
            seq_features.append(seq)
            
            # Create corresponding company_ids sequence
            if company_ids_tensor is not None:
                # Extract company_ids for this sequence and pad if needed
                company_seq = company_ids_tensor[start_idx:end_idx]
                if len(company_seq) < seq_len:
                    # Pad with last company_id to maintain company identity
                    last_id = company_seq[-1] if len(company_seq) > 0 else 0
                    padding = torch.full((seq_len - len(company_seq),), last_id, dtype=torch.long)
                    company_seq = torch.cat([company_seq, padding], dim=0)
                seq_company_ids.append(company_seq)
        
        # Handle case with no full sequence
        if not seq_features:
            seq_features = [torch.zeros((seq_len, node_features.shape[1]), dtype=torch.float32)]
            if company_ids_tensor is not None:
                seq_company_ids = [torch.zeros(seq_len, dtype=torch.long)]
            
        # Stack sequences
        seq_features = torch.stack(seq_features)
        
        # Stack company_ids sequences if available
        if company_ids_tensor is not None and seq_company_ids:
            seq_company_ids_tensor = torch.stack(seq_company_ids)
            print(f"Created seq_company_ids_tensor with shape: {seq_company_ids_tensor.shape}")
        else:
            seq_company_ids_tensor = None
    
        # --- 8. Sequence Timestamps (Consistent with Padding) ---
        seq_timestamps = []
        for start_idx in range(0, len(df), seq_len):
            end_idx = min(start_idx + seq_len, len(df))
            seq = timestamps_tensor[start_idx:end_idx]
    
            if len(seq) < seq_len:
                padding = torch.zeros((seq_len-len(seq),), dtype=torch.float32)
                seq = torch.cat([seq, padding], dim=0)
            seq_timestamps.append(seq)
            
        # Handle case with no full sequence
        if not seq_timestamps:
            seq_timestamps = [torch.zeros((seq_len,), dtype=torch.float32)]
            
        seq_timestamps = torch.stack(seq_timestamps)
    
        # --- 9. Tabular Features (First `batch_size` rows from node_features) ---
        # Assuming batch_size is the number of sequences
        batch_size = min(batch_size, seq_features.shape[0])
        tabular_features = node_features[:batch_size]
    
        # --- 10. Company Features (If Applicable) ---
        company_features = None
        if 'company_type' in df.columns and 'company_size' in df.columns:
            company_feats = []
            
            # Process company_type
            company_type = df['company_type'].fillna('UNKNOWN').astype(str).str.upper()
            company_type = company_type.apply(lambda x: x if x in FIXED_CATEGORIES['company_type'] else 'UNKNOWN')
            company_type_cat = pd.Categorical(company_type, categories=FIXED_CATEGORIES['company_type'])
            company_type_dummies = pd.get_dummies(company_type_cat, prefix='company_type')
            
            # Process company_size
            company_size = df['company_size'].fillna('UNKNOWN').astype(str).str.upper()
            company_size = company_size.apply(lambda x: x if x in FIXED_CATEGORIES['company_size'] else 'UNKNOWN')
            company_size_cat = pd.Categorical(company_size, categories=FIXED_CATEGORIES['company_size'])
            company_size_dummies = pd.get_dummies(company_size_cat, prefix='company_size')
            
            # Combine all company features
            for cat in FIXED_CATEGORIES['company_type']:
                col_name = f'company_type_{cat}'
                if col_name in company_type_dummies.columns:
                    company_feats.append(company_type_dummies[col_name].values)
                else:
                    company_feats.append(np.zeros(len(df)))
    
            for cat in FIXED_CATEGORIES['company_size']:
                col_name = f'company_size_{cat}'
                if col_name in company_size_dummies.columns:
                    company_feats.append(company_size_dummies[col_name].values)
                else:
                    company_feats.append(np.zeros(len(df)))
    
            if company_feats:
                company_feat_matrix = np.column_stack(company_feats)
                company_features = torch.tensor(company_feat_matrix, dtype=torch.float32)
    
        # --- 11. Return Data Dictionary ---
        data = {
            'x': node_features,  # Match forward method parameter 'x' instead of 'node_features'
            'seq_features': seq_features[:batch_size],  # Limit to actual batch size
            'tabular_features': tabular_features,
            'timestamps': seq_timestamps[:batch_size],  # Limit to actual batch size
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_type': edge_type,
            'company_features': company_features,
            # Use properly formatted sequence of company_ids if available
            'company_ids': seq_company_ids_tensor[:batch_size] if (seq_company_ids_tensor is not None) else company_ids_tensor,
            't0': 0.0,  # Start time for ODE integration
            't1': 1.0,  # End time for ODE integration
            'batch_size': batch_size,
            'seq_len': seq_len
        }
        
        # Log company_ids information for debugging
        if 'company_ids' in data and data['company_ids'] is not None:
            company_ids_shape = data['company_ids'].shape
            print(f"Returning company_ids with shape: {company_ids_shape}")
            if len(company_ids_shape) >= 2:
                print(f"Properly formatted for DynamicContextualTemporal with batch_size={company_ids_shape[0]}, seq_len={company_ids_shape[1]}")
            else:
                print(f"WARNING: company_ids has unexpected shape {company_ids_shape} - DynamicContextualTemporal may use standard approach")
        else:
            print("WARNING: No company_ids available for DynamicContextualTemporal")
            
        return data
        
    def _create_sequence_features(self, df, node_features, batch_size, seq_len):
        """Helper method to create sequence features with improved temporal coherence"""
        # Try to group by company_id for better temporal sequences if available
        if 'company_id' in df.columns:
            print("Creating company-grouped sequence features")
            company_groups = df.groupby('company_id')
            companies = list(company_groups.groups.keys())
            
            # Initialize sequence features tensor
            seq_features = torch.zeros((batch_size, seq_len, node_features.shape[1]))
            
            for i in range(batch_size):
                # Select a company (cycle if needed)
                company_idx = i % len(companies)
                company = companies[company_idx]
                
                # Get indices for this company
                indices = company_groups.get_group(company).index.tolist()
                
                # Sort by timestamp if available
                if any(col in df.columns for col in ['timestamp', 'generated_timestamp', 'books_create_timestamp']):
                    ts_col = next(col for col in ['timestamp', 'generated_timestamp', 'books_create_timestamp'] 
                                 if col in df.columns)
                    try:
                        # Sort indices by timestamp
                        ts_series = pd.to_datetime(df.loc[indices, ts_col], errors='coerce')
                        ts_series = ts_series.fillna(ts_series.median())
                        sorted_indices = [idx for _, idx in sorted(zip(ts_series, indices))]
                        indices = sorted_indices
                    except Exception as e:
                        print(f"Error sorting by timestamp: {e}")
                
                # Take up to seq_len transactions from this company
                valid_len = min(seq_len, len(indices))
                
                # Use available transactions and pad if needed
                for j in range(seq_len):
                    if j < valid_len:
                        seq_features[i, j] = node_features[indices[j]]
                    else:
                        # Padding - use last valid transaction
                        seq_features[i, j] = node_features[indices[valid_len-1]]
        else:
            print("Creating standard sequence features (no company grouping available)")
            # Standard approach - create sequences from consecutive records
            seq_features = torch.zeros((batch_size, seq_len, node_features.shape[1]))
            
            for i in range(batch_size):
                start_idx = min(i * seq_len, max(0, len(df) - seq_len))
                
                for j in range(seq_len):
                    if start_idx + j < len(df):
                        seq_features[i, j] = node_features[start_idx + j]
                    else:
                        # Padding with repeated last transaction
                        last_valid = min(start_idx + max(0, j-1), len(df)-1)
                        seq_features[i, j] = node_features[last_valid]
        
        return seq_features
    
    def _create_timestamps(self, df, batch_size, seq_len):
        """Helper method to create timestamp tensor with improved handling"""
        timestamps_tensor = None
        
        # Try each timestamp column in order of preference
        timestamp_columns = ['generated_timestamp', 'timestamp', 'books_create_timestamp', 
                           'update_timestamp', 'transaction_date']
        
        for ts_col in timestamp_columns:
            if ts_col in df.columns and timestamps_tensor is None:
                try:
                    print(f"Trying to create timestamps from {ts_col}")
                    # Convert to datetime with error handling
                    timestamps = pd.to_datetime(df[ts_col], errors='coerce')
                    
                    # Skip if all timestamps are NaT
                    if timestamps.isna().all():
                        print(f"All values in {ts_col} are NaN after conversion, trying next column")
                        continue
                    
                    # Fill NaT values with median timestamp
                    median_ts = timestamps.median()
                    timestamps = timestamps.fillna(median_ts)
                    
                    # Convert to seconds since epoch for numerical processing
                    timestamps_int = timestamps.astype('int64') // 10**9
                    
                    # Normalize to avoid extreme values
                    min_ts = timestamps_int.min()
                    timestamps_norm = timestamps_int - min_ts
                    
                    # Check if we can group by company_id
                    if 'company_id' in df.columns:
                        timestamps_tensor = self._create_company_grouped_timestamps(
                            df, timestamps_norm, batch_size, seq_len)
                    else:
                        timestamps_tensor = self._create_sequential_timestamps(
                            timestamps_norm, batch_size, seq_len)
                    
                    print(f"Successfully created timestamps tensor with shape {timestamps_tensor.shape}")
                    break
                    
                except Exception as e:
                    print(f"Error processing {ts_col}: {e}")
                    continue
        
        # If all timestamp columns failed or none found, use synthetic timestamps
        if timestamps_tensor is None:
            print("No valid timestamp columns found. Using synthetic timestamps.")
            timestamps_tensor = self._create_synthetic_timestamps(df, batch_size, seq_len)
        
        # Final safety check - replace any NaNs and extreme values
        timestamps_tensor = torch.nan_to_num(timestamps_tensor, nan=0.0, posinf=1e5, neginf=0.0)
        timestamps_tensor = torch.clamp(timestamps_tensor, min=0.0, max=1e5)
        
        return timestamps_tensor
    
    def _create_company_grouped_timestamps(self, df, timestamps_norm, batch_size, seq_len):
        """Create timestamps grouped by company for better business-related patterns"""
        print("Creating company-grouped timestamps")
        company_groups = df.groupby('company_id')
        companies = list(company_groups.groups.keys())
        
        # Determine how many companies we can use
        actual_batch_size = min(batch_size, len(companies))
        print(f"Using {actual_batch_size} companies for temporal sequences")
        
        # Initialize collection of company sequences
        all_company_sequences = []
        max_valid_seq_len = 0
        
        # Process each company
        for company_idx, company in enumerate(companies):
            if company_idx >= actual_batch_size:
                break
                
            # Get company's transactions and their timestamps
            company_df = company_groups.get_group(company)
            company_ts = timestamps_norm[company_df.index].sort_values()
            
            if len(company_ts) > 0:
                # Convert to numpy array
                company_ts_array = company_ts.to_numpy().astype(np.float32)
                
                # Use actual sequence length based on available data
                valid_seq_len = min(seq_len, len(company_ts_array))
                max_valid_seq_len = max(max_valid_seq_len, valid_seq_len)
                
                # Create chronologically ordered sequence
                temp_array = [float(company_ts_array[j]) for j in range(valid_seq_len)]
                all_company_sequences.append(temp_array)
        
        # If no valid sequences were created, return synthetic ones
        if not all_company_sequences:
            return self._create_synthetic_timestamps(df, batch_size, seq_len)
        
        # Ensure all sequences have the same length
        padded_sequences = []
        for seq in all_company_sequences:
            if len(seq) < max_valid_seq_len:
                # Pad with last timestamp + small increment to maintain temporal order
                last_val = seq[-1] if seq else 0.0
                padding = [float(last_val + i + 0.01) for i in range(max_valid_seq_len - len(seq))]
                padded_sequences.append(seq + padding)
            else:
                padded_sequences.append(seq)
        
        # Create final timestamps tensor
        final_sequences = []
        for i in range(batch_size):
            seq_idx = i % len(padded_sequences)
            final_sequences.append(padded_sequences[seq_idx])
        
        # Convert to tensor
        timestamps_tensor = torch.tensor(final_sequences, dtype=torch.float32)
        
        # Add noise to duplicated sequences to avoid exact duplication
        if len(padded_sequences) < batch_size:
            noise = torch.randn_like(timestamps_tensor) * 0.01
            timestamps_tensor = timestamps_tensor + noise
        
        return timestamps_tensor
    
    def _create_sequential_timestamps(self, timestamps_norm, batch_size, seq_len):
        """Create timestamps from sequential transactions (when company grouping isn't available)"""
        # Sort timestamps chronologically
        ts_array = timestamps_norm.to_numpy().astype(np.float32)
        sorted_ts = np.sort(ts_array)
        
        # Calculate how many complete sequences we can make
        num_complete_seqs = max(1, len(sorted_ts) // seq_len)
        
        # Initialize timestamps tensor
        timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
        
        # Fill available complete sequences
        for i in range(min(batch_size, num_complete_seqs)):
            start_idx = i * seq_len
            if start_idx + seq_len <= len(sorted_ts):
                seq_array = sorted_ts[start_idx:start_idx+seq_len]
                timestamps_tensor[i] = torch.tensor([float(x) for x in seq_array], dtype=torch.float32)
        
        # If we need more sequences, duplicate with variations
        if num_complete_seqs < batch_size:
            for i in range(num_complete_seqs, batch_size):
                source_idx = i % num_complete_seqs
                base_seq = timestamps_tensor[source_idx].clone()
                
                # Add small noise to maintain temporal character
                noise = torch.randn_like(base_seq) * 0.01
                timestamps_tensor[i] = base_seq + noise
        
        return timestamps_tensor
    
    def _create_synthetic_timestamps(self, df, batch_size, seq_len):
        """Create synthetic timestamps that mimic realistic business patterns"""
        # Check if we can group by company_id
        if 'company_id' in df.columns:
            print("Creating company-based synthetic timestamps")
            company_groups = df.groupby('company_id')
            companies = list(company_groups.groups.keys())
            
            # Initialize tensor
            timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
            
            # For each batch, use one company's pattern
            for i in range(batch_size):
                # Select a company (cycling if needed)
                company_idx = i % len(companies)
                
                # Create realistic business day pattern
                base_time = i * 24 * 3600  # Different day for each company
                business_start = 9 * 3600  # 9 AM
                
                # First timestamp at random time during business hours
                temp_array = []
                start_time = base_time + business_start + np.random.random() * 8 * 3600
                last_time = start_time
                temp_array.append(float(last_time))
                
                # Create subsequent timestamps with realistic patterns
                for j in range(1, seq_len):
                    # Time between transactions varies by hour of day
                    hour_of_day = (last_time / 3600) % 24
                    
                    if 9 <= hour_of_day < 17:  # Business hours
                        time_gap = np.random.exponential(1800)  # ~30 min
                    elif 17 <= hour_of_day < 20:  # Evening
                        time_gap = np.random.exponential(7200)  # ~2 hours
                    else:  # Overnight - skip to next business day
                        time_gap = (24 - hour_of_day + 9 + np.random.random()) * 3600
                    
                    last_time += time_gap
                    temp_array.append(float(last_time))
                
                # Normalize to avoid large values
                min_val = min(temp_array)
                normalized = [t - min_val for t in temp_array]
                
                timestamps_tensor[i] = torch.tensor(normalized, dtype=torch.float32)
        else:
            print("Creating generic synthetic timestamps")
            # Generic approach - create consistent patterns
            timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
            
            for i in range(batch_size):
                base_time = i * 24 * 3600  # Different day for each sequence
                
                # Create timestamps with realistic daily pattern
                temp_array = []
                for j in range(seq_len):
                    # Distribute throughout the day with some randomness
                    hour = 8 + (j * 2) % 10  # Hours between 8am-6pm
                    minute = np.random.randint(0, 60)
                    second = np.random.randint(0, 60)
                    
                    # Day offset increases every 5 transactions
                    day_offset = j // 5
                    
                    # Calculate timestamp
                    timestamp = base_time + day_offset * 86400 + hour * 3600 + minute * 60 + second
                    temp_array.append(float(timestamp))
                
                # Normalize
                min_val = min(temp_array)
                normalized = [t - min_val for t in temp_array]
                
                timestamps_tensor[i] = torch.tensor(normalized, dtype=torch.float32)
        
        return timestamps_tensor
    
    def _create_company_features(self, df, batch_size):
        """Helper method to extract company-level features if available"""
        company_features = None
        
        # Check if company-specific columns exist
        company_columns = [
            'company_type', 'company_size', 'industry_name', 
            'company_age', 'num_employees', 'annual_revenue'
        ]
        
        available_columns = [col for col in company_columns if col in df.columns]
        
        if available_columns:
            print(f"Creating company features from {len(available_columns)} columns")
            company_feats = []
            
            for col in available_columns:
                if col in ['company_type', 'company_size', 'industry_name']:
                    # Categorical columns - one-hot encode
                    try:
                        # Only use categories that appear frequently
                        value_counts = df[col].value_counts()
                        frequent_cats = value_counts[value_counts >= 3].index.tolist()
                        
                        if frequent_cats:
                            filtered_col = df[col].copy()
                            filtered_col[~filtered_col.isin(frequent_cats)] = 'other'
                            dummies = pd.get_dummies(filtered_col)
                            company_feats.append(dummies.values)
                    except Exception as e:
                        print(f"Error processing company column {col}: {e}")
                else:
                    # Numeric columns - normalize
                    try:
                        values = df[col].values.astype(float)
                        normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
                        company_feats.append(normalized.reshape(-1, 1))
                    except Exception as e:
                        print(f"Error processing numeric company column {col}: {e}")
            
            if company_feats:
                # Combine all features
                company_feat_matrix = np.hstack(company_feats) if len(company_feats) > 1 else company_feats[0]
                
                # Limit to batch size
                batch_company_features = company_feat_matrix[:batch_size]
                
                # Convert to tensor
                company_features = torch.tensor(batch_company_features, dtype=torch.float)
        
        return company_features
    
    def extract_embeddings(self, data):
        """
        Extract embeddings from the graph model.
        
        Args:
            data: Dictionary with prepared data
            
        Returns:
            Node embeddings
        """
        self.eval()
        with torch.no_grad():
            # Extract embeddings from graph model
            if hasattr(self.graph_model, 'extract_embeddings'):
                # Check if we need to project the features to match expected dimensions
                input_dim = data['x'].shape[1]
                hidden_dim = self.hidden_dim
                
                print(f"Preparing to extract embeddings - input shape: {data['x'].shape}")
                
                # Create a projection if needed to match dimensions with the model
                if not hasattr(self, '_extraction_projection') or self._extraction_projection.in_features != input_dim:
                    print(f"Creating projection from {input_dim} to {hidden_dim} dimensions")
                    self._extraction_projection = nn.Linear(input_dim, hidden_dim).to(data['x'].device)
                
                # Project the input features to match model's expected dimensions
                projected_x = self._extraction_projection(data['x'])
                print(f"Projected features shape: {projected_x.shape}")
                
                # Extract embeddings using the projected features
                embeddings = self.graph_model.extract_embeddings(
                    x=projected_x,
                    edge_index=data['edge_index'],
                    edge_type=data['edge_type'],
                    edge_attr=data['edge_attr']
                )
                
                print(f"Successfully extracted embeddings with shape: {embeddings.shape}")
                return embeddings
            else:
                # Fallback for models without explicit embedding extraction
                print("Graph model does not support embedding extraction")
                return None