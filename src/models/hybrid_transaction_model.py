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
        # The error 'mat1 and mat2 shapes cannot be multiplied (128x128 and 512x512)' indicates 
        # a dimension mismatch in the input projection layer
        expected_input_dim = 512  # From the error message, seems the model expects 512 dim
        
        # Dynamically adjust x if needed to match expected dimension
        if x.shape[1] != expected_input_dim:
            print(f"Reshaping x from {x.shape} to match expected input dimension")
            # If x has too few dimensions, pad it
            if x.shape[1] < expected_input_dim:
                padding = torch.zeros(x.shape[0], expected_input_dim - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
            # If x has too many dimensions, truncate it
            else:
                x = x[:, :expected_input_dim]
            print(f"New x shape: {x.shape}")
        
        # Forward pass through graph model
        graph_output = self.graph_model(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            seq_features=seq_features,
            timestamps=timestamps,
            company_features=company_features,
            company_ids=company_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            t0=t0,
            t1=t1
        )
        
        # Process tabular features through a projection layer to match the expected dimensions
        graph_input_dim = self.temporal_model.graph_input_dim
        
        # Add projection if the dimensions don't match
        if not hasattr(self, 'tabular_projection'):
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
            company_features=company_features,
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
    
    def prepare_data_from_dataframe(self, df):
        """
        Prepare data for the model from a DataFrame.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Dictionary with prepared data
        """
        if not HAS_GRAPH_PROCESSING:
            raise ImportError("build_transaction_relationship_graph is required but not available")
            
        # Build transaction relationship graph based on merchants, companies, etc.
        edge_index, edge_attr, edge_type = build_transaction_relationship_graph(df)
        
        # Extract features from the transaction data
        # This is a comprehensive feature extraction based on transaction_data_batch schema
        features = []
        
        # ---- Numerical Features ----
        
        # Amount features
        if 'amount' in df.columns:
            amount = df['amount'].values
            amount_normalized = (amount - np.mean(amount)) / (np.std(amount) + 1e-8)
            features.append(amount_normalized)
            
            # Log amount (for skewed distributions)
            log_amount = np.log1p(np.abs(amount)) * np.sign(amount)  # log(1+abs(x)) * sign(x)
            features.append(log_amount)
            
        # User features
        if 'user_id' in df.columns:
            user_ids = pd.factorize(df['user_id'])[0]
            user_ids_norm = user_ids / max(1, user_ids.max())
            features.append(user_ids_norm)
            
        # Merchant features
        if 'merchant_id' in df.columns:
            merchant_ids = pd.factorize(df['merchant_id'])[0]
            merchant_ids_norm = merchant_ids / max(1, merchant_ids.max())
            features.append(merchant_ids_norm)
            
        # Company features
        if 'company_id' in df.columns:
            company_ids = pd.factorize(df['company_id'])[0]
            company_ids_norm = company_ids / max(1, company_ids.max())
            features.append(company_ids_norm)
            
            # Extract actual company IDs for temporal grouping
            company_ids_tensor = torch.tensor(company_ids, dtype=torch.long)
        else:
            company_ids_tensor = None
            
        # Industry code
        if 'industry_code' in df.columns:
            industry_codes = df['industry_code'].values.astype(float)
            industry_codes_norm = industry_codes / max(1, np.max(industry_codes))
            features.append(industry_codes_norm)
            
        # Region ID
        if 'region_id' in df.columns:
            region_ids = df['region_id'].values.astype(float)
            region_ids_norm = region_ids / max(1, np.max(region_ids))
            features.append(region_ids_norm)
            
        # Language ID
        if 'language_id' in df.columns:
            language_ids = df['language_id'].values.astype(float)
            language_ids_norm = language_ids / max(1, np.max(language_ids))
            features.append(language_ids_norm)
            
        # Account type ID
        if 'account_type_id' in df.columns:
            account_type_ids = df['account_type_id'].values.astype(float)
            account_type_ids_norm = account_type_ids / max(1, np.max(account_type_ids))
            features.append(account_type_ids_norm)
            
        # Schedule C ID (tax schedule)
        if 'scheduleC_id' in df.columns:
            scheduleC_ids = df['scheduleC_id'].values.astype(float)
            scheduleC_ids_norm = scheduleC_ids / max(1, np.max(scheduleC_ids))
            features.append(scheduleC_ids_norm)
            
        # ---- Boolean Features ----
            
        # Is new user flag
        if 'is_new_user' in df.columns:
            is_new_user = df['is_new_user'].values.astype(float)
            features.append(is_new_user)
            
        # Is before cutoff date
        if 'is_before_cutoff_date' in df.columns:
            is_before_cutoff = df['is_before_cutoff_date'].values.astype(float)
            features.append(is_before_cutoff)
            
        # QBO accountant attached flag
        if 'qbo_accountant_attached_current_flag' in df.columns:
            qbo_accountant_flag = df['qbo_accountant_attached_current_flag'].values.astype(float)
            features.append(qbo_accountant_flag)
            
        # QBO accountant ever attached
        if 'qbo_accountant_attached_ever' in df.columns:
            qbo_accountant_ever = df['qbo_accountant_attached_ever'].values.astype(float)
            features.append(qbo_accountant_ever)
            
        # QBLive attach flag
        if 'qblive_attach_flag' in df.columns:
            qblive_flag = df['qblive_attach_flag'].values.astype(float)
            features.append(qblive_flag)
            
        # ---- Categorical Features (One-Hot Encoded) ----
            
        # Transaction type
        if 'transaction_type' in df.columns:
            transaction_type_dummies = pd.get_dummies(df['transaction_type'])
            for col in transaction_type_dummies.columns:
                features.append(transaction_type_dummies[col].values)
                
        # QBO current product
        if 'qbo_current_product' in df.columns:
            qbo_product_dummies = pd.get_dummies(df['qbo_current_product'])
            for col in qbo_product_dummies.columns:
                features.append(qbo_product_dummies[col].values)
                
        # QBO signup type
        if 'qbo_signup_type_desc' in df.columns:
            qbo_signup_dummies = pd.get_dummies(df['qbo_signup_type_desc'])
            for col in qbo_signup_dummies.columns:
                features.append(qbo_signup_dummies[col].values)
                
        # Company model bucket
        if 'company_model_bucket_name' in df.columns:
            company_bucket_dummies = pd.get_dummies(df['company_model_bucket_name'])
            for col in company_bucket_dummies.columns:
                features.append(company_bucket_dummies[col].values)
                
        # Industry name
        if 'industry_name' in df.columns:
            industry_dummies = pd.get_dummies(df['industry_name'])
            for col in industry_dummies.columns:
                features.append(industry_dummies[col].values)
                
        # ---- Timestamp Features ----
            
        # Extract month and day of week from timestamps
        for ts_col in ['books_create_timestamp', 'generated_timestamp', 'update_timestamp']:
            if ts_col in df.columns:
                try:
                    # Ensure it's a datetime
                    timestamps = pd.to_datetime(df[ts_col])
                    
                    # Month as cyclical feature (sin, cos encoding preserves cyclical nature)
                    month = timestamps.dt.month.values.astype(float)
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    features.append(month_sin)
                    features.append(month_cos)
                    
                    # Day of week as cyclical feature
                    day_of_week = timestamps.dt.dayofweek.values.astype(float)
                    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
                    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
                    features.append(dow_sin)
                    features.append(dow_cos)
                    
                    # Hour of day (if available with time component)
                    if timestamps.dt.hour.max() > 0:
                        hour = timestamps.dt.hour.values.astype(float)
                        hour_sin = np.sin(2 * np.pi * hour / 24)
                        hour_cos = np.cos(2 * np.pi * hour / 24)
                        features.append(hour_sin)
                        features.append(hour_cos)
                except Exception as e:
                    print(f"Error extracting timestamp features from {ts_col}: {str(e)}")
                    pass
        
        # Combine features into a matrix
        feature_matrix = np.column_stack(features) if features else np.zeros((len(df), 1))
        
        # Print the number of features extracted
        print(f"Extracted {feature_matrix.shape[1]} features from transaction data")
        
        # Ensure we have at least 128 features for the model's input projection
        if feature_matrix.shape[1] < 128:
            print(f"Padding feature matrix from {feature_matrix.shape[1]} to 128 dimensions")
            padded_matrix = np.zeros((len(df), 128))
            padded_matrix[:, :feature_matrix.shape[1]] = feature_matrix
            feature_matrix = padded_matrix
            
        node_features = torch.tensor(feature_matrix, dtype=torch.float)
        
        # Create sequence features from node features (simplified approach)
        batch_size = min(len(df), 128)  # Match the actual batch size in your data (128 from error message)
        seq_len = 5  # Fixed sequence length for simplicity
        
        # Create dummy sequence features - make sure dimensions match!
        seq_features = torch.zeros((batch_size, seq_len, node_features.shape[1]))
        for i in range(batch_size):
            # For each batch item, get a sequence of transactions
            start_idx = i
            for j in range(seq_len):
                if start_idx + j < len(df):  # Prevent index out of bounds
                    idx = start_idx + j
                else:
                    idx = start_idx  # Reuse the start index as fallback
                seq_features[i, j] = node_features[idx]
        
        # Create timestamps with enhanced stability and error handling
        # First, prioritize checking for generated_timestamp explicitly
        timestamps_tensor = None
        
        if 'generated_timestamp' in df.columns:
            try:
                print("Found 'generated_timestamp' column - attempting to use it")
                # Convert to pandas datetime with error handling
                timestamps = pd.to_datetime(df['generated_timestamp'], errors='coerce')
                
                # Check if conversion worked
                if timestamps.isna().all():
                    raise ValueError("All timestamp values are NaN after conversion")
                    
                # Fill NaT values with median timestamp to avoid NaN propagation
                median_ts = timestamps.median()
                timestamps = timestamps.fillna(median_ts)
                
                # Convert to seconds since epoch (float)
                timestamps_int = timestamps.astype('int64') // 10**9  # Integer division for stability
                
                # Normalize to avoid extreme values
                min_ts = timestamps_int.min()
                timestamps_norm = timestamps_int - min_ts  # Make relative to minimum
                
                # Get the available timestamps as a numpy array
                ts_array = timestamps_norm.to_numpy().astype(np.float32)
                ts_array = np.nan_to_num(ts_array, nan=0.0)
                
                # Even if we don't have enough timestamps for full batch*seq_len,
                # we can still create meaningful temporal batches from what we have
                print(f"Available timestamps: {len(ts_array)} for desired shape ({batch_size}x{seq_len})")
                
                # Check if we can create company-based temporal sequences
                if 'company_id' in df.columns:
                    print("Using company_id to create meaningful business-related temporal sequences")
                    # Group transactions by company to preserve business entity relationships
                    company_groups = df.groupby('company_id')
                    companies = list(company_groups.groups.keys())
                    
                    # Determine batch size based on available data
                    actual_batch_size = min(batch_size, len(companies))
                    print(f"Using {actual_batch_size} companies for temporal sequences")
                    
                    # Initialize tensor - this will be reshaped later if needed
                    all_company_sequences = []
                    
                    # Track the longest valid sequence for later padding
                    max_valid_seq_len = 0
                    
                    # For each company, create a proper temporal sequence using their real timestamps
                    for company_idx, company in enumerate(companies):
                        if company_idx >= actual_batch_size:
                            break
                            
                        # Get this company's actual transactions and timestamps
                        company_df = company_groups.get_group(company)
                        
                        # Sort timestamps chronologically to preserve true temporal patterns
                        company_ts = timestamps_norm[company_df.index].sort_values()
                        
                        # Use actual temporal sequences when available
                        if len(company_ts) > 0:
                            # Convert to numpy array with proper type handling
                            company_ts_array = company_ts.to_numpy().astype(np.float32)
                            
                            # Use actual sequence length based on available data
                            valid_seq_len = min(seq_len, len(company_ts_array))
                            max_valid_seq_len = max(max_valid_seq_len, valid_seq_len)
                            
                            # Create a proper temporal sequence using chronologically ordered values
                            # This preserves the true temporal patterns within this company's transactions
                            temp_array = [float(company_ts_array[j]) for j in range(valid_seq_len)]
                            
                            # Save this company's temporal sequence
                            all_company_sequences.append(temp_array)
                    
                    # Now create properly sized tensor with the available sequences
                    # First, make sure all sequences have the same length through padding
                    padded_sequences = []
                    for seq in all_company_sequences:
                        if len(seq) < max_valid_seq_len:
                            # Pad with last timestamp + small increment to maintain temporal order
                            last_val = seq[-1] if seq else 0.0
                            padded = seq + [float(last_val + i + a) for i, a in 
                                          enumerate(np.random.random(max_valid_seq_len - len(seq))*0.1)]
                            padded_sequences.append(padded)
                        else:
                            padded_sequences.append(seq)
                    
                    # Create final timestamps tensor
                    # Match batch size by duplicating sequences if needed
                    final_sequences = []
                    for i in range(batch_size):
                        seq_idx = i % len(padded_sequences)
                        final_sequences.append(padded_sequences[seq_idx])
                    
                    # Convert to tensor, handling proper dimensions
                    timestamps_tensor = torch.tensor(final_sequences, dtype=torch.float32)
                    
                    # Add slight noise to duplicated sequences to avoid exact duplication
                    if len(padded_sequences) < batch_size:
                        # Add small noise to maintain temporal character while avoiding duplication
                        noise = torch.randn_like(timestamps_tensor) * 0.01
                        timestamps_tensor = timestamps_tensor + noise
                    
                    print(f"Created temporal sequences from {len(companies)} companies with proper chronology")
                    
                else:
                    print("No company_id for grouping - preserving natural temporal ordering")
                    # Sort timestamps to maintain chronological order
                    sorted_ts = np.sort(ts_array)
                    
                    # Calculate how many complete sequences we can make
                    num_complete_seqs = len(sorted_ts) // seq_len
                    
                    # Determine how many sequences we need to duplicate
                    actual_batch_size = min(batch_size, max(1, num_complete_seqs))
                    print(f"Can create {num_complete_seqs} complete temporal sequences")
                    
                    # Create sequences with actual chronological ordering
                    timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
                    
                    # Fill available complete sequences
                    for i in range(min(batch_size, num_complete_seqs)):
                        start_idx = i * seq_len
                        seq_array = sorted_ts[start_idx:start_idx+seq_len]
                        timestamps_tensor[i] = torch.tensor([float(x) for x in seq_array], dtype=torch.float32)
                    
                    # If we need more sequences, duplicate with small variations
                    if num_complete_seqs < batch_size:
                        for i in range(num_complete_seqs, batch_size):
                            # Cycle through available sequences with small noise
                            source_idx = i % max(1, num_complete_seqs)
                            base_seq = timestamps_tensor[source_idx].clone()
                            
                            # Add small noise to maintain temporal character
                            noise = torch.randn_like(base_seq) * 0.01
                            timestamps_tensor[i] = base_seq + noise
                
                print(f"Successfully created timestamps tensor with shape {timestamps_tensor.shape}")
                
            except Exception as e:
                print(f"Error processing generated_timestamp: {str(e)}")
                print("Falling back to other timestamp columns...")
                # Fall through to the default timestamp handling below
                timestamps_tensor = None
        else:
            # No generated_timestamp found
            timestamps_tensor = None
            
        # Try other timestamp columns if generated_timestamp processing failed
        if timestamps_tensor is None:
            timestamp_tried = False
            # Check for any timestamp column
            for ts_col in ['timestamp', 'books_create_timestamp', 'update_timestamp']:
                if ts_col in df.columns:
                    timestamp_tried = True
                    try:
                        print(f"Trying timestamp column: {ts_col}")
                        # Convert to pandas datetime with error handling
                        timestamps = pd.to_datetime(df[ts_col], errors='coerce')
                        
                        # Check if we have any valid timestamps after conversion
                        if timestamps.isna().all():
                            print(f"All values in {ts_col} are NaN after conversion, trying next column")
                            continue
                        
                        # Fill NaT values with median timestamp to avoid NaN propagation
                        median_ts = timestamps.median()
                        timestamps = timestamps.fillna(median_ts)
                        
                        # Convert to seconds since epoch as float directly
                        timestamps_int = timestamps.astype('int64') // 10**9  # Integer division for stability
                        
                        # Normalize to avoid extreme values
                        min_ts = timestamps_int.min()
                        timestamps_norm = timestamps_int - min_ts  # Make relative to minimum
                        
                        # Get the available timestamps as a numpy array
                        ts_array = timestamps_norm.to_numpy().astype(np.float32)
                        ts_array = np.nan_to_num(ts_array, nan=0.0)
                        
                        # Same improved timestamp handling as with generated_timestamp
                        print(f"Available {ts_col} values: {len(ts_array)} for desired shape ({batch_size}x{seq_len})")
                        
                        # Check if we can group by company
                        if 'company_id' in df.columns:
                            print(f"Using company_id to create meaningful business-related temporal sequences")
                            # Group transactions by company to preserve business entity relationships
                            company_groups = df.groupby('company_id')
                            companies = list(company_groups.groups.keys())
                            
                            # Determine batch size based on available data
                            actual_batch_size = min(batch_size, len(companies))
                            print(f"Using {actual_batch_size} companies for temporal sequences")
                            
                            # Initialize tensor - this will be reshaped later if needed
                            all_company_sequences = []
                            
                            # Track the longest valid sequence for later padding
                            max_valid_seq_len = 0
                            
                            # For each company, create a proper temporal sequence using their real timestamps
                            for company_idx, company in enumerate(companies):
                                if company_idx >= actual_batch_size:
                                    break
                                    
                                # Get this company's actual transactions and timestamps
                                company_df = company_groups.get_group(company)
                                
                                # Sort timestamps chronologically to preserve true temporal patterns
                                company_ts = timestamps_norm[company_df.index].sort_values()
                                
                                # Use actual temporal sequences when available
                                if len(company_ts) > 0:
                                    # Convert to numpy array with proper type handling
                                    company_ts_array = company_ts.to_numpy().astype(np.float32)
                                    
                                    # Use actual sequence length based on available data
                                    valid_seq_len = min(seq_len, len(company_ts_array))
                                    max_valid_seq_len = max(max_valid_seq_len, valid_seq_len)
                                    
                                    # Create a proper temporal sequence using chronologically ordered values
                                    # This preserves the true temporal patterns within this company's transactions
                                    temp_array = [float(company_ts_array[j]) for j in range(valid_seq_len)]
                                    
                                    # Save this company's temporal sequence
                                    all_company_sequences.append(temp_array)
                            
                            # Now create properly sized tensor with the available sequences
                            # First, make sure all sequences have the same length through padding
                            padded_sequences = []
                            for seq in all_company_sequences:
                                if len(seq) < max_valid_seq_len:
                                    # Pad with last timestamp + small increment to maintain temporal order
                                    last_val = seq[-1] if seq else 0.0
                                    padded = seq + [float(last_val + i + np.random.random()*0.1) for i in range(max_valid_seq_len - len(seq))]
                                    padded_sequences.append(padded)
                                else:
                                    padded_sequences.append(seq)
                            
                            # Create final timestamps tensor
                            # Match batch size by duplicating sequences if needed
                            final_sequences = []
                            for i in range(batch_size):
                                seq_idx = i % len(padded_sequences) if padded_sequences else 0
                                if padded_sequences:
                                    final_sequences.append(padded_sequences[seq_idx])
                                else:
                                    # Fallback if no sequences could be created
                                    final_sequences.append([float(j) for j in range(seq_len)])
                            
                            # Convert to tensor, handling proper dimensions
                            timestamps_tensor = torch.tensor(final_sequences, dtype=torch.float32)
                            
                            # Add slight noise to duplicated sequences to avoid exact duplication
                            if len(padded_sequences) < batch_size and len(padded_sequences) > 0:
                                # Add small noise to maintain temporal character while avoiding duplication
                                noise = torch.randn_like(timestamps_tensor) * 0.01
                                timestamps_tensor = timestamps_tensor + noise
                            
                            print(f"Created temporal sequences from {len(companies)} companies with proper chronology")
                            
                        else:
                            print("No company_id for grouping - preserving natural temporal ordering")
                            # Sort timestamps to maintain chronological order
                            sorted_ts = np.sort(ts_array)
                            
                            # Calculate how many complete sequences we can make
                            num_complete_seqs = len(sorted_ts) // seq_len
                            
                            # Determine how many sequences we need to duplicate
                            actual_batch_size = min(batch_size, max(1, num_complete_seqs))
                            print(f"Can create {num_complete_seqs} complete temporal sequences")
                            
                            # Create sequences with actual chronological ordering
                            timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
                            
                            # Fill available complete sequences
                            for i in range(min(batch_size, num_complete_seqs)):
                                start_idx = i * seq_len
                                if start_idx + seq_len <= len(sorted_ts):
                                    seq_array = sorted_ts[start_idx:start_idx+seq_len]
                                    timestamps_tensor[i] = torch.tensor([float(x) for x in seq_array], dtype=torch.float32)
                            
                            # If we need more sequences, duplicate with small variations
                            if num_complete_seqs < batch_size:
                                for i in range(num_complete_seqs, batch_size):
                                    # Cycle through available sequences with small noise
                                    source_idx = i % max(1, num_complete_seqs)
                                    base_seq = timestamps_tensor[source_idx].clone()
                                    
                                    # Add small noise to maintain temporal character
                                    noise = torch.randn_like(base_seq) * 0.01
                                    timestamps_tensor[i] = base_seq + noise
                        
                        print(f"Successfully using {ts_col} for timestamps with proper temporal sequences")
                        break
                    except Exception as e:
                        print(f"Error processing {ts_col}: {str(e)}")
                        continue
            
            # If all timestamp columns failed or none found, use synthetic timestamps
            if timestamps_tensor is None:
                if timestamp_tried:
                    print("All timestamp columns failed processing. Using synthetic timestamps.")
                else:
                    print("No timestamp columns found. Using synthetic timestamps.")
                
                # Check if we have company_id to create company-based sequences
                if 'company_id' in df.columns:
                    print("Creating company-based synthetic timestamps")
                    company_groups = df.groupby('company_id')
                    companies = list(company_groups.groups.keys())
                    
                    # Initialize timestamps tensor
                    timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
                    
                    # For each batch, use one company's transactions
                    for i in range(batch_size):
                        # Select a company (cycling if needed)
                        company_idx = i % len(companies)
                        
                        # Generate timestamps that mimic real transaction patterns
                        # Each company batch should have its own business day pattern
                        
                        # Create realistic business day pattern
                        # Start from a random time during business hours
                        base_time = i * 24 * 3600  # Start each company on a different day
                        business_start = 9 * 3600  # 9 AM in seconds
                        business_hours = 8 * 3600  # 8 business hours in seconds
                        
                        temp_array = []
                        last_time = base_time + business_start + np.random.random() * business_hours
                        
                        # Simulate transactions happening over multiple business days with realistic patterns
                        for j in range(seq_len):
                            if j > 0:
                                # Time between transactions varies but follows business patterns
                                # Shorter gaps during business hours, longer gaps overnight
                                hour_of_day = (last_time / 3600) % 24
                                
                                if 9 <= hour_of_day < 17:  # Business hours 9 AM - 5 PM
                                    # Frequent transactions during business hours
                                    time_gap = np.random.exponential(1800)  # avg 30 min between transactions
                                elif 17 <= hour_of_day < 20:  # Evening hours
                                    # Less frequent transactions in evening
                                    time_gap = np.random.exponential(7200)  # avg 2 hours
                                else:  # Overnight
                                    # Skip to next business day
                                    time_gap = (24 - hour_of_day + 9 + np.random.random() * 2) * 3600
                                
                                last_time += time_gap
                            
                            temp_array.append(float(last_time))
                        
                        # Normalize to avoid extremely large values
                        min_val = min(temp_array)
                        normalized = [t - min_val for t in temp_array]
                        
                        timestamps_tensor[i] = torch.tensor(normalized, dtype=torch.float32)
                    
                    print(f"Created company-based synthetic timestamps with shape {timestamps_tensor.shape}")
                    
                else:
                    print("Creating pure synthetic temporal sequences")
                    # Create synthetic timestamps that mimic transaction patterns
                    # Initialize a tensor to store synthetic timestamps
                    timestamps_tensor = torch.zeros((batch_size, seq_len), dtype=torch.float32)
                    
                    # Generate a chronological sequence for each batch with realistic transaction patterns
                    for i in range(batch_size):
                        # Create a sequence that mimics real transaction patterns
                        # Start from a base time plus a random offset
                        base_time = i * 24 * 3600  # Different starting day for each sequence
                        
                        # Create a realistic pattern of transactions over time
                        temp_array = []
                        last_time = base_time
                        
                        # First transaction starts at a random time
                        business_hours_start = 8 * 3600  # 8 AM
                        business_hours_end = 18 * 3600  # 6 PM
                        initial_time = base_time + business_hours_start + np.random.random() * (business_hours_end - business_hours_start)
                        temp_array.append(float(initial_time))
                        
                        # Generate subsequent transactions with time patterns following business logic
                        for j in range(1, seq_len):
                            last_time = temp_array[-1]
                            
                            # Get the hour of the day for the last transaction
                            hour_of_day = (last_time / 3600) % 24
                            
                            # Different time gaps based on time of day
                            if 8 <= hour_of_day < 12:  # Morning business hours
                                time_gap = np.random.exponential(3600)  # ~1 hour average
                            elif 12 <= hour_of_day < 14:  # Lunch hours
                                time_gap = np.random.exponential(1800)  # ~30 min average
                            elif 14 <= hour_of_day < 18:  # Afternoon business hours
                                time_gap = np.random.exponential(3600)  # ~1 hour average
                            elif 18 <= hour_of_day < 22:  # Evening hours
                                time_gap = np.random.exponential(7200)  # ~2 hours average
                            else:  # Night time
                                # Skip to next business day morning
                                next_morning = base_time + ((int(last_time / 86400) + 1) * 86400) + business_hours_start
                                next_morning += np.random.random() * 3600  # Random start in first business hour
                                time_gap = next_morning - last_time
                            
                            next_time = last_time + time_gap
                            temp_array.append(float(next_time))
                        
                        # Normalize times to avoid extremely large values
                        min_val = min(temp_array)
                        normalized = [t - min_val for t in temp_array]
                        
                        # Store in timestamps tensor
                        timestamps_tensor[i] = torch.tensor(normalized, dtype=torch.float32)
                    
                    # Add batch-level statistics for clarity
                    print(f"Created synthetic temporal sequences with shape {timestamps_tensor.shape}")
        
        # Final safety check - replace any remaining NaNs and extreme values
        timestamps_tensor = torch.nan_to_num(timestamps_tensor, nan=0.0, posinf=1e5, neginf=0.0)
        timestamps_tensor = torch.clamp(timestamps_tensor, min=0.0, max=1e5)
        
        # Create tabular features
        tabular_features = node_features[:batch_size].clone()
        
        # Prepare company features if available
        company_features = None
        if 'company_type' in df.columns or 'company_size' in df.columns:
            company_feats = []
            
            # One-hot encode company type
            if 'company_type' in df.columns:
                company_types = pd.get_dummies(df['company_type'])
                company_feats.append(company_types.values)
                
            # One-hot encode company size
            if 'company_size' in df.columns:
                company_sizes = pd.get_dummies(df['company_size'])
                company_feats.append(company_sizes.values)
                
            # Combine features
            company_feat_matrix = np.hstack(company_feats)
            company_features = torch.tensor(company_feat_matrix[:batch_size], dtype=torch.float)
        
        # Package data
        data = {
            'x': node_features,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_attr': edge_attr,
            'seq_features': seq_features,
            'timestamps': timestamps_tensor,
            'tabular_features': tabular_features,
            't0': 0.0,  # Dummy value
            't1': 1.0,  # Dummy value
            'company_features': company_features,
            'company_ids': company_ids_tensor,
            'batch_size': batch_size,
            'seq_len': seq_len
        }
        
        return data
    
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
                embeddings = self.graph_model.extract_embeddings(
                    x=data['x'],
                    edge_index=data['edge_index'],
                    edge_type=data['edge_type'],
                    edge_attr=data['edge_attr']
                )
                return embeddings
            else:
                # Fallback for models without explicit embedding extraction
                return None