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
        
        # Extract features
        # This is a simplified placeholder - in practice you'd have more sophisticated
        # feature extraction based on your data schema
        features = []
        
        # Amount features
        if 'amount' in df.columns:
            amount = df['amount'].values
            amount_normalized = (amount - np.mean(amount)) / (np.std(amount) + 1e-8)
            features.append(amount_normalized)
        
        # Merchant features if available
        if 'merchant_id' in df.columns:
            # Convert to categorical indices
            merchant_ids = pd.factorize(df['merchant_id'])[0]
            # Normalize to [0, 1] range
            merchant_ids_norm = merchant_ids / max(1, merchant_ids.max())
            features.append(merchant_ids_norm)
        
        # Company features if available
        if 'company_id' in df.columns:
            company_ids = pd.factorize(df['company_id'])[0]
            company_ids_norm = company_ids / max(1, company_ids.max())
            features.append(company_ids_norm)
            
            # Extract actual company IDs for temporal grouping
            company_ids_tensor = torch.tensor(company_ids, dtype=torch.long)
        else:
            company_ids_tensor = None
        
        # Combine features into a matrix
        feature_matrix = np.column_stack(features) if features else np.zeros((len(df), 1))
        node_features = torch.tensor(feature_matrix, dtype=torch.float)
        
        # Create sequence features from node features (simplified approach)
        batch_size = min(100, len(df))  # Limit batch size
        seq_len = 5  # Fixed sequence length for simplicity
        
        # Create dummy sequence features
        seq_features = torch.zeros((batch_size, seq_len, node_features.shape[1]))
        for i in range(batch_size):
            # For each batch item, get a sequence of transactions
            start_idx = i
            for j in range(seq_len):
                idx = (start_idx + j) % len(df)
                seq_features[i, j] = node_features[idx]
        
        # Create timestamps (simplified)
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            if not isinstance(timestamps[0], (int, float)):
                # Convert to seconds if datetime
                timestamps = pd.to_datetime(timestamps).astype(int) / 10**9
            timestamps_tensor = torch.tensor(timestamps[:batch_size * seq_len], dtype=torch.float)
            timestamps_tensor = timestamps_tensor.view(batch_size, seq_len)
        else:
            # Create dummy timestamps
            timestamps_tensor = torch.arange(batch_size * seq_len, dtype=torch.float).view(batch_size, seq_len)
        
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