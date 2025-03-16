import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, JumpingKnowledge, RGCNConv, GatedGraphConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

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