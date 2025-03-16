import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, JumpingKnowledge
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union

class EnhancedTransactionGNN(torch.nn.Module):
    """
    Enhanced Heterogeneous Graph Neural Network for transaction classification.
    This model includes more advanced GNN techniques like residual connections, 
    jumping knowledge, and attention mechanisms to improve classification 
    performance for 400 transaction categories.
    """
    
    def __init__(self, hidden_channels: int = 128, num_layers: int = 3, 
                 dropout: float = 0.4, conv_type: str = 'sage',
                 heads: int = 2, use_jumping_knowledge: bool = True,
                 use_batch_norm: bool = True, metadata: Optional[Tuple] = None):
        """
        Initialize the EnhancedTransactionGNN model.
        
        Args:
            hidden_channels: Dimension of hidden node features
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat')
            heads: Number of attention heads for GAT
            use_jumping_knowledge: Whether to use jumping knowledge
            use_batch_norm: Whether to use batch normalization
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
        
        # If metadata is not provided, use default for transaction graph
        if metadata is None:
            self.metadata = (
                ['transaction', 'merchant', 'category'],  # Node types
                [('transaction', 'belongs_to', 'merchant'),  # Edge types
                 ('transaction', 'has_category', 'category'),
                 # Add reverse edges for better message passing
                 ('merchant', 'rev_belongs_to', 'transaction'),
                 ('category', 'rev_has_category', 'transaction')]
            )
        else:
            self.metadata = metadata
        
        # Input linear layers for each node type
        self.node_encoders = nn.ModuleDict()
        
        # Convolution layers
        self.convs = nn.ModuleList()
        
        # Batch normalization layers if used
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList()
        
        # Initialize node encoders and convolution layers
        self._init_layers()
        
        # Jumping knowledge if used
        if self.use_jumping_knowledge:
            self.jumping_knowledge = JumpingKnowledge('lstm', hidden_channels, num_layers)
        
        # Output layer for transaction classification
        self.pre_classifier = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 400)  # 400 categories
    
    def _init_layers(self):
        """
        Initialize node encoders and convolution layers.
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
            
            # Add batch normalization if used
            if self.use_batch_norm:
                batch_norm_dict = {}
                for node_type in self.metadata[0]:
                    batch_norm_dict[node_type] = nn.BatchNorm1d(self.hidden_channels)
                self.batch_norms.append(nn.ModuleDict(batch_norm_dict))
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            
        Returns:
            Transaction node embeddings after classification layer
        """
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
            x_transaction = self.jumping_knowledge(self.transaction_xs)
        else:
            x_transaction = x_dict['transaction']
        
        # Apply classifier to transaction nodes
        x_transaction = self.pre_classifier(x_transaction)
        x_transaction = F.relu(x_transaction)
        x_transaction = F.dropout(x_transaction, p=self.dropout, training=self.training)
        return self.classifier(x_transaction)
    
    def predict(self, graph: HeteroData) -> torch.Tensor:
        """
        Make predictions on a heterogeneous graph.
        
        Args:
            graph: PyTorch Geometric HeteroData object
            
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
            logits = self(x_dict, edge_index_dict)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            return probs