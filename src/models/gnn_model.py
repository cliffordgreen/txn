import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union

class TransactionGNN(torch.nn.Module):
    """
    Heterogeneous Graph Neural Network for transaction classification.
    This model processes a graph with transaction, merchant, and category nodes
    to classify transactions into predefined categories.
    """
    
    def __init__(self, hidden_channels: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, conv_type: str = 'sage',
                 metadata: Optional[Tuple] = None):
        """
        Initialize the TransactionGNN model.
        
        Args:
            hidden_channels: Dimension of hidden node features
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat')
            metadata: Graph metadata (node types and edge types)
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        
        # If metadata is not provided, use default for transaction graph
        if metadata is None:
            self.metadata = (
                ['transaction', 'merchant', 'category'],  # Node types
                [('transaction', 'belongs_to', 'merchant'),  # Edge types
                 ('transaction', 'has_category', 'category')]
            )
        else:
            self.metadata = metadata
        
        # Input linear layers for each node type
        self.node_encoders = nn.ModuleDict()
        
        # Convolution layers
        self.convs = nn.ModuleList()
        
        # Initialize node encoders and convolution layers
        self._init_layers()
        
        # Output layer for transaction classification
        self.classifier = nn.Linear(hidden_channels, 400)  # 400 categories
    
    def _init_layers(self):
        """
        Initialize node encoders and convolution layers.
        """
        # Node encoders (linear transformation of input features)
        for node_type in self.metadata[0]:
            self.node_encoders[node_type] = Linear(-1, self.hidden_channels)
        
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
                    conv = GATConv((-1, -1), self.hidden_channels, heads=1)
                else:
                    raise ValueError(f"Unsupported convolution type: {self.conv_type}")
                
                conv_dict[edge_type] = conv
            
            # Create heterogeneous convolution
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
    
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
        
        # Apply graph convolutions
        for conv in self.convs:
            # Apply convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply non-linearity and dropout to each node type
            for node_type in x_dict.keys():
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        # Apply classifier to transaction nodes
        return self.classifier(x_dict['transaction'])
    
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
                              for edge_type in self.metadata[1]}
            
            # Forward pass
            logits = self(x_dict, edge_index_dict)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            return probs