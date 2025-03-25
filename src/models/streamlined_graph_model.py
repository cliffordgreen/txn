import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math

# Import PyTorch Geometric dependencies
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv

# Import related models
from src.models.hyper_temporal_model import (
    HyperTemporalTransactionModel,
    DynamicContextualTemporal
)

# Import graph processing utilities
from src.data_processing.transaction_graph import (
    build_transaction_relationship_graph,
    extract_graph_features
)


class RelationAwareGraphLayer(MessagePassing):
    """Graph layer that processes different edge types separately."""
    
    def __init__(self, in_channels, out_channels, num_relations=5, aggr='add', bias=True):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        
        # Create a separate weight matrix for each relation type
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x, edge_index, edge_type, edge_attr=None):
        """Forward pass through the layer."""
        # Normalize edge types
        edge_type = torch.clamp(edge_type, 0, self.num_relations - 1)
        
        # Initialize output
        out = torch.zeros((x.size(0), self.out_channels), device=x.device)
        
        # Process each relation type separately
        for rel in range(self.num_relations):
            # Get edges of this relation type
            mask = (edge_type == rel)
            if not mask.any():
                continue
                
            rel_edge_index = edge_index[:, mask]
            rel_edge_attr = edge_attr[mask] if edge_attr is not None else None
            
            # Transform source nodes with relation-specific weight
            rel_weight = self.weight[rel]
            rel_out = self.propagate(
                rel_edge_index, 
                x=x, 
                weight=rel_weight,
                edge_attr=rel_edge_attr
            )
            
            # Add to output
            out += rel_out
        
        # Add bias
        if self.bias is not None:
            out += self.bias
            
        return out
    
    def message(self, x_j, weight, edge_attr=None):
        """Define the message function."""
        # Transform source node features
        msg = torch.matmul(x_j, weight)
        
        # Weight by edge attributes if available
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            msg = msg * edge_attr
            
        return msg


class GraphEnhancedTemporalModel(nn.Module):
    """
    Enhanced transaction classification model that combines:
    1. Graph-based processing of transaction relationships
    2. Temporal processing with company-based grouping
    3. Hyperbolic transaction encoding
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 400,
                 num_heads: int = 8, num_graph_layers: int = 2, num_temporal_layers: int = 2, 
                 dropout: float = 0.2, use_hyperbolic: bool = True, use_neural_ode: bool = False,
                 multi_task: bool = True, num_relations: int = 5, tax_type_output_dim: int = 20,
                 company_input_dim: Optional[int] = None):
        """Initialize the graph-enhanced temporal model."""
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_hyperbolic = use_hyperbolic
        self.use_neural_ode = use_neural_ode
        self.multi_task = multi_task
        self.num_relations = num_relations
        self.tax_type_output_dim = tax_type_output_dim
        self.company_input_dim = company_input_dim or input_dim
        
        # Input projection with dynamic dimension handling
        self.input_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Company feature projection
        if self.company_input_dim != hidden_dim:
            self.company_projection = nn.Linear(self.company_input_dim, hidden_dim)
        
        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(num_relations, hidden_dim)
        
        # Graph layers
        self.graph_layers = nn.ModuleList([
            RelationAwareGraphLayer(hidden_dim, hidden_dim, num_relations=num_relations)
            for _ in range(num_graph_layers)
        ])
        
        # Graph attention layer
        self.graph_attention = GATConv(
            hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout
        )
        
        # Temporal model
        self.temporal_model = HyperTemporalTransactionModel(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout,
            use_hyperbolic=use_hyperbolic,
            use_neural_ode=use_neural_ode,
            multi_task=False,  # Handle multi-task ourselves
            company_input_dim=hidden_dim
        )
        
        # Feature fusion - combine graph and temporal features
        self.feature_fusion = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.LazyLinear(hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Secondary output for tax type prediction
        if self.multi_task:
            self.tax_output_layer = nn.Sequential(
                nn.LazyLinear(hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, tax_type_output_dim)
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None, seq_features: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None, company_features: Optional[torch.Tensor] = None,
                company_ids: Optional[torch.Tensor] = None, 
                batch_size: Optional[int] = None, seq_len: Optional[int] = None,
                t0: float = 0.0, t1: float = 1.0):
        """Forward pass through the graph-enhanced temporal model."""
        # Input projection
        h = self.input_projection(x)
        
        # Process company features if provided
        if company_features is not None:
            company_h = self.company_projection(company_features) if hasattr(self, 'company_projection') else company_features
        else:
            company_h = None
        
        # Process graph structure
        graph_h = h
        for layer in self.graph_layers:
            graph_h = F.gelu(layer(graph_h, edge_index, edge_type, edge_attr))
            graph_h = F.dropout(graph_h, p=self.dropout, training=self.training)
        
        # Apply graph attention
        graph_h = self.graph_attention(graph_h, edge_index)
        
        # Infer batch_size and seq_len if not provided
        if batch_size is None or seq_len is None:
            if seq_features is not None:
                batch_size, seq_len = seq_features.shape[:2]
            else:
                batch_size, seq_len = 1, graph_h.size(0)
        
        # Prepare input for temporal model
        if seq_features is None and graph_h.dim() == 2:
            temp_input = graph_h.view(batch_size, seq_len, -1)
        else:
            temp_input = seq_features
        
        # Process with temporal model
        temporal_output = self.temporal_model(
            temp_input, temp_input, temp_input,
            timestamps, t0, t1, None,
            company_features=company_h,
            company_ids=company_ids
        )
        
        # Get temporal hidden representation
        if hasattr(self.temporal_model, '_last_pooled'):
            temporal_h = self.temporal_model._last_pooled
        else:
            temporal_h = temp_input.mean(dim=1)
        
        # Reshape graph_h if needed to match batch dimension
        if graph_h.size(0) != batch_size:
            graph_h_pooled = graph_h.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            graph_h_pooled = graph_h
        
        # Ensure dimensions match for fusion
        if hasattr(self, 'graph_projection') or graph_h_pooled.size(1) != self.hidden_dim:
            if not hasattr(self, 'graph_projection'):
                self.graph_projection = nn.Linear(graph_h_pooled.size(1), self.hidden_dim).to(graph_h_pooled.device)
            graph_h_pooled = self.graph_projection(graph_h_pooled)
            
        if hasattr(self, 'temporal_projection') or temporal_h.size(1) != self.hidden_dim:
            if not hasattr(self, 'temporal_projection'):
                self.temporal_projection = nn.Linear(temporal_h.size(1), self.hidden_dim).to(temporal_h.device)
            temporal_h = self.temporal_projection(temporal_h)
        
        # Combine features
        combined_h = torch.cat([graph_h_pooled, temporal_h], dim=1)
        fused_h = self.feature_fusion(combined_h)
        
        # Output projection
        category_logits = self.output_layer(fused_h)
        
        # Secondary output for multi-task learning
        if self.multi_task:
            tax_type_logits = self.tax_output_layer(fused_h)
            return category_logits, tax_type_logits
        else:
            return category_logits
            
    def extract_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                          edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract embeddings from the graph model."""
        self.eval()
        with torch.no_grad():
            # Process input features
            h = self.input_projection(x)
            
            # Process graph structure
            graph_h = h
            for layer in self.graph_layers:
                graph_h = F.gelu(layer(graph_h, edge_index, edge_type, edge_attr))
            
            # Apply graph attention
            if hasattr(self, 'graph_attention'):
                graph_h = self.graph_attention(graph_h, edge_index)
            
            return graph_h


class GraphEnhancedTransactionClassifier:
    """Wrapper for the GraphEnhancedTemporalModel that handles data preparation and training."""
    
    def __init__(self, hidden_dim: int = 256, category_dim: int = 400,
                 tax_type_dim: int = 20, num_heads: int = 8, 
                 num_graph_layers: int = 2, num_temporal_layers: int = 2,
                 dropout: float = 0.2, use_hyperbolic: bool = True,
                 use_neural_ode: bool = False, multi_task: bool = True,
                 num_relations: int = 5, lr: float = 1e-3, 
                 weight_decay: float = 1e-5):
        """Initialize the graph-enhanced transaction classifier."""
        self.hidden_dim = hidden_dim
        self.category_dim = category_dim
        self.tax_type_dim = tax_type_dim
        self.num_heads = num_heads
        self.num_graph_layers = num_graph_layers
        self.num_temporal_layers = num_temporal_layers
        self.dropout = dropout
        self.use_hyperbolic = use_hyperbolic
        self.use_neural_ode = use_neural_ode
        self.multi_task = multi_task
        self.num_relations = num_relations
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, df):
        """Prepare transaction data for the model."""
        # Build transaction relationship graph
        edge_index, edge_attr, edge_type = build_transaction_relationship_graph(df)
        
        # Extract node features
        node_features = extract_graph_features(df)
        
        # Extract timestamps if available
        timestamps = None
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                timestamps = pd.to_datetime(df['timestamp'])
            else:
                timestamps = df['timestamp']
            
            # Convert to seconds since earliest timestamp
            timestamps = (timestamps - timestamps.min()).dt.total_seconds()
            timestamps = torch.tensor(timestamps.values, dtype=torch.float)
        
        # Create sequence features
        seq_features = None
        if timestamps is not None:
            batch_size = len(df)
            seq_len = min(5, batch_size)
            
            if batch_size >= seq_len and node_features.size(0) >= batch_size:
                seq_features = node_features[:batch_size].view(batch_size // seq_len, seq_len, -1)
                timestamps = timestamps[:batch_size].view(batch_size // seq_len, seq_len)
        
        # Extract company features if available
        company_features = None
        company_ids = None
        if 'company_id' in df.columns:
            unique_companies = df['company_id'].unique()
            company_to_idx = {comp: i for i, comp in enumerate(unique_companies)}
            company_ids = torch.tensor(df['company_id'].map(company_to_idx).values, dtype=torch.long)
            
            # Extract company features if available
            company_cols = [col for col in df.columns if col.startswith('company_')]
            if company_cols:
                company_features_list = []
                for col in company_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        values = df[col].values
                        company_features_list.append(torch.tensor(values, dtype=torch.float).unsqueeze(1))
                
                if company_features_list:
                    company_features = torch.cat(company_features_list, dim=1)
        
        # Organize data
        data = {
            'node_features': node_features.to(self.device),
            'edge_index': edge_index.to(self.device),
            'edge_attr': edge_attr.to(self.device),
            'edge_type': edge_type.to(self.device),
            'seq_features': seq_features.to(self.device) if seq_features is not None else None,
            'timestamps': timestamps.to(self.device) if timestamps is not None else None,
            'company_features': company_features.to(self.device) if company_features is not None else None,
            'company_ids': company_ids.to(self.device) if company_ids is not None else None,
            'df': df
        }
        
        return data
    
    def initialize_model(self, input_dim, company_input_dim=None):
        """Initialize the model with the correct input dimensions."""
        self.model = GraphEnhancedTemporalModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.category_dim,
            num_heads=self.num_heads,
            num_graph_layers=self.num_graph_layers,
            num_temporal_layers=self.num_temporal_layers,
            dropout=self.dropout,
            use_hyperbolic=self.use_hyperbolic,
            use_neural_ode=self.use_neural_ode,
            multi_task=self.multi_task,
            num_relations=self.num_relations,
            tax_type_output_dim=self.tax_type_dim,
            company_input_dim=company_input_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
    def train_step(self, data, labels):
        """Perform a single training step."""
        if self.model is None:
            raise ValueError("Model must be initialized before training")
            
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(
            x=data['node_features'],
            edge_index=data['edge_index'],
            edge_type=data['edge_type'],
            edge_attr=data['edge_attr'],
            seq_features=data['seq_features'],
            timestamps=data['timestamps'],
            company_features=data['company_features'],
            company_ids=data['company_ids']
        )
        
        # Compute loss and metrics
        if self.multi_task:
            category_logits, tax_type_logits = output
            
            category_loss = F.cross_entropy(category_logits, labels['category'])
            category_acc = (category_logits.argmax(dim=1) == labels['category']).float().mean().item()
            
            tax_type_loss = F.cross_entropy(tax_type_logits, labels['tax_type']) if 'tax_type' in labels else 0
            tax_type_acc = (tax_type_logits.argmax(dim=1) == labels['tax_type']).float().mean().item() if 'tax_type' in labels else 0
                
            # Combined loss
            loss = 0.7 * category_loss + 0.3 * tax_type_loss
            
            metrics = {
                'loss': loss.item(),
                'category_loss': category_loss.item(),
                'category_acc': category_acc,
                'tax_type_loss': tax_type_loss.item() if isinstance(tax_type_loss, torch.Tensor) else tax_type_loss,
                'tax_type_acc': tax_type_acc
            }
        else:
            # Single task
            category_logits = output
            loss = F.cross_entropy(category_logits, labels['category'])
            category_acc = (category_logits.argmax(dim=1) == labels['category']).float().mean().item()
            
            metrics = {
                'loss': loss.item(),
                'category_acc': category_acc
            }
        
        # Backward and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return metrics
    
    def predict(self, data):
        """Make predictions using the model."""
        if self.model is None:
            raise ValueError("Model must be initialized before prediction")
            
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(
                x=data['node_features'],
                edge_index=data['edge_index'],
                edge_type=data['edge_type'],
                edge_attr=data['edge_attr'],
                seq_features=data['seq_features'],
                timestamps=data['timestamps'],
                company_features=data['company_features'],
                company_ids=data['company_ids']
            )
        
        if self.multi_task:
            category_logits, tax_type_logits = output
            return {
                'category': category_logits,
                'tax_type': tax_type_logits
            }
        else:
            return {
                'category': output
            }
            
    def extract_embeddings(self, data):
        """Extract node embeddings from the model."""
        if self.model is None:
            raise ValueError("Model must be initialized before extracting embeddings")
            
        self.model.eval()
        
        with torch.no_grad():
            embeddings = self.model.extract_embeddings(
                x=data['node_features'],
                edge_index=data['edge_index'],
                edge_type=data['edge_type'],
                edge_attr=data['edge_attr']
            )
                
        return embeddings