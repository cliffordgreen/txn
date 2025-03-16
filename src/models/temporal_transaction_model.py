import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, JumpingKnowledge, RGCNConv, GatedGraphConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

class TemporalTransactionEncoder(nn.Module):
    """
    Temporal transaction encoder that captures sequential patterns in user transactions.
    Uses a combination of recurrent networks and attention mechanisms to model
    transaction sequences and temporal dynamics.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True, 
                 use_attention: bool = True, use_transformer: bool = True):
        """
        Initialize the temporal transaction encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
            use_attention: Whether to use attention mechanism
            use_transformer: Whether to use transformer encoder
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        
        # Initial feature projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Recurrent network for sequence modeling
        # We use GRU as it's more effective for variable length sequences
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust hidden dimension if using bidirectional RNN
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Transformer encoder for capturing global dependencies
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=rnn_output_dim,
                nhead=4,
                dim_feedforward=rnn_output_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=2
            )
        
        # Attention mechanism for weighted aggregation
        if use_attention:
            self.attention_query = nn.Linear(rnn_output_dim, 1)
        
        # Output projection
        self.output_projection = nn.Linear(rnn_output_dim, hidden_dim)
    
    def forward(self, sequences: torch.Tensor, 
                lengths: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the temporal encoder.
        
        Args:
            sequences: Transaction sequences [batch_size, seq_len, input_dim]
            lengths: Sequence lengths [batch_size]
            timestamps: Transaction timestamps [batch_size, seq_len]
            
        Returns:
            Encoded sequences [batch_size, hidden_dim]
        """
        batch_size, max_seq_len, _ = sequences.shape
        
        # Project input features
        x = F.relu(self.input_projection(sequences))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pack sequences for efficient RNN processing
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Apply RNN
        packed_output, hidden = self.rnn(packed_sequences)
        
        # Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply transformer if enabled
        if self.use_transformer:
            # Create padding mask (1 = masked position)
            mask = torch.arange(max_seq_len, device=lengths.device)[None, :] >= lengths[:, None]
            output = self.transformer(output, src_key_padding_mask=mask)
        
        # Apply attention if enabled
        if self.use_attention:
            # Compute attention scores
            attention_scores = self.attention_query(output).squeeze(-1)
            
            # Mask out padding positions
            mask = torch.arange(max_seq_len, device=lengths.device)[None, :] < lengths[:, None]
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
            
            # Normalize scores
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Apply attention weights
            context = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        else:
            # Use last non-padded output as context
            idx = (lengths - 1).view(-1, 1).expand(-1, output.size(-1)).unsqueeze(1)
            context = output.gather(1, idx).squeeze(1)
        
        # Project to output dimension
        output = self.output_projection(context)
        output = F.relu(output)
        
        return output


class TimeEncodedEmbedding(nn.Module):
    """
    Time-encoded embedding layer that encodes absolute and relative time information.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, time_dim: int = 16):
        """
        Initialize the time-encoded embedding layer.
        
        Args:
            num_embeddings: Number of embeddings (vocabulary size)
            embedding_dim: Dimension of embeddings
            time_dim: Dimension of time encoding
        """
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim - time_dim)
        self.time_encoder = nn.Linear(2, time_dim)  # Encode absolute and relative time
    
    def forward(self, indices: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the time-encoded embedding layer.
        
        Args:
            indices: Token indices [batch_size, seq_len]
            timestamps: Timestamps [batch_size, seq_len]
            
        Returns:
            Time-encoded embeddings [batch_size, seq_len, embedding_dim]
        """
        # Get basic embeddings
        embeddings = self.embedding(indices)
        
        # Compute relative timestamps (time since first transaction in sequence)
        batch_size, seq_len = timestamps.shape
        first_timestamp = timestamps[:, 0].unsqueeze(1).expand(-1, seq_len)
        relative_time = timestamps - first_timestamp
        
        # Normalize time features
        abs_time_normalized = timestamps / (3600 * 24 * 365)  # Normalize to years
        rel_time_normalized = relative_time / (3600 * 24 * 30)  # Normalize to months
        
        # Combine time features
        time_features = torch.stack([abs_time_normalized, rel_time_normalized], dim=-1)
        
        # Encode time
        time_encoding = self.time_encoder(time_features)
        time_encoding = F.relu(time_encoding)
        
        # Combine embeddings and time encoding
        time_encoded_embeddings = torch.cat([embeddings, time_encoding], dim=-1)
        
        return time_encoded_embeddings


class TemporalGraphAttention(nn.Module):
    """
    Temporal graph attention layer that combines graph structure with
    temporal information to capture dynamic transaction patterns.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.2):
        """
        Initialize the temporal graph attention layer.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert output_dim % num_heads == 0, "Output dimension must be divisible by number of heads"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        # Linear transformations for queries, keys, values
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        # Temporal attention parameters
        self.temporal_query = nn.Linear(2, num_heads)  # Query based on time differences
        
        # Output projection
        self.output_projection = nn.Linear(output_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_time: torch.Tensor, node_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal graph attention layer.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_time: Edge timestamps [num_edges]
            node_time: Node timestamps [num_nodes]
            
        Returns:
            Updated node features [num_nodes, output_dim]
        """
        num_nodes = x.size(0)
        
        # Linear transformations
        q = self.query(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.key(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.value(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Source and target nodes
        src, dst = edge_index
        
        # Gather features for source and target nodes
        q_dst = q[dst]  # [num_edges, num_heads, head_dim]
        k_src = k[src]  # [num_edges, num_heads, head_dim]
        v_src = v[src]  # [num_edges, num_heads, head_dim]
        
        # Compute time differences
        time_diff = edge_time - node_time[dst]
        time_direction = torch.sign(time_diff).float()
        time_magnitude = torch.log1p(torch.abs(time_diff) / 3600).float()  # Log-scaled hours
        time_features = torch.stack([time_direction, time_magnitude], dim=-1)
        
        # Compute temporal attention weights
        temporal_weights = self.temporal_query(time_features)  # [num_edges, num_heads]
        
        # Compute attention scores
        scale = (self.head_dim) ** -0.5
        scores = (q_dst * k_src).sum(dim=-1) * scale  # [num_edges, num_heads]
        
        # Add temporal weights
        scores = scores + temporal_weights
        
        # Normalize scores (softmax over destination nodes)
        alpha = F.softmax(scores, dim=0)
        alpha = self.dropout(alpha)
        
        # Apply attention weights to values
        out = alpha.unsqueeze(-1) * v_src  # [num_edges, num_heads, head_dim]
        
        # Aggregate messages for each destination node
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, dst, out)
        
        # Reshape output
        out = out.view(num_nodes, -1)
        
        # Apply output projection
        out = self.output_projection(out)
        
        return out


class AdvancedTemporalTransactionGNN(nn.Module):
    """
    Advanced Temporal Transaction GNN that combines graph structure with
    temporal dynamics to capture complex transaction patterns over time.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 400,
                 num_rnn_layers: int = 2, num_gnn_layers: int = 3, dropout: float = 0.3,
                 use_attention: bool = True, bidirectional: bool = True, 
                 gnn_type: str = 'gated'):
        """
        Initialize the advanced temporal transaction GNN.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            output_dim: Dimension of output features (num classes)
            num_rnn_layers: Number of RNN layers
            num_gnn_layers: Number of GNN layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            bidirectional: Whether to use bidirectional RNN
            gnn_type: Type of GNN ('gated', 'rgcn', 'gat')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Temporal encoder for sequential patterns
        self.temporal_encoder = TemporalTransactionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_rnn_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            use_transformer=True
        )
        
        # GNN layers for graph structure
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if gnn_type == 'gated':
                layer = GatedGraphConv(hidden_dim, num_layers=2)
            elif gnn_type == 'rgcn':
                layer = RGCNConv(hidden_dim, hidden_dim, num_relations=3)
            elif gnn_type == 'gat':
                layer = GATConv(hidden_dim, hidden_dim // 4, heads=4)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            self.gnn_layers.append(layer)
        
        # Temporal graph attention layers
        self.temporal_gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            layer = TemporalGraphAttention(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_heads=4,
                dropout=dropout
            )
            self.temporal_gnn_layers.append(layer)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layers
        self.pre_output = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                sequences: torch.Tensor, seq_lengths: torch.Tensor,
                timestamps: torch.Tensor, edge_time: torch.Tensor,
                node_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the advanced temporal transaction GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            sequences: Transaction sequences [batch_size, seq_len, input_dim]
            seq_lengths: Sequence lengths [batch_size]
            timestamps: Sequence timestamps [batch_size, seq_len]
            edge_time: Edge timestamps [num_edges]
            node_time: Node timestamps [num_nodes]
            
        Returns:
            Output logits [num_nodes, output_dim]
        """
        # Project input features
        x = F.relu(self.input_projection(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process sequential data
        seq_features = self.temporal_encoder(sequences, seq_lengths, timestamps)
        
        # Apply GNN layers
        h = x
        for i, (gnn_layer, temporal_layer, bn_layer) in enumerate(
            zip(self.gnn_layers, self.temporal_gnn_layers, self.batch_norms)):
            
            # Apply standard GNN layer
            if self.gnn_type == 'rgcn':
                # For RGCN, we need edge types
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
                h1 = gnn_layer(h, edge_index, edge_type)
            else:
                h1 = gnn_layer(h, edge_index)
            
            # Apply temporal graph attention
            h2 = temporal_layer(h, edge_index, edge_time, node_time)
            
            # Combine results
            h_new = h1 + h2
            
            # Apply batch normalization
            h_new = bn_layer(h_new)
            
            # Add residual connection (except for first layer)
            if i > 0:
                h_new = h_new + h
            
            # Apply non-linearity and dropout
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Update hidden state
            h = h_new
        
        # For transaction nodes, combine GNN features with sequential features
        # Assuming transaction nodes are the first `batch_size` nodes
        batch_size = seq_features.size(0)
        h_transactions = h[:batch_size]
        
        # Concatenate GNN and sequential features
        combined_features = torch.cat([h_transactions, seq_features], dim=-1)
        
        # Apply output layers
        out = self.pre_output(combined_features)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        logits = self.output(out)
        
        return logits


class TemporalTransactionEnsemble(nn.Module):
    """
    Ensemble model that combines multiple temporal transaction models
    for improved performance and robustness.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 400,
                 num_models: int = 3, dropout: float = 0.3):
        """
        Initialize the temporal transaction ensemble.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            output_dim: Dimension of output features (num classes)
            num_models: Number of models in the ensemble
            dropout: Dropout probability
        """
        super().__init__()
        
        self.models = nn.ModuleList()
        
        # Create diverse models for the ensemble
        for i in range(num_models):
            # Vary model hyperparameters for diversity
            model = AdvancedTemporalTransactionGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim + (i * 16),  # Vary hidden size
                output_dim=output_dim,
                num_rnn_layers=1 + i % 3,  # Vary RNN depth
                num_gnn_layers=2 + i % 2,  # Vary GNN depth
                dropout=dropout,
                use_attention=i % 2 == 0,  # Vary attention usage
                bidirectional=i % 2 == 0,  # Vary bidirectionality
                gnn_type=['gated', 'rgcn', 'gat'][i % 3]  # Vary GNN type
            )
            
            self.models.append(model)
        
        # Meta-learner for ensemble aggregation
        self.meta_learner = nn.Sequential(
            nn.Linear(output_dim * num_models, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                sequences: torch.Tensor, seq_lengths: torch.Tensor,
                timestamps: torch.Tensor, edge_time: torch.Tensor,
                node_time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal transaction ensemble.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            sequences: Transaction sequences [batch_size, seq_len, input_dim]
            seq_lengths: Sequence lengths [batch_size]
            timestamps: Sequence timestamps [batch_size, seq_len]
            edge_time: Edge timestamps [num_edges]
            node_time: Node timestamps [num_nodes]
            
        Returns:
            Output logits [num_nodes, output_dim]
        """
        # Get predictions from all models
        all_logits = []
        for model in self.models:
            logits = model(x, edge_index, sequences, seq_lengths, timestamps, edge_time, node_time)
            all_logits.append(logits)
        
        # Concatenate all logits
        concat_logits = torch.cat(all_logits, dim=1)
        
        # Apply meta-learner
        ensemble_logits = self.meta_learner(concat_logits)
        
        return ensemble_logits