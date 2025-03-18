import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, JumpingKnowledge, RGCNConv, GatedGraphConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import math

# Import modern text processors
from src.models.modern_text_processor import FinBERTProcessor, TransactionLLMProcessor, MultiModalTransactionProcessor

class HyperbolicTransactionEncoder(nn.Module):
    """
    Hyperbolic transaction encoder that models transactions in hyperbolic space,
    which is better suited for hierarchical relationships in transaction data.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, curvature: float = 1.0):
        """
        Initialize the hyperbolic transaction encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            curvature: Curvature of the hyperbolic space
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        
        # Mapping from Euclidean to hyperbolic space
        self.euclidean_to_hyperbolic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Hyperbolic feedforward network
        self.hyperbolic_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def _exponential_map(self, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Maps points from the tangent space to the hyperbolic space using the exponential map.
        
        Args:
            x: Points in tangent space [batch_size, dim]
            c: Curvature of hyperbolic space
            
        Returns:
            Points in hyperbolic space [batch_size, dim]
        """
        # Pre-processing - aggressively handle NaNs and large values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip extreme values before any calculations
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Calculate norm with enhanced numerical stability
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
        
        # Calculate exp map with enhanced stability
        sqrt_c = math.sqrt(c)
        
        # Clamp values for tanh stability - use tighter bounds
        tanh_input = (sqrt_c * norm).clamp(min=-10.0, max=10.0)
        
        # Safely compute the result with careful handling of divisions
        tanh_val = torch.tanh(tanh_input)
        x_normalized = x / norm
        exp_map = tanh_val * x_normalized
        
        # Final safeguard - ensure no NaNs or extreme values
        exp_map = torch.nan_to_num(exp_map, nan=0.0, posinf=1.0, neginf=-1.0)
        exp_map = torch.clamp(exp_map, min=-1.0, max=1.0)
            
        return exp_map
    
    def _logarithmic_map(self, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Maps points from the hyperbolic space to the tangent space using the logarithmic map.
        
        Args:
            x: Points in hyperbolic space [batch_size, dim]
            c: Curvature of hyperbolic space
            
        Returns:
            Points in tangent space [batch_size, dim]
        """
        # Pre-processing - aggressively handle NaNs and large values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip extreme values before any calculations
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Calculate norm with enhanced numerical stability
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10, max=0.9)
        
        # Calculate atanh with enhanced stability
        sqrt_c = math.sqrt(c)
        
        # Use much tighter bounds for atanh stability - well within (-1, 1)
        # atanh becomes unstable as inputs approach ±1
        atanh_input = (sqrt_c * norm).clamp(min=-0.9, max=0.9)
        
        # Safely compute the result with careful handling of divisions
        x_normalized = x / norm
        
        # Use a numerically stable version of atanh
        # tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        # Thus, atanh(y) = 0.5 * log((1 + y) / (1 - y))
        atanh_val = 0.5 * torch.log((1 + atanh_input) / (1 - atanh_input))
        log_map = atanh_val * x_normalized
        
        # Final safeguard - ensure no NaNs or extreme values
        log_map = torch.nan_to_num(log_map, nan=0.0, posinf=1.0, neginf=-1.0)
        log_map = torch.clamp(log_map, min=-10.0, max=10.0)
            
        return log_map
    
    def _mobius_addition(self, x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Adds two points in hyperbolic space using the Möbius addition.
        
        Args:
            x: First point in hyperbolic space [batch_size, dim]
            y: Second point in hyperbolic space [batch_size, dim]
            c: Curvature of hyperbolic space
            
        Returns:
            Sum of points in hyperbolic space [batch_size, dim]
        """
        # Pre-processing - aggressively handle NaNs and large values
        x = torch.nan_to_num(x, nan=0.0, posinf=0.5, neginf=-0.5)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.5, neginf=-0.5)
        
        # Clip extreme values before any calculations
        x = torch.clamp(x, min=-0.9, max=0.9)
        y = torch.clamp(y, min=-0.9, max=0.9)
        
        # Compute dot products with enhanced stability
        xy = torch.sum(x * y, dim=-1, keepdim=True).clamp(min=-0.9, max=0.9)
        x_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(min=1e-10, max=0.9)
        y_sq = torch.sum(y * y, dim=-1, keepdim=True).clamp(min=1e-10, max=0.9)
        
        # Use a more numerically stable approach by breaking down the computation
        term1 = x  # Base term
        term2 = 2 * c * xy * x  # xy term
        term3 = c * y_sq * x    # y_sq term
        term4 = (1 - c * x_sq) * y  # x_sq term
        
        # Numerator as sum of terms (more stable than direct multiplication)
        num = term1 + term2 + term3 + term4
        
        # Compute denominator more carefully
        term5 = 1.0  # Base term
        term6 = 2 * c * xy  # xy term
        term7 = (c**2) * x_sq * y_sq  # Product term, carefully computed
        
        # Denominator as sum of terms
        denom = term5 + term6 + term7
        
        # Ensure denominator is safely away from zero
        denom = denom.clamp(min=1e-6)
        
        # Compute result with careful division
        result = num / denom
        
        # Final safeguard - ensure no NaNs or extreme values
        result = torch.nan_to_num(result, nan=0.0, posinf=0.9, neginf=-0.9)
        result = torch.clamp(result, min=-0.9, max=0.9)
            
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hyperbolic transaction encoder.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Encoded features in Euclidean space [batch_size, hidden_dim]
        """
        # Map input to hyperbolic space
        h = self.euclidean_to_hyperbolic(x)
        h_hyper = self._exponential_map(h, self.curvature)
        
        # Apply feedforward network in tangent space
        h_tangent = self._logarithmic_map(h_hyper, self.curvature)
        h_transformed = self.hyperbolic_ffn(h_tangent)
        
        # Map back to hyperbolic space
        h_hyper_transformed = self._exponential_map(h_transformed, self.curvature)
        
        # Map to Euclidean space for output
        h_output = self._logarithmic_map(h_hyper_transformed, self.curvature)
        output = self.output_projection(h_output)
        
        return output


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module that combines features from different modalities
    (graph, sequence, tabular, company) using cross-attention and gating mechanisms.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        """
        Initialize the multi-modal fusion module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Cross-attention for graph-to-sequence
        self.graph_to_seq_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention for sequence-to-graph
        self.seq_to_graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention for tabular-to-multimodal
        self.tabular_to_multimodal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention for company-to-multimodal
        self.company_to_multimodal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanisms
        self.graph_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.seq_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.tabular_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.company_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # Updated for 4 modalities
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, graph_features: torch.Tensor, seq_features: torch.Tensor, 
                tabular_features: torch.Tensor, company_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-modal fusion module.
        
        Args:
            graph_features: Graph features [batch_size, hidden_dim]
            seq_features: Sequence features [batch_size, hidden_dim]
            tabular_features: Tabular features [batch_size, hidden_dim]
            company_features: Company features [batch_size, hidden_dim] (optional)
            
        Returns:
            Fused features [batch_size, hidden_dim]
        """
        batch_size = graph_features.size(0)
        
        # Reshape for attention
        graph_features_reshaped = graph_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        seq_features_reshaped = seq_features.unsqueeze(1)      # [batch_size, 1, hidden_dim]
        tabular_features_reshaped = tabular_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross-attention: graph to sequence
        graph_to_seq, _ = self.graph_to_seq_attention(
            query=seq_features_reshaped,
            key=graph_features_reshaped,
            value=graph_features_reshaped
        )
        
        # Cross-attention: sequence to graph
        seq_to_graph, _ = self.seq_to_graph_attention(
            query=graph_features_reshaped,
            key=seq_features_reshaped,
            value=seq_features_reshaped
        )
        
        # Combine graph and sequence features
        graph_seq_combined = torch.cat([
            graph_features_reshaped, seq_to_graph
        ], dim=-1)
        
        seq_graph_combined = torch.cat([
            seq_features_reshaped, graph_to_seq
        ], dim=-1)
        
        # Apply gating
        graph_gate_values = self.graph_gate(graph_seq_combined)
        seq_gate_values = self.seq_gate(seq_graph_combined)
        
        gated_graph = graph_features_reshaped * graph_gate_values
        gated_seq = seq_features_reshaped * seq_gate_values
        
        # Combine gated features
        multimodal_features = torch.cat([gated_graph, gated_seq], dim=1)  # [batch_size, 2, hidden_dim]
        
        # Cross-attention: tabular to multimodal
        tabular_to_multimodal, _ = self.tabular_to_multimodal_attention(
            query=multimodal_features,
            key=tabular_features_reshaped,
            value=tabular_features_reshaped
        )
        
        # Combine tabular features
        tabular_combined = torch.cat([
            tabular_features_reshaped,
            tabular_to_multimodal.mean(dim=1, keepdim=True)
        ], dim=-1)
        
        # Apply gating
        tabular_gate_values = self.tabular_gate(tabular_combined)
        gated_tabular = tabular_features_reshaped * tabular_gate_values
        
        # Process company features if provided
        if company_features is not None:
            # Reshape company features
            company_features_reshaped = company_features.unsqueeze(1) if company_features.dim() == 2 else company_features
            
            # Update multimodal features to include existing gated features
            multimodal_with_tabular = torch.cat([gated_graph, gated_seq, gated_tabular], dim=1)
            
            # Cross-attention: company to multimodal
            company_to_multimodal, _ = self.company_to_multimodal_attention(
                query=multimodal_with_tabular,
                key=company_features_reshaped,
                value=company_features_reshaped
            )
            
            # Combine company features
            company_combined = torch.cat([
                company_features_reshaped,
                company_to_multimodal.mean(dim=1, keepdim=True)
            ], dim=-1)
            
            # Apply gating
            company_gate_values = self.company_gate(company_combined)
            gated_company = company_features_reshaped * company_gate_values
            
            # Concatenate all gated features including company features
            all_features = torch.cat([
                gated_graph.squeeze(1),
                gated_seq.squeeze(1),
                gated_tabular.squeeze(1),
                gated_company.squeeze(1)
            ], dim=-1)
        else:
            # If no company features, use zero padding for the 4th modality
            zeros = torch.zeros_like(gated_graph.squeeze(1))
            all_features = torch.cat([
                gated_graph.squeeze(1),
                gated_seq.squeeze(1),
                gated_tabular.squeeze(1),
                zeros
            ], dim=-1)
        
        # Aggressive pre-processing before projection - handle all numerical issues
        all_features = torch.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)
        all_features = torch.clamp(all_features, min=-10.0, max=10.0)
        
        # Apply layer normalization for additional stability
        if not hasattr(self, 'pre_norm'):
            self.pre_norm = nn.LayerNorm(all_features.size(-1)).to(all_features.device)
        all_features = self.pre_norm(all_features)
        
        # Apply output projection with gradient clipping during training
        with torch.set_grad_enabled(self.training):
            if self.training:
                # Apply gradient clipping 
                torch.nn.utils.clip_grad_norm_(self.output_projection.parameters(), max_norm=1.0)
            fused_features = self.output_projection(all_features)
        
        # Final normalization and safeguards
        fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1.0, neginf=-1.0)
        fused_features = torch.clamp(fused_features, min=-10.0, max=10.0)
        
        return fused_features


class DynamicContextualTemporal(nn.Module):
    """
    Dynamic contextual temporal layer that captures evolving temporal patterns
    with varying time scales and contextual dependencies.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
                 num_timescales: int = 3, dropout: float = 0.2):
        """
        Initialize the dynamic contextual temporal layer.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            num_timescales: Number of time scales to model
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_timescales = num_timescales
        
        # Time scale encoders
        self.timescale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim // num_timescales),
                nn.LayerNorm(hidden_dim // num_timescales),
                nn.GELU()
            )
            for _ in range(num_timescales)
        ])
        
        # Time scale attention
        self.timescale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_timescales)
        ])
        
        # Dynamic time scale mixer
        self.time_mixer = nn.Sequential(
            nn.Linear(hidden_dim * num_timescales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_timescales),
            nn.Softmax(dim=-1)
        )
        
        # Contextual encoder
        self.contextual_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor, company_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the dynamic contextual temporal layer.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            timestamps: Timestamps [batch_size, seq_len]
            company_ids: Company IDs for each transaction [batch_size, seq_len] (optional)
            
        Returns:
            Temporally encoded features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Check if company grouping should be applied
        if company_ids is not None:
            print(f"DynamicContextualTemporal received company_ids with shape: {company_ids.shape}")
            print(f"Number of unique company IDs: {len(torch.unique(company_ids))}")
            print(f"Company ID distribution: {[(c.item(), (company_ids == c).sum().item()) for c in torch.unique(company_ids)]}")
            return self._forward_with_company_grouping(x, timestamps, company_ids)
        else:
            print(f"DynamicContextualTemporal: No company IDs provided, using standard approach")
            return self._forward_standard(x, timestamps)
    
    def _forward_standard(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass without company grouping.
        """
        batch_size, seq_len, _ = x.shape
        
        # Create time differences
        timestamps_expanded = timestamps.unsqueeze(2).expand(-1, -1, seq_len)
        timestamps_transposed = timestamps.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Time difference and direction
        time_diff = timestamps_expanded - timestamps_transposed  # [batch_size, seq_len, seq_len]
        time_direction = torch.sign(time_diff).float()
        time_magnitude = torch.log1p(torch.abs(time_diff) / 3600).float()  # Log-scaled hours
        
        # Time features for different time scales
        time_features = torch.stack([time_direction, time_magnitude], dim=-1)  # [batch_size, seq_len, seq_len, 2]
        
        # Process each time scale
        timescale_outputs = []
        for i in range(self.num_timescales):
            # Scale factor for different time scales (hours, days, weeks)
            scale_factor = 10 ** i
            scaled_time_features = time_features.clone()
            scaled_time_features[..., 1] = scaled_time_features[..., 1] / scale_factor
            
            # Encode time features
            encoded_time = self.timescale_encoders[i](scaled_time_features)  # [batch_size, seq_len, seq_len, hidden_dim//num_timescales]
            
            # Reshape for attention
            encoded_time = encoded_time.view(batch_size, seq_len, seq_len, -1)
            
            # Create attention mask from time features
            attention_weights = torch.mean(encoded_time, dim=-1)  # [batch_size, seq_len, seq_len]
            
            # Reshape attention_weights to match batch size and sequence length
            # Make sure attention_weights is properly formatted for attention
            if attention_weights.shape[0] != x.shape[0]:
                # Expand or repeat attention weights to match batch dimension
                attention_weights = attention_weights.repeat(x.shape[0] // attention_weights.shape[0], 1, 1)
                
            # Apply attention (use None for attn_mask if shape issues persist)
            try:
                attended_features, _ = self.timescale_attention[i](
                    query=x,
                    key=x,
                    value=x,
                    attn_mask=attention_weights
                )
            except RuntimeError:
                # Fallback to no mask if shape mismatch
                attended_features, _ = self.timescale_attention[i](
                    query=x,
                    key=x,
                    value=x
                )
            
            timescale_outputs.append(attended_features)
        
        # Combine different time scales
        combined_timescales = torch.cat(timescale_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim * num_timescales]
        
        # Dynamic mixing weights
        mixing_weights = self.time_mixer(combined_timescales)  # [batch_size, seq_len, num_timescales]
        
        # Weight each time scale output
        weighted_outputs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        for i in range(self.num_timescales):
            weighted_outputs += mixing_weights[..., i:i+1] * timescale_outputs[i]
        
        # Apply contextual encoder
        contextual_output = self.contextual_encoder(weighted_outputs)
        
        # Output projection
        output = self.output_projection(contextual_output)
        
        return output
    
    def _forward_with_company_grouping(self, x: torch.Tensor, timestamps: torch.Tensor, company_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with company-based grouping for temporal modeling.
        
        This applies temporal modeling separately to transactions from each company,
        respecting the business entity boundaries.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Handle special case when timestamps has shape [batch_size, 1]
        if timestamps.dim() == 2 and timestamps.size(1) == 1:
            print(f"Detected timestamps with shape {timestamps.shape} - expanding to match seq_len {seq_len}")
            # Expand the timestamps to match the expected sequence length
            expanded_timestamps = timestamps.expand(-1, seq_len)
            timestamps = expanded_timestamps
        
        # Get unique company IDs
        if company_ids.dim() == 1:
            # If company_ids is [batch_size], expand to [batch_size, seq_len]
            company_ids = company_ids.unsqueeze(1).expand(-1, seq_len)
        
        # Flatten for easier processing
        flat_x = x.reshape(-1, x.size(-1))  # [batch_size * seq_len, hidden_dim]
        # Handle 2D timestamps - need to flatten to 1D 
        flat_timestamps = timestamps.reshape(-1)  # [batch_size * seq_len]
        flat_company_ids = company_ids.reshape(-1)  # [batch_size * seq_len]
        
        # Get unique companies
        unique_companies = torch.unique(flat_company_ids)
        num_companies = len(unique_companies)
        
        # Pre-allocate output tensor
        flat_output = torch.zeros_like(flat_x)
        
        # Process each company's transactions separately
        for company_idx in unique_companies:
            # Get mask for this company's transactions
            company_mask = (flat_company_ids == company_idx)
            
            # Skip if no transactions for this company
            if not torch.any(company_mask):
                continue
                
            # Get company data
            company_x = flat_x[company_mask]
            company_timestamps = flat_timestamps[company_mask]
            
            # Need at least 2 transactions for temporal processing
            if len(company_timestamps) < 2:
                # Just apply a simple projection for single transactions
                company_output = self.output_projection(company_x.unsqueeze(0)).squeeze(0)
                flat_output[company_mask] = company_output
                continue
                
            # Sort by timestamp within this company
            sorted_indices = torch.argsort(company_timestamps)
            sorted_x = company_x[sorted_indices]
            sorted_timestamps = company_timestamps[sorted_indices]
            
            # Reshape for temporal processing
            company_seq_len = len(sorted_indices)
            reshaped_x = sorted_x.unsqueeze(0)  # [1, company_seq_len, hidden_dim]
            reshaped_timestamps = sorted_timestamps.unsqueeze(0)  # [1, company_seq_len]
            
            # Create time differences for this company only
            timestamps_expanded = reshaped_timestamps.unsqueeze(2).expand(-1, -1, company_seq_len)
            timestamps_transposed = reshaped_timestamps.unsqueeze(1).expand(-1, company_seq_len, -1)
            
            # Time difference and direction
            time_diff = timestamps_expanded - timestamps_transposed  # [1, company_seq_len, company_seq_len]
            time_direction = torch.sign(time_diff).float()
            time_magnitude = torch.log1p(torch.abs(time_diff) / 3600).float()  # Log-scaled hours
            
            # Time features for different time scales
            time_features = torch.stack([time_direction, time_magnitude], dim=-1)  # [1, company_seq_len, company_seq_len, 2]
            
            # Process each time scale
            timescale_outputs = []
            for i in range(self.num_timescales):
                # Scale factor for different time scales (hours, days, weeks)
                scale_factor = 10 ** i
                scaled_time_features = time_features.clone()
                scaled_time_features[..., 1] = scaled_time_features[..., 1] / scale_factor
                
                # Encode time features
                encoded_time = self.timescale_encoders[i](scaled_time_features)
                
                # Reshape for attention
                encoded_time = encoded_time.view(1, company_seq_len, company_seq_len, -1)
                
                # Create attention mask from time features
                attention_weights = torch.mean(encoded_time, dim=-1)  # [1, company_seq_len, company_seq_len]
                
                # Apply attention
                try:
                    attended_features, _ = self.timescale_attention[i](
                        query=reshaped_x,
                        key=reshaped_x,
                        value=reshaped_x,
                        attn_mask=attention_weights
                    )
                except RuntimeError:
                    # Fallback to no mask if shape mismatch
                    attended_features, _ = self.timescale_attention[i](
                        query=reshaped_x,
                        key=reshaped_x,
                        value=reshaped_x
                    )
                
                timescale_outputs.append(attended_features)
            
            # Combine different time scales
            combined_timescales = torch.cat(timescale_outputs, dim=-1)  # [1, company_seq_len, hidden_dim * num_timescales]
            
            # Dynamic mixing weights
            mixing_weights = self.time_mixer(combined_timescales)  # [1, company_seq_len, num_timescales]
            
            # Weight each time scale output
            weighted_outputs = torch.zeros(1, company_seq_len, self.hidden_dim, device=device)
            for i in range(self.num_timescales):
                weighted_outputs += mixing_weights[..., i:i+1] * timescale_outputs[i]
            
            # Apply contextual encoder - only if we have enough sequence length
            if company_seq_len >= 2:
                contextual_output = self.contextual_encoder(weighted_outputs)
            else:
                contextual_output = weighted_outputs
            
            # Output projection
            company_output = self.output_projection(contextual_output.squeeze(0))
            
            # Unsort to original order
            unsorted_output = torch.zeros_like(company_output)
            for i, idx in enumerate(sorted_indices):
                unsorted_output[idx] = company_output[i]
            
            # Store in the correct position in the overall output
            flat_output[company_mask] = unsorted_output
        
        # Reshape output back to original batch shape
        output = flat_output.view(batch_size, seq_len, -1)
        
        return output


class NeuralODELayer(nn.Module):
    """
    Neural ODE layer that models continuous-time dynamics in transaction data
    using ordinary differential equations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, time_steps: int = 10):
        """
        Initialize the Neural ODE layer.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            time_steps: Number of time steps for numerical integration
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        
        # ODE function (dynamics model)
        self.dynamics = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def _ode_step(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Perform a single ODE integration step using the Euler method.
        
        Args:
            x: Current state [batch_size, input_dim]
            dt: Time step size
            
        Returns:
            Next state [batch_size, input_dim]
        """
        # Compute derivative
        dx = self.dynamics(x)
        
        # Euler step
        x_next = x + dx * dt
        
        return x_next
    
    def forward(self, x: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        """
        Forward pass of the Neural ODE layer.
        
        Args:
            x: Input features [batch_size, input_dim]
            t0: Start time
            t1: End time
            
        Returns:
            Features evolved from t0 to t1 [batch_size, input_dim]
        """
        dt = (t1 - t0) / self.time_steps
        
        # Numerical integration
        for i in range(self.time_steps):
            x = self._ode_step(x, dt)
        
        return x


class CompanyAwareContextLayer(nn.Module):
    """
    Company-aware context layer that enriches transaction representations 
    with company information (business type, size, etc.)
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        """
        Initialize the company-aware context layer.
        
        Args:
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Company contextualization
        self.company_context = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Company-transaction fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, transaction_features: torch.Tensor, company_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the company-aware context layer.
        
        Args:
            transaction_features: Transaction features [batch_size, seq_len, hidden_dim]
            company_features: Company features [batch_size, company_seq_len, hidden_dim]
            
        Returns:
            Company-enriched transaction features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = transaction_features.shape
        
        # Ensure company features have batch dimension
        if company_features.dim() == 2:
            company_features = company_features.unsqueeze(1)
        
        # Apply company context transformer
        enhanced_company = self.company_context(company_features)
        
        # Broadcast company features to match transaction sequence
        if enhanced_company.size(1) == 1:
            company_broadcast = enhanced_company.expand(batch_size, seq_len, hidden_dim)
        else:
            # If multiple company features, use mean
            company_mean = enhanced_company.mean(dim=1, keepdim=True).expand(batch_size, seq_len, hidden_dim)
            company_broadcast = company_mean
        
        # Fuse transaction and company features
        combined = torch.cat([transaction_features, company_broadcast], dim=-1)
        fused = self.fusion_layer(combined)
        
        # Residual connection
        output = transaction_features + fused
        
        return output


class HyperTemporalTransactionModel(nn.Module):
    """
    Hyper-temporal transaction model that combines hyperbolic geometry,
    multi-modal fusion, dynamic temporal modeling, and neural ODEs for
    state-of-the-art transaction classification.
    
    Extended to support dual target prediction (category and tax account type)
    and new transaction data format with user feedback integration.
    
    Now includes support for company features to model business-specific patterns.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 400,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.2,
                 use_hyperbolic: bool = True, use_neural_ode: bool = True,
                 use_text_processor: bool = True, text_processor_type: str = "finbert",
                 text_dim: Optional[int] = None, graph_input_dim: Optional[int] = None,
                 company_input_dim: Optional[int] = None,
                 tax_type_output_dim: Optional[int] = 20, multi_task: bool = True):
        """
        Initialize the hyper-temporal transaction model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden features
            output_dim: Dimension of output features (num categories)
            num_heads: Number of attention heads
            num_layers: Number of model layers
            dropout: Dropout probability
            use_hyperbolic: Whether to use hyperbolic encoding
            use_neural_ode: Whether to use neural ODE layers
            use_text_processor: Whether to use text processing
            text_processor_type: Type of text processor to use ('finbert', 'llm', 'bert', etc.)
            text_dim: Dimension of text features (if different from hidden_dim)
            graph_input_dim: Dimension of graph input (if different from sequence input_dim)
            company_input_dim: Dimension of company input (if different from input_dim)
            tax_type_output_dim: Dimension of secondary output (num tax types)
            multi_task: Whether to use multi-task learning for dual prediction
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_hyperbolic = use_hyperbolic
        self.use_neural_ode = use_neural_ode
        self.use_text_processor = use_text_processor
        self.text_processor_type = text_processor_type
        self.text_dim = text_dim if text_dim is not None else hidden_dim
        self.graph_input_dim = graph_input_dim if graph_input_dim is not None else input_dim
        self.company_input_dim = company_input_dim if company_input_dim is not None else input_dim
        self.tax_type_output_dim = tax_type_output_dim
        self.multi_task = multi_task
        
        # Initialize text processor if enabled
        if use_text_processor:
            if text_processor_type == "finbert":
                self.text_processor = FinBERTProcessor(
                    output_dim=self.text_dim,
                    pooling_strategy="mean",
                    test_mode=False  # Set to False in production for real models
                )
            elif text_processor_type == "llm":
                self.text_processor = TransactionLLMProcessor(
                    output_dim=self.text_dim
                )
            else:
                # Default to multi-modal processor that combines multiple strategies
                self.text_processor = MultiModalTransactionProcessor(
                    text_model=text_processor_type,
                    output_dim=self.text_dim,
                    use_llm=(text_processor_type == "llm"),
                    numerical_dim=input_dim
                )
            
            # Text feature projection
            self.text_projection = nn.Sequential(
                nn.Linear(self.text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # Add extra text-only layer for better representations
            self.text_enhancer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Input projections with flexible dimensions to handle mismatches
        self.graph_projection = nn.Sequential(
            nn.Linear(self.graph_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.sequence_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.tabular_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Company feature projection
        self.company_projection = nn.Sequential(
            nn.Linear(self.company_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Company-aware context layer
        self.company_context_layer = CompanyAwareContextLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads // 2,
            dropout=dropout
        )
        
        # Dimension alignment layer for handling mismatches
        self.dim_alignment = nn.ModuleDict({
            'graph': nn.Linear(self.graph_input_dim, input_dim),
            'tabular': nn.Linear(input_dim, input_dim),
            'company': nn.Linear(self.company_input_dim, input_dim)
        })
        
        # Hyperbolic encoding (for hierarchical transaction relationships)
        if use_hyperbolic:
            self.hyperbolic_encoder = HyperbolicTransactionEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
        
        # Multi-modal fusion - Modified to accept company features
        self.fusion_module = MultiModalFusion(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Dynamic contextual temporal layers
        self.temporal_layers = nn.ModuleList([
            DynamicContextualTemporal(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Neural ODE layers for continuous-time modeling
        if use_neural_ode:
            self.ode_layers = nn.ModuleList([
                NeuralODELayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim * 2
                )
                for _ in range(num_layers)
            ])
        
        # Layer normalization and residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Primary output for category prediction
        self.output_category = nn.Linear(hidden_dim, output_dim)
        
        # Secondary output for tax account type prediction (if using multi-task learning)
        if self.multi_task:
            self.output_tax_type = nn.Linear(hidden_dim, self.tax_type_output_dim)
            
            # Shared task representation
            self.task_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),  # 2 tasks: category and tax type
                nn.Softmax(dim=-1)
            )
        
        # Text-aware fusion layer - combines text embeddings with other features
        if use_text_processor:
            self.text_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # Text-enriched contextualization layer
            self.text_context_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads // 2,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
    
    def forward(self, graph_features: torch.Tensor, seq_features: torch.Tensor,
                tabular_features: torch.Tensor, timestamps: torch.Tensor, 
                t0: float, t1: float, descriptions: List[str] = None,
                auto_align_dims: bool = True, user_features: Optional[torch.Tensor] = None,
                is_new_user: Optional[torch.Tensor] = None, 
                company_features: Optional[torch.Tensor] = None,
                company_ids: Optional[torch.Tensor] = None,
                debug_nans: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        def check_nan(tensor, name):
            if debug_nans and isinstance(tensor, torch.Tensor):
                has_nan = torch.isnan(tensor).any().item()
                if has_nan:
                    print(f"NaN detected in {name}: {torch.isnan(tensor).sum().item()} NaN values")
                    # Try to identify where the NaNs are
                    if tensor.dim() > 1:
                        for i in range(min(3, tensor.shape[0])):  # Just check first few to avoid verbose output
                            if torch.isnan(tensor[i]).any().item():
                                print(f"  NaN in {name}[{i}]")
                    return True
            return False
        
        # Check input tensors for NaNs
        check_nan(graph_features, "graph_features")
        check_nan(seq_features, "seq_features")
        check_nan(tabular_features, "tabular_features")
        check_nan(timestamps, "timestamps")
        if user_features is not None:
            check_nan(user_features, "user_features")
        if company_features is not None:
            check_nan(company_features, "company_features")
        """
        Forward pass of the hyper-temporal transaction model.
        
        Args:
            graph_features: Graph features [batch_size, graph_input_dim] or [batch_size, 1, graph_input_dim]
            seq_features: Sequence features [batch_size, seq_len, input_dim]
            tabular_features: Tabular features [batch_size, input_dim] or [batch_size, 1, input_dim]
            timestamps: Timestamps [batch_size, seq_len]
            t0: Start time for ODE integration
            t1: End time for ODE integration
            descriptions: List of transaction descriptions (optional)
            auto_align_dims: Automatically align feature dimensions if mismatched
            user_features: User features [batch_size, user_dim] (optional)
            is_new_user: Boolean tensor indicating if user is new [batch_size] (optional)
            company_features: Company features [batch_size, company_input_dim] (optional)
            company_ids: Company IDs for grouping transactions in temporal modeling [batch_size] or [batch_size, seq_len] (optional)
                         When provided, temporal modeling will be applied separately to transactions from each company
            
        Returns:
            If multi_task=True: Tuple of (category_logits, tax_type_logits)
                                [batch_size, output_dim], [batch_size, tax_type_output_dim]
            If multi_task=False: Category logits [batch_size, output_dim]
        """
        batch_size, seq_len, seq_feat_dim = seq_features.shape
        
        # Handle potential dimension mismatches
        if auto_align_dims:
            # Ensure graph_features has the right dimensions
            if graph_features.dim() == 2:
                if graph_features.size(1) != seq_feat_dim and graph_features.size(1) == self.graph_input_dim:
                    # Project to match sequence feature dim
                    graph_features = self.dim_alignment['graph'](graph_features)
            elif graph_features.dim() == 3:
                if graph_features.size(2) != seq_feat_dim and graph_features.size(2) == self.graph_input_dim:
                    # Reshape, project, and reshape back
                    batch_orig_shape = graph_features.shape[:2]
                    graph_features = self.dim_alignment['graph'](graph_features.view(-1, self.graph_input_dim))
                    graph_features = graph_features.view(*batch_orig_shape, seq_feat_dim)
            
            # Ensure tabular_features has the right dimensions
            if tabular_features.dim() == 2 and tabular_features.size(1) != seq_feat_dim:
                tabular_features = self.dim_alignment['tabular'](tabular_features)
            elif tabular_features.dim() == 3 and tabular_features.size(2) != seq_feat_dim:
                batch_orig_shape = tabular_features.shape[:2]
                tabular_features = self.dim_alignment['tabular'](tabular_features.view(-1, tabular_features.size(2)))
                tabular_features = tabular_features.view(*batch_orig_shape, seq_feat_dim)
                
            # Ensure company_features has the right dimensions if provided
            if company_features is not None:
                if company_features.dim() == 2 and company_features.size(1) != seq_feat_dim:
                    # Check if alignment layer exists
                    if 'company' not in self.dim_alignment:
                        print(f"Creating company dimension alignment layer: {company_features.size(1)} -> {seq_feat_dim}")
                        self.dim_alignment['company'] = nn.Sequential(
                            nn.Linear(company_features.size(1), seq_feat_dim),
                            nn.LayerNorm(seq_feat_dim)
                        ).to(company_features.device)
                    company_features = self.dim_alignment['company'](company_features)
                elif company_features.dim() == 3 and company_features.size(2) != seq_feat_dim:
                    # Check if alignment layer exists
                    if 'company' not in self.dim_alignment:
                        print(f"Creating company dimension alignment layer: {company_features.size(2)} -> {seq_feat_dim}")
                        self.dim_alignment['company'] = nn.Sequential(
                            nn.Linear(company_features.size(2), seq_feat_dim),
                            nn.LayerNorm(seq_feat_dim)
                        ).to(company_features.device)
                    batch_orig_shape = company_features.shape[:2]
                    company_features = self.dim_alignment['company'](company_features.view(-1, company_features.size(2)))
                    company_features = company_features.view(*batch_orig_shape, seq_feat_dim)
        
        # Process text features if enabled and descriptions are provided
        if self.use_text_processor and descriptions is not None:
            try:
                if isinstance(self.text_processor, nn.Module) and hasattr(self.text_processor, 'forward'):
                    # If it's a nn.Module with a forward method, prepare numerical features
                    numerical_features = tabular_features.view(batch_size, -1) if tabular_features.dim() > 2 else tabular_features
                    text_embeddings = self.text_processor(descriptions, numerical_features)
                else:
                    # Otherwise use the process_batch method
                    text_embeddings = self.text_processor.process_batch(descriptions)
                    
                # Project text embeddings to hidden dimension
                text_features = self.text_projection(text_embeddings)
                
                # Apply additional text enhancement
                text_features = self.text_enhancer(text_features)
            except Exception as e:
                print(f"Error processing text: {e}. Using zero embeddings instead.")
                text_features = torch.zeros(batch_size, self.hidden_dim, device=seq_features.device)
        else:
            # Create dummy text features with zeros if not provided
            text_features = torch.zeros(batch_size, self.hidden_dim, device=seq_features.device)
        
        # Project input features
        if graph_features.dim() == 3:
            graph_h = self.graph_projection(graph_features.reshape(-1, graph_features.size(2)))
            graph_h = graph_h.reshape(batch_size, graph_features.size(1), self.hidden_dim)
        else:
            graph_h = self.graph_projection(graph_features)
            graph_h = graph_h.unsqueeze(1) if graph_h.dim() == 2 else graph_h  # Add sequence dimension if needed
            
        seq_h = self.sequence_projection(seq_features.reshape(-1, seq_feat_dim))
        seq_h = seq_h.reshape(batch_size, seq_len, self.hidden_dim)
        
        if tabular_features.dim() == 3:
            tabular_h = self.tabular_projection(tabular_features.reshape(-1, tabular_features.size(2)))
            tabular_h = tabular_h.reshape(batch_size, tabular_features.size(1), self.hidden_dim)
        else:
            tabular_h = self.tabular_projection(tabular_features)
            tabular_h = tabular_h.unsqueeze(1) if tabular_h.dim() == 2 else tabular_h  # Add sequence dimension if needed
            
        # Process company features if provided
        company_h = None
        if company_features is not None:
            # Debug dimension info
            try:
                if self.training and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                    print(f"Company features shape: {company_features.shape}, company_input_dim: {self.company_input_dim}")
            except:
                # Handle case where distributed is not available
                if self.training:
                    print(f"Company features shape: {company_features.shape}, company_input_dim: {self.company_input_dim}")
                
            if company_features.dim() == 3:
                # Handle 3D company features [batch, seq, feat]
                company_h = self.company_projection(company_features.reshape(-1, company_features.size(2)))
                company_h = company_h.reshape(batch_size, company_features.size(1), self.hidden_dim)
            else:
                # Handle 2D company features [batch, feat] - ensure dimensions match
                if company_features.size(1) != self.company_input_dim:
                    # Handle dimension mismatch - project to correct dimension first
                    print(f"INFO: Aligning company feature dimension from {company_features.size(1)} to {self.company_input_dim}")
                    # Create an adapter dynamically with the correct dimensions
                    adapter_name = f"company_dim_adapter_{company_features.size(1)}"
                    if not hasattr(self, adapter_name):
                        # Create adapter layer on the fly if needed
                        # If dimensions are large, use a more efficient approach with two layers
                        if company_features.size(1) > 1000:
                            setattr(self, adapter_name, nn.Sequential(
                                nn.Linear(company_features.size(1), min(512, company_features.size(1) // 10)),
                                nn.LayerNorm(min(512, company_features.size(1) // 10)),
                                nn.ReLU(),
                                nn.Linear(min(512, company_features.size(1) // 10), self.company_input_dim),
                                nn.LayerNorm(self.company_input_dim)
                            ).to(company_features.device))
                        else:
                            setattr(self, adapter_name, nn.Sequential(
                                nn.Linear(company_features.size(1), self.company_input_dim),
                                nn.LayerNorm(self.company_input_dim)
                            ).to(company_features.device))
                    # Apply dimension adapter
                    adapter = getattr(self, adapter_name)
                    company_features = adapter(company_features)
                
                # Now apply the main projection
                company_h = self.company_projection(company_features)
                company_h = company_h.unsqueeze(1) if company_h.dim() == 2 else company_h  # Add sequence dimension if needed
        
        # Apply hyperbolic encoding if enabled
        if self.use_hyperbolic:
            # Reshape for hyperbolic encoding
            graph_h_flat = graph_h.reshape(-1, self.hidden_dim)
            seq_h_flat = seq_h.reshape(-1, self.hidden_dim)
            tabular_h_flat = tabular_h.reshape(-1, self.hidden_dim)
            
            # Apply hyperbolic encoding
            graph_h_flat = self.hyperbolic_encoder(graph_h_flat)
            seq_h_flat = self.hyperbolic_encoder(seq_h_flat)
            tabular_h_flat = self.hyperbolic_encoder(tabular_h_flat)
            
            # Reshape back
            graph_h = graph_h_flat.reshape(*graph_h.shape)
            seq_h = seq_h_flat.reshape(batch_size, seq_len, self.hidden_dim)
            tabular_h = tabular_h_flat.reshape(*tabular_h.shape)
            
            # Apply hyperbolic encoding to company features if provided
            if company_h is not None:
                company_h_flat = company_h.reshape(-1, self.hidden_dim)
                company_h_flat = self.hyperbolic_encoder(company_h_flat)
                company_h = company_h_flat.reshape(*company_h.shape)
            
            # Also apply to text features if available
            if self.use_text_processor and descriptions is not None:
                text_features = self.hyperbolic_encoder(text_features)
        
        # If text features are available, fuse them with tabular features
        if self.use_text_processor and descriptions is not None:
            # Combine text and tabular features
            tabular_features_for_fusion = tabular_h.view(batch_size, -1, self.hidden_dim)
            combined_tabular = self.text_fusion(
                torch.cat([
                    tabular_features_for_fusion.mean(dim=1), 
                    text_features
                ], dim=1)
            )
            
            # Apply text-enriched contextualization
            combined_tabular = combined_tabular.unsqueeze(1)  # Add sequence dimension
            combined_tabular = self.text_context_layer(combined_tabular)
            
            # Update tabular features with text-enriched features
            # Use broadcasting to enhance all elements of tabular_h with text information
            tabular_h = tabular_h + combined_tabular
        
        # Adapt graph_h to ensure its first dimension matches batch_size
        if graph_h.size(0) != batch_size:
            # If graph_h has a different batch dimension (e.g., from graph pooling)
            graph_h_adapted = graph_h[:batch_size]
        else:
            graph_h_adapted = graph_h
            
        # Ensure proper dimensions for fusion module
        if graph_h_adapted.dim() == 3 and graph_h_adapted.size(1) == 1:
            graph_h_adapted = graph_h_adapted.squeeze(1)
        
        # Fuse multi-modal features
        fused_features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=seq_h.device)
        for i in range(seq_len):
            # Extract tabular features correctly based on their shape
            if tabular_h.size(1) == 1:  # Single tabular feature per batch
                tab_feature = tabular_h.squeeze(1)
            elif tabular_h.size(1) == seq_len:  # Sequence of tabular features
                tab_feature = tabular_h[:, i]
            else:  # Fall back to using the mean
                tab_feature = tabular_h.mean(dim=1)
                
            # Extract company features if available
            company_feat = None
            if company_h is not None:
                if company_h.size(1) == 1:  # Single company feature per batch
                    company_feat = company_h.squeeze(1)
                elif company_h.size(1) == seq_len:  # Sequence of company features
                    company_feat = company_h[:, i]
                else:  # Fall back to using the mean
                    company_feat = company_h.mean(dim=1)
                
            fused_features[:, i] = self.fusion_module(
                graph_h_adapted if graph_h_adapted.dim() == 2 else graph_h_adapted.mean(dim=1),
                seq_h[:, i],
                tab_feature,
                company_feat
            )
        
        # Apply company-aware context layer if company features are available
        if company_h is not None:
            fused_features = self.company_context_layer(
                fused_features,
                company_h.mean(dim=1, keepdim=True) if company_h.dim() > 2 and company_h.size(1) > 1 else company_h
            )
        
        # Apply temporal layers with residual connections
        h = fused_features
        
        # Check if company_ids were explicitly provided in the function call
        if company_ids is None and company_features is not None:
            try:
                # If we have company features, extract company IDs for temporal grouping
                if hasattr(self, 'graph') and self.graph is not None:
                    if hasattr(self.graph, 'company_ids'):
                        # Use explicitly set company_ids from the graph
                        # This is a custom attribute we set in our test
                        company_ids = self.graph.company_ids
                        print(f"Using company_ids from graph with {len(torch.unique(company_ids))} unique companies")
                        
                        # Make sure we have the right number of IDs
                        if company_ids.size(0) != batch_size:
                            # Try to get a subset or expand
                            if batch_size < company_ids.size(0):
                                # Use first batch_size elements
                                company_ids = company_ids[:batch_size]
                                print(f"Taking subset of company_ids to match batch_size {batch_size}")
                            else:
                                # Repeat the IDs to match batch size
                                num_repeats = (batch_size + company_ids.size(0) - 1) // company_ids.size(0)
                                company_ids = company_ids.repeat(num_repeats)[:batch_size]
                                print(f"Expanded company_ids to match batch_size {batch_size}")
                    elif hasattr(self.graph, 'company_id_mapping') and hasattr(self.graph, 'transaction_to_company'):
                        # Use graph-based mappings if available
                        company_ids = self.graph.transaction_to_company
                    elif hasattr(self.graph, 'transaction') and hasattr(self.graph.transaction, 'company_id'):
                        # Use the company_id column if it exists in the graph's transaction data
                        company_ids = self.graph.transaction.company_id
            except Exception as e:
                print(f"Warning: Could not extract company IDs from graph structure: {e}")
            
            # If we still don't have company IDs, generate them from the batch
            if company_ids is None:
                if company_features.size(0) == batch_size:
                    # Don't use batch indices as proxy for company ID, as this treats each transaction as a separate company
                    # Instead, we should leave company_ids as None, which will use standard temporal processing
                    # WITHOUT applying company grouping
                    print(f"No valid company IDs found - using standard temporal processing (no company grouping)")
        
        for i in range(self.num_layers):
            # Apply temporal layer with company grouping if IDs are available
            if company_ids is not None:
                h_temporal = self.temporal_layers[i](h, timestamps, company_ids)
            else:
                h_temporal = self.temporal_layers[i](h, timestamps)
            
            # Apply neural ODE if enabled
            if self.use_neural_ode:
                h_flat = h_temporal.reshape(-1, self.hidden_dim)
                h_ode = self.ode_layers[i](h_flat, t0, t1)
                h_temporal = h_ode.reshape(batch_size, seq_len, self.hidden_dim)
            
            # Add residual connection
            h = h + h_temporal
            
            # Apply layer normalization
            h = self.layer_norms[i](h)
        
        # Global pooling (mean over sequence dimension)
        h_pooled = torch.mean(h, dim=1)
        
        # If text features available, add them to the pooled representation
        if self.use_text_processor and descriptions is not None:
            # Fuse text features with sequence features at the output stage
            h_pooled = h_pooled + 0.2 * text_features  # Use a small weight to avoid dominating
        
        # Process user features if provided
        if user_features is not None:
            # Ensure user_features has the correct shape
            if user_features.dim() == 1:
                user_features = user_features.unsqueeze(1)
                
            # Project to same dimension
            if user_features.size(-1) != self.hidden_dim:
                user_projection = nn.Linear(user_features.size(-1), self.hidden_dim).to(h_pooled.device)
                user_features = user_projection(user_features)
                
            # Add user feature contribution to pooled representation
            if user_features.size(0) != batch_size:
                # User features have a different batch dimension - need to align
                user_indices = torch.arange(batch_size) % user_features.size(0)
                aligned_user_features = user_features[user_indices]
                h_pooled = h_pooled + 0.3 * aligned_user_features.squeeze(1) if aligned_user_features.dim() > 1 else aligned_user_features
            else:
                h_pooled = h_pooled + 0.3 * user_features.squeeze(1) if user_features.dim() > 1 else user_features
        
        # Add is_new_user flag if provided
        if is_new_user is not None:
            # Convert to float and reshape
            if is_new_user.dim() == 1:
                is_new_user = is_new_user.float().unsqueeze(1)
            
            # Create a learnable embedding for the is_new_user flag
            is_new_embedding = nn.Linear(1, self.hidden_dim).to(h_pooled.device)
            new_user_effect = is_new_embedding(is_new_user)
            
            # Add small contribution from is_new_user
            h_pooled = h_pooled + 0.1 * new_user_effect.squeeze(1)
        
        # Add company features contribution to pooled representation if provided
        if company_h is not None:
            # Get a fixed representation of company features
            company_repr = company_h.mean(dim=1) if company_h.dim() > 2 else company_h.squeeze(1)
            
            # Add weighted company feature contribution
            h_pooled = h_pooled + 0.25 * company_repr
        
        # Apply comprehensive stability fixes to pooled hidden state
        h_pooled = torch.nan_to_num(h_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
        h_pooled = torch.clamp(h_pooled, min=-10.0, max=10.0)
        
        # Apply layer normalization for enhanced stability
        if not hasattr(self, 'pooled_norm'):
            self.pooled_norm = nn.LayerNorm(h_pooled.size(-1)).to(h_pooled.device)
        h_pooled_norm = self.pooled_norm(h_pooled)
        
        # Apply pre-output layer with gradient control during training
        with torch.set_grad_enabled(self.training):
            if self.training:
                # Apply gradient clipping to pre_output layer
                torch.nn.utils.clip_grad_norm_(self.pre_output.parameters(), max_norm=0.5)
            
            # Use the normalized hidden state
            h_pre = self.pre_output(h_pooled_norm)
        
        # Comprehensive numerical stability for intermediate representation
        h_pre = torch.nan_to_num(h_pre, nan=0.0, posinf=1.0, neginf=-1.0)
        h_pre = torch.clamp(h_pre, min=-5.0, max=5.0)  # Use more conservative bounds
        
        # Apply second layer normalization before final projection
        if not hasattr(self, 'pre_norm'):
            self.pre_norm = nn.LayerNorm(h_pre.size(-1)).to(h_pre.device)
        h_pre_norm = self.pre_norm(h_pre)
        
        # Apply output layer with gradient control
        with torch.set_grad_enabled(self.training):
            if self.training:
                # Apply gradient clipping to output layer
                torch.nn.utils.clip_grad_norm_(self.output_category.parameters(), max_norm=0.5)
            
            # Generate logits using normalized representation
            category_logits = self.output_category(h_pre_norm)
        
        # Final safety measures for output logits
        category_logits = torch.nan_to_num(category_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        category_logits = torch.clamp(category_logits, min=-20.0, max=20.0)
        
        # Secondary task: tax account type prediction (if multi-task)
        if self.multi_task:
            # Apply task attention with stability measures
            with torch.set_grad_enabled(self.training):
                if self.training:
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.task_attention.parameters(), max_norm=0.5)
                
                # Use the normalized intermediate representation
                task_weights = self.task_attention(h_pre_norm)
            
            # Ensure valid probability distribution with strong safeguards
            task_weights = torch.nan_to_num(task_weights, nan=0.5, posinf=0.5, neginf=0.5)
            
            # Ensure weights sum to 1.0 by applying softmax again
            task_weights = F.softmax(task_weights.clamp(min=-5.0, max=5.0), dim=-1)
            
            # Apply task-specific weighting using normalized intermediate representation
            category_weighted = h_pre_norm * task_weights[:, 0:1]
            tax_type_weighted = h_pre_norm * task_weights[:, 1:2]
            
            # Apply layer normalization for tax type task
            if not hasattr(self, 'tax_norm'):
                self.tax_norm = nn.LayerNorm(tax_type_weighted.size(-1)).to(tax_type_weighted.device)
            tax_type_weighted = self.tax_norm(tax_type_weighted)
            
            # Generate tax type logits with stability measures 
            with torch.set_grad_enabled(self.training):
                if self.training:
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.output_tax_type.parameters(), max_norm=0.5)
                
                tax_type_logits = self.output_tax_type(tax_type_weighted)
            
            # Final safety measures for tax type logits
            tax_type_logits = torch.nan_to_num(tax_type_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            tax_type_logits = torch.clamp(tax_type_logits, min=-20.0, max=20.0)
            
            # Return both outputs
            return category_logits, tax_type_logits
        else:
            # Return only category prediction
            return category_logits


class HyperTemporalEnsemble(nn.Module):
    """
    Ensemble of hyper-temporal transaction models with specialized architectures
    and hyperparameters for optimal performance.
    
    Now supports business entity features like company type and size.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 400,
                 num_models: int = 5, dropout: float = 0.2, use_text_processors: bool = True,
                 company_input_dim: Optional[int] = None):
        """
        Initialize the hyper-temporal ensemble.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Base dimension of hidden features
            output_dim: Dimension of output features (num classes)
            num_models: Number of models in the ensemble
            dropout: Dropout probability
            use_text_processors: Whether to use text processing in ensemble models
            company_input_dim: Dimension of company input features (if different from input_dim)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_models = num_models
        self.use_text_processors = use_text_processors
        self.company_input_dim = company_input_dim if company_input_dim is not None else input_dim
        
        # Create diverse models for the ensemble
        self.models = nn.ModuleList()
        
        # Text processor types for diversity
        text_processor_types = ["finbert", "llm", "bert", "distilbert", "roberta"]
        
        for i in range(num_models):
            # Vary hyperparameters for model diversity
            use_hyperbolic = (i % 2 == 0)
            use_neural_ode = (i % 3 == 0)
            
            # Vary text processor type for diversity
            if use_text_processors:
                text_processor_type = text_processor_types[i % len(text_processor_types)]
                use_text_processor = True
            else:
                text_processor_type = "finbert"  # Default, won't be used
                use_text_processor = False
            
            # Create model with varied architecture
            model = HyperTemporalTransactionModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim + (i * 32),  # Vary hidden size
                output_dim=output_dim,
                num_heads=4 + i % 5,  # Vary number of heads
                num_layers=2 + i % 3,  # Vary depth
                dropout=dropout + (i * 0.05) % 0.2,  # Vary dropout
                use_hyperbolic=use_hyperbolic,
                use_neural_ode=use_neural_ode,
                use_text_processor=use_text_processor,
                text_processor_type=text_processor_type,
                company_input_dim=self.company_input_dim
            )
            
            self.models.append(model)
        
        # Mixture of experts attention
        self.expert_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1)
        )
        
        # Company-aware expert selection
        self.company_expert_selector = nn.Sequential(
            nn.Linear(self.company_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * num_models, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, graph_features: torch.Tensor, seq_features: torch.Tensor,
                tabular_features: torch.Tensor, timestamps: torch.Tensor, 
                t0: float, t1: float, descriptions: List[str] = None,
                user_features: Optional[torch.Tensor] = None,
                is_new_user: Optional[torch.Tensor] = None,
                company_features: Optional[torch.Tensor] = None,
                company_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the hyper-temporal ensemble.
        
        Args:
            graph_features: Graph features [batch_size, input_dim]
            seq_features: Sequence features [batch_size, seq_len, input_dim]
            tabular_features: Tabular features [batch_size, input_dim]
            timestamps: Timestamps [batch_size, seq_len]
            t0: Start time for ODE integration
            t1: End time for ODE integration
            descriptions: List of transaction descriptions (optional)
            user_features: User features [batch_size, user_dim] (optional)
            is_new_user: Boolean tensor indicating if user is new [batch_size] (optional)
            company_features: Company features [batch_size, company_input_dim] (optional)
            company_ids: Company IDs for grouping transactions in temporal modeling [batch_size] or [batch_size, seq_len] (optional)
                         When provided, temporal modeling will be applied separately to transactions from each company
            
        Returns:
            Output logits [batch_size, output_dim]
        """
        batch_size = graph_features.size(0)
        
        # Get predictions from all models
        all_logits = []
        for model in self.models:
            logits = model(
                graph_features, seq_features, tabular_features, 
                timestamps, t0, t1, descriptions,
                user_features=user_features,
                is_new_user=is_new_user,
                company_features=company_features,
                company_ids=company_ids
            )
            all_logits.append(logits)
        
        # Compute expert attention weights based on sequence features
        context_features = torch.mean(seq_features, dim=1)
        expert_weights = self.expert_attention(context_features)  # [batch_size, num_models]
        
        # If company features are available, use them to adjust expert weights
        if company_features is not None:
            # Get company-based expert weights
            if company_features.dim() > 2:
                # Take mean across sequence dimension if it exists
                company_feat = company_features.mean(dim=1) 
            else:
                company_feat = company_features
                
            company_expert_weights = self.company_expert_selector(company_feat)
            
            # Combine sequence-based and company-based expert weights
            expert_weights = 0.7 * expert_weights + 0.3 * company_expert_weights
        
        # Apply expert weights
        weighted_logits = torch.zeros(batch_size, self.output_dim, device=graph_features.device)
        for i, logits in enumerate(all_logits):
            if isinstance(logits, tuple) and len(logits) == 2:
                # If multi-task model returns tuple, use only category logits
                category_logits, _ = logits
                weighted_logits += expert_weights[:, i:i+1] * category_logits
            else:
                weighted_logits += expert_weights[:, i:i+1] * logits
        
        # Also use concatenated approach for the meta-learner
        # First ensure all logits are compatible (handle multi-task case)
        processed_logits = []
        for logits in all_logits:
            if isinstance(logits, tuple) and len(logits) == 2:
                # If multi-task model returns tuple, use only category logits
                category_logits, _ = logits
                processed_logits.append(category_logits)
            else:
                processed_logits.append(logits)
                
        concat_logits = torch.cat(processed_logits, dim=1)  # [batch_size, output_dim * num_models]
        meta_logits = self.output_projection(concat_logits)
        
        # Combine both approaches
        ensemble_logits = weighted_logits + meta_logits
        
        return ensemble_logits