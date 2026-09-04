"""
LSTM–GNN–Attention 多模态混合预测模型
论文第五章 §5.1.2 模型架构

Architecture (from paper §5.1.2):
  1. Input layer: multi-source heterogeneous data (historical ops, vessel plans,
     IoT sensor data, environmental factors)
  2. Feature extraction:
     - Time dimension: Bidirectional 2-layer LSTM (64 hidden units) + 4-head self-attention
     - Space dimension: 2-layer GCN for spatial dependency modelling among yard blocks
  3. Fusion: Cross-attention between LSTM and GNN output streams
  4. Deep ensemble: K=5 models trained with different random seeds
  5. Output: mean arrival prediction + quantile-based confidence interval (PICP)

Training (from paper §5.1.2):
  - Input window: 14 days → predict next 7 days
  - Optimizer: Adam, lr=0.001
  - Early stopping patience = 10 epochs
  - Quantile loss (50 % central quantile + 10 % / 90 % for intervals)
  - ~213,255 parameters
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
#  Spatial Graph Construction   (论文 §5.1.2 "空间维度")
# ═══════════════════════════════════════════════════════════════════

def build_yard_adjacency(n_blocks: int,
                         distance_matrix: Optional[torch.Tensor] = None,
                         threshold: float = 0.3) -> torch.Tensor:
    """
    Build a normalised adjacency matrix for yard block spatial graph.

    If *distance_matrix* is provided (n_blocks × n_blocks), edges are
    kept where normalised distance < *threshold* and connectivity is
    symmetric; otherwise a simple ring topology is assumed (each block
    connected to its immediate neighbours).

    Returns
    -------
    torch.Tensor of shape (n_blocks, n_blocks) – symmetric, row-normalised.
    """
    if distance_matrix is not None:
        # Normalise distances to [0, 1] across the matrix
        d = distance_matrix.clone()
        d_max = d.max().clamp(min=1e-12)
        d_norm = d / d_max
        adj = (d_norm < threshold).float()
    else:
        # Default: ring adjacency (each block connects to ±1 neighbour)
        adj = torch.zeros(n_blocks, n_blocks)
        for i in range(n_blocks):
            adj[i, (i - 1) % n_blocks] = 1.0
            adj[i, (i + 1) % n_blocks] = 1.0

    # Symmetrise and row-normalise
    adj = (adj + adj.T).clamp(max=1.0)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-12)
    adj_norm = adj / deg
    return adj_norm


# ═══════════════════════════════════════════════════════════════════
#  GCN Layer   (论文 §5.1.2 "空间专家 — GCN")
# ═══════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """
    Single graph convolutional layer:

        H' = σ( A_norm · H · W )

    where *A_norm* is the row-normalised adjacency matrix.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_blocks, in_features)
        adj_norm : (n_blocks, n_blocks)  row-normalised adjacency

        Returns
        -------
        (batch, n_blocks, out_features)
        """
        # Graph convolution: A_norm · X · W
        support = self.linear(x)            # (B, N, out)
        out = torch.bmm(adj_norm.unsqueeze(0).expand(x.size(0), -1, -1), support)
        out = F.relu(out)
        out = self.dropout(out)
        return out


# ═══════════════════════════════════════════════════════════════════
#  2-Layer GCN   (论文 §5.1.2 — 2-layer GCN, 64→32 hidden dims)
# ═══════════════════════════════════════════════════════════════════

class SpatialGNN(nn.Module):
    """
    Two-layer GCN for spatial dependency modelling among yard blocks.

    Paper: 2-layer GCN, hidden dims 64 → 32.
    """

    def __init__(self,
                 in_features: int = 14,
                 hidden_dim: int = 64,
                 out_dim: int = 32,
                 dropout: float = 0.2,
                 n_blocks: int = 8):
        super().__init__()
        self.register_buffer("adj_norm", build_yard_adjacency(n_blocks))
        self.gcn1 = GCNLayer(in_features, hidden_dim, dropout=dropout)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_blocks, in_features)

        Returns
        -------
        (batch, n_blocks, out_dim)
        """
        adj = self.adj_norm
        h = self.gcn1(x, adj)   # (B, N, hidden_dim)
        h = self.gcn2(h, adj)   # (B, N, out_dim)
        return h


# ═══════════════════════════════════════════════════════════════════
#  Multi-Head Self-Attention over time   (论文 §5.1.2 "聚焦器")
# ═══════════════════════════════════════════════════════════════════

class TimeSelfAttention(nn.Module):
    """
    4-head self-attention applied over the temporal dimension of LSTM outputs.
    Paper: "attention mechanism as focuser — dynamically weights inputs during
    anomalies".
    """

    def __init__(self, hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, hidden_dim)
        mask : optional (batch, seq_len) boolean – True = padded position.

        Returns
        -------
        (batch, seq_len, hidden_dim)
        """
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        # Project and reshape -> (B, H, T, Dh)
        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T) broadcast over heads & key positions
            attn = attn.masked_fill(mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)       # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out


# ═══════════════════════════════════════════════════════════════════
#  Cross-Attention Fusion   (论文 §5.1.2 "融合 — cross-attention")
# ═══════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between LSTM (time) and GNN (space) output streams.
    LSTM output serves as *query*, GNN output as *key/value*.

    Paper: "fusion: cross-attention between LSTM and GNN outputs".
    """

    def __init__(self, time_dim: int, space_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        # Project to common dimension
        self.time_proj = nn.Linear(time_dim, time_dim)
        self.space_proj = nn.Linear(space_dim, time_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=time_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(time_dim, time_dim)

    def forward(self, time_feat: torch.Tensor,
                space_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        time_feat  : (batch, time_dim)
        space_feat : (batch, n_blocks, space_dim)

        Returns
        -------
        (batch, time_dim) fused representation
        """
        # Project time feature to query
        query = self.time_proj(time_feat).unsqueeze(1)  # (B, 1, time_dim)

        # Aggregate space features: mean pooling across blocks then project to key/value
        space_pooled = space_feat.mean(dim=1)            # (B, space_dim)
        key = self.space_proj(space_pooled).unsqueeze(1) # (B, 1, time_dim)
        value = key                                       # same projection for simplicity

        attn_out, _ = self.cross_attn(query, key, value)  # (B, 1, time_dim)
        fused = self.out_proj(attn_out.squeeze(1))         # (B, time_dim)
        return fused


# ═══════════════════════════════════════════════════════════════════
#  Full LSTM–GNN–Attention Hybrid Model   (论文 §5.1.2)
# ═══════════════════════════════════════════════════════════════════

class LSTMGNNAttentionPredictor(nn.Module):
    """
    Multi-modal hybrid prediction model combining:

    * LSTM as **time expert** — captures daily/weekly periodicity of yard ops
    * GNN as **space expert** — models spatial relationships between yard blocks
    * Attention mechanism as **focuser** — dynamically weights inputs during anomalies

    Architecture (paper §5.1.2):
      Input (14-day window) → Bidirectional 2-layer LSTM (64 hidden, dropout=0.2)
                                → 4-head self-attention (focuser)
      Input (14-day window) → 2-layer GCN (64→32 hidden dims)
      Fusion → Cross-attention between LSTM and GNN outputs
      Output → Linear head → mean arrival + confidence interval

    Parameters
    ----------
    input_window : int
        Number of time steps in the input sequence (default 14 days).
    forecast_steps : int
        Number of steps to predict (default 7 days).
    n_blocks : int
        Number of yard blocks in the spatial graph.
    lstm_hidden : int
        Hidden units per LSTM direction (default 64).
    lstm_layers : int
        Number of LSTM layers (default 2).
    gcn_hidden : int
        Hidden dimension of first GCN layer (default 64).
    gcn_out : int
        Output dimension of second GCN layer (default 32).
    n_attn_heads : int
        Number of self-attention heads (default 4).
    dropout : float
        Dropout rate (default 0.2).
    """

    def __init__(
        self,
        input_window: int = 14,
        forecast_steps: int = 7,
        n_blocks: int = 8,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        gcn_hidden: int = 64,
        gcn_out: int = 32,
        n_attn_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_window = input_window
        self.forecast_steps = forecast_steps
        self.n_blocks = n_blocks

        # ── Feature embedding (project raw features into model space) ──
        # Input: for each time step and each block we have occupancy + covariates.
        # We treat the blocks as a flat vector per timestep: (batch, seq, n_blocks * n_features)
        # For simplicity we let the LSTM learn the embedding.
        # Default: 1 feature = occupancy rate per block → n_features = 1.
        # With vessel plans + IoT + environment, we use 3 features per block.
        n_features_per_block = 3  # occupancy, vessel arrivals, external factor
        lstm_input_dim = n_blocks * n_features_per_block

        # ── Bidirectional 2-layer LSTM (time expert) ──
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # ── 4-Head Self-Attention over time (focuser) ──
        self.time_attention = TimeSelfAttention(
            hidden_dim=lstm_out_dim, n_heads=n_attn_heads, dropout=dropout,
        )

        # ── 2-Layer GCN (space expert) ──
        self.gnn = SpatialGNN(
            in_features=n_features_per_block,
            hidden_dim=gcn_hidden,
            out_dim=gcn_out,
            dropout=dropout,
            n_blocks=n_blocks,
        )

        # ── Fusion: cross-attention ──
        self.fusion = CrossAttentionFusion(
            time_dim=lstm_out_dim,
            space_dim=gcn_out,
            n_heads=n_attn_heads,
            dropout=dropout,
        )

        # ── Prediction head ──
        # Outputs: mean, lower_quantile (10%), upper_quantile (90%)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_steps * 3),  # mean, q10, q90 for each forecast step
        )

        self._init_weights()

    def _init_weights(self):
        """Initialise weights with xavier uniform for linear, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if "lstm" in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x_time: torch.Tensor,
                x_space: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_time  : (batch, input_window, n_blocks * n_features)
            Flattened per-timestep features across all blocks.
        x_space : (batch, n_blocks, n_features_per_block)
            Static/averaged features per block for GCN input.
            In practice we use the last timestep or temporal average.

        Returns
        -------
        mean_pred : (batch, forecast_steps)
        q10_pred  : (batch, forecast_steps)
        q90_pred  : (batch, forecast_steps)
        """
        # ── Time path: LSTM + Self-Attention ──
        lstm_out, (h_n, c_n) = self.lstm(x_time)           # (B, T, lstm_out_dim)
        attn_out = self.time_attention(lstm_out)            # (B, T, lstm_out_dim)

        # Pool over time: use the attention-weighted output at the last position
        # as the global time representation
        time_feat = attn_out[:, -1, :]                      # (B, lstm_out_dim)

        # ── Space path: GCN ──
        space_feat = self.gnn(x_space)                      # (B, n_blocks, gcn_out)

        # ── Fusion ──
        fused = self.fusion(time_feat, space_feat)           # (B, lstm_out_dim)

        # ── Output head ──
        raw = self.head(fused)                              # (B, forecast_steps * 3)
        mean_pred = raw[:, :self.forecast_steps]
        q10_pred = raw[:, self.forecast_steps:2 * self.forecast_steps]
        q90_pred = raw[:, 2 * self.forecast_steps:]

        # Constrain quantiles so q10 ≤ mean ≤ q90 — use softplus offset
        q10_pred = mean_pred - F.softplus(mean_pred - q10_pred)
        q90_pred = mean_pred + F.softplus(q90_pred - mean_pred)

        return mean_pred, q10_pred, q90_pred


    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
#  Quantile Loss   (论文 §5.1.2 — PICP quantile loss)
# ═══════════════════════════════════════════════════════════════════

def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                  quantile: float) -> torch.Tensor:
    """
    Pinball / quantile loss:

        L(y, ŷ, τ) = max(τ(y − ŷ), (τ−1)(y − ŷ))

    Parameters
    ----------
    pred : (batch, forecast_steps)  — predicted quantile
    target : (batch, forecast_steps) — ground truth
    quantile : float in (0, 1)

    Returns
    -------
    Scalar loss (mean over batch × time).
    """
    error = target - pred
    loss = torch.max(quantile * error, (quantile - 1.0) * error)
    return loss.mean()


def total_quantile_loss(mean_pred: torch.Tensor,
                        q10_pred: torch.Tensor,
                        q90_pred: torch.Tensor,
                        target: torch.Tensor,
                        alpha: float = 0.5) -> torch.Tensor:
    """
    Combined quantile loss for central prediction and PI coverage.

    Paper: quantile loss for PICP (50 % quantile + 10 % / 90 % for intervals).
    The central (median) quantile is weighted more heavily.

    Parameters
    ----------
    mean_pred : (batch, forecast_steps)
    q10_pred  : (batch, forecast_steps)
    q90_pred  : (batch, forecast_steps)
    target    : (batch, forecast_steps)
    alpha     : float — weight for central quantile vs. interval bounds.

    Returns
    -------
    Scalar loss.
    """
    loss_median = quantile_loss(mean_pred, target, 0.50)
    loss_q10 = quantile_loss(q10_pred, target, 0.10)
    loss_q90 = quantile_loss(q90_pred, target, 0.90)
    return alpha * loss_median + (1 - alpha) * 0.5 * (loss_q10 + loss_q90)


# ═══════════════════════════════════════════════════════════════════
#  PICP & PINRW Metrics   (论文 §5.1.2 — prediction interval quality)
# ═══════════════════════════════════════════════════════════════════

def picp_metric(q10: torch.Tensor, q90: torch.Tensor,
                target: torch.Tensor) -> float:
    """
    Prediction Interval Coverage Probability — fraction of targets falling
    inside the predicted interval [q10, q90].

    Returns scalar in [0, 1].
    """
    inside = ((target >= q10) & (target <= q90)).float().mean().item()
    return inside


def pinrw_metric(q10: torch.Tensor, q90: torch.Tensor,
                 target: torch.Tensor) -> float:
    """
    Prediction Interval Normalised Root Width — average interval width
    normalised by the range of the target.

    Returns scalar ≥ 0.
    """
    width = (q90 - q10).mean().item()
    target_range = (target.max() - target.min()).clamp(min=1e-12).item()
    return width / target_range
