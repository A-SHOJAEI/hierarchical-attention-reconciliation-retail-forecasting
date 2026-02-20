"""Core model implementation."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .components import HierarchicalReconciliationLayer, TemporalAttention

logger = logging.getLogger(__name__)


class HierarchicalAttentionForecaster(nn.Module):
    """Hierarchical attention-based forecaster with learnable reconciliation.

    This model combines:
    1. Temporal attention to capture important historical patterns
    2. Feature extraction with encoder-decoder architecture
    3. Learnable hierarchical reconciliation (novel contribution)
    4. Probabilistic quantile forecasting

    Args:
        config: Model configuration dictionary
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.config = config

        # Model dimensions
        self.input_dim = config.get('output_dim', 1)
        self.feature_dim = config.get('feature_dim', 256)
        self.attention_hidden_dim = config.get('attention_hidden_dim', 128)
        self.num_heads = config.get('attention_heads', 8)
        self.num_quantiles = len(config.get('quantiles', [0.1, 0.5, 0.9]))
        self.prediction_length = config.get('prediction_length', None)

        # Architecture parameters
        self.num_encoder_layers = config.get('num_encoder_layers', 3)
        self.num_decoder_layers = config.get('num_decoder_layers', 2)

        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.feature_dim)

        # Temporal attention layers (encoder)
        self.encoder_attention_layers = nn.ModuleList([
            TemporalAttention(
                input_dim=self.feature_dim,
                hidden_dim=self.attention_hidden_dim,
                num_heads=self.num_heads,
                dropout=config.get('attention_dropout', 0.1),
            )
            for _ in range(self.num_encoder_layers)
        ])

        # Feed-forward layers
        self.encoder_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(config.get('attention_dropout', 0.1)),
                nn.Linear(self.feature_dim * 4, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
            )
            for _ in range(self.num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_attention_layers = nn.ModuleList([
            TemporalAttention(
                input_dim=self.feature_dim,
                hidden_dim=self.attention_hidden_dim,
                num_heads=self.num_heads,
                dropout=config.get('attention_dropout', 0.1),
            )
            for _ in range(self.num_decoder_layers)
        ])

        self.decoder_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(config.get('attention_dropout', 0.1)),
                nn.Linear(self.feature_dim * 4, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
            )
            for _ in range(self.num_decoder_layers)
        ])

        # Hierarchical reconciliation layer (novel component)
        self.reconciliation_layer = HierarchicalReconciliationLayer(
            num_levels=4,  # store, department, category, item
            hidden_dim=config.get('reconciliation_hidden_dim', 128),
            learnable=config.get('learnable_reconciliation', True),
        )

        # Hierarchy embedding
        self.hierarchy_embedding = nn.Linear(4, self.feature_dim)

        # Output projection for quantile forecasting
        self.output_projection = nn.Linear(self.feature_dim, self.num_quantiles)

        logger.info(
            f"Initialized HierarchicalAttentionForecaster: "
            f"feature_dim={self.feature_dim}, "
            f"encoder_layers={self.num_encoder_layers}, "
            f"decoder_layers={self.num_decoder_layers}, "
            f"quantiles={self.num_quantiles}"
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode input sequence with attention.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (encoded_features, attention_weights_list)
        """
        # Project input
        h = self.input_projection(x)

        attention_weights = []

        # Apply encoder layers
        for attn_layer, ffn_layer in zip(
            self.encoder_attention_layers, self.encoder_ffn
        ):
            # Attention
            h_attn, attn_w = attn_layer(h)
            attention_weights.append(attn_w)

            # Feed-forward with residual
            h = h_attn + ffn_layer(h_attn)

        return h, attention_weights

    def decode(
        self, encoded: torch.Tensor, prediction_length: int
    ) -> torch.Tensor:
        """Decode to generate forecasts.

        Args:
            encoded: Encoded features (batch, seq_len, feature_dim)
            prediction_length: Number of steps to forecast

        Returns:
            Decoded features (batch, prediction_length, feature_dim)
        """
        batch_size = encoded.shape[0]

        # Initialize decoder input with last encoded state
        h = encoded[:, -1:, :].repeat(1, prediction_length, 1)

        # Apply decoder layers
        for attn_layer, ffn_layer in zip(
            self.decoder_attention_layers, self.decoder_ffn
        ):
            # Self-attention on decoder
            h_attn, _ = attn_layer(h)

            # Feed-forward with residual
            h = h_attn + ffn_layer(h_attn)

        return h

    def forward(
        self, x: torch.Tensor, hierarchy_info: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input sequence (batch, seq_len, input_dim)
            hierarchy_info: Hierarchy level encoding (batch, num_levels)

        Returns:
            Dictionary containing predictions and attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Encode input sequence
        encoded, attention_weights = self.encode(x)

        # Incorporate hierarchy information
        hierarchy_emb = self.hierarchy_embedding(hierarchy_info)
        encoded = encoded + hierarchy_emb.unsqueeze(1)

        # Decode to generate forecast
        # Use configured prediction length if available, else use seq_len
        prediction_length = self.prediction_length if self.prediction_length is not None else seq_len
        decoded = self.decode(encoded, prediction_length)

        # Project to quantile predictions
        quantile_forecasts = self.output_projection(decoded)

        # Apply hierarchical reconciliation
        # Use median quantile for reconciliation
        median_idx = self.num_quantiles // 2
        median_forecast = quantile_forecasts[:, :, median_idx:median_idx+1].clone()

        reconciled_forecast, coherence_score = self.reconciliation_layer(
            median_forecast, hierarchy_info
        )

        # Replace median with reconciled version (avoid in-place operation)
        quantile_forecasts = torch.cat([
            quantile_forecasts[:, :, :median_idx],
            reconciled_forecast,
            quantile_forecasts[:, :, median_idx+1:]
        ], dim=-1)

        return {
            'predictions': quantile_forecasts,
            'reconciled': reconciled_forecast,
            'coherence_score': coherence_score,
            'attention_weights': attention_weights,
        }

    def predict(
        self, x: torch.Tensor, hierarchy_info: torch.Tensor
    ) -> torch.Tensor:
        """Make predictions (inference mode).

        Args:
            x: Input sequence
            hierarchy_info: Hierarchy information

        Returns:
            Median quantile predictions
        """
        with torch.no_grad():
            outputs = self.forward(x, hierarchy_info)
            median_idx = self.num_quantiles // 2
            return outputs['predictions'][:, :, median_idx:median_idx+1]
