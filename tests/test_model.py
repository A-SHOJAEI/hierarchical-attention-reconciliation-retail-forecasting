"""Tests for model components."""

import pytest
import torch

from hierarchical_attention_reconciliation_retail_forecasting.models.components import (
    CoherenceLoss,
    CombinedLoss,
    HierarchicalReconciliationLayer,
    QuantileLoss,
    TemporalAttention,
)
from hierarchical_attention_reconciliation_retail_forecasting.models.model import (
    HierarchicalAttentionForecaster,
)


class TestTemporalAttention:
    """Test TemporalAttention module."""

    def test_initialization(self):
        """Test attention layer initialization."""
        attention = TemporalAttention(
            input_dim=64, hidden_dim=128, num_heads=8, dropout=0.1
        )
        assert attention.num_heads == 8
        assert attention.hidden_dim == 128

    def test_forward(self):
        """Test forward pass."""
        attention = TemporalAttention(
            input_dim=64, hidden_dim=128, num_heads=8, dropout=0.1
        )

        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64)

        output, attention_weights = attention(x)

        assert output.shape == (batch_size, seq_len, 64)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)


class TestHierarchicalReconciliationLayer:
    """Test HierarchicalReconciliationLayer module."""

    def test_initialization_learnable(self):
        """Test learnable reconciliation layer."""
        layer = HierarchicalReconciliationLayer(
            num_levels=4, hidden_dim=128, learnable=True
        )
        assert layer.learnable is True
        assert len(layer.projection_networks) == 3

    def test_initialization_non_learnable(self):
        """Test traditional reconciliation."""
        layer = HierarchicalReconciliationLayer(
            num_levels=4, hidden_dim=128, learnable=False
        )
        assert layer.learnable is False

    def test_forward_learnable(self):
        """Test forward pass with learnable reconciliation."""
        layer = HierarchicalReconciliationLayer(
            num_levels=4, hidden_dim=128, learnable=True
        )

        batch_size = 4
        pred_len = 10
        base_forecast = torch.randn(batch_size, pred_len, 1)
        hierarchy_info = torch.randn(batch_size, 4)

        reconciled, coherence = layer(base_forecast, hierarchy_info)

        assert reconciled.shape == base_forecast.shape
        assert coherence.shape == (batch_size, 1)

    def test_forward_non_learnable(self):
        """Test forward pass with traditional reconciliation."""
        layer = HierarchicalReconciliationLayer(
            num_levels=4, hidden_dim=128, learnable=False
        )

        batch_size = 4
        pred_len = 10
        base_forecast = torch.randn(batch_size, pred_len, 1)
        hierarchy_info = torch.randn(batch_size, 4)

        reconciled, coherence = layer(base_forecast, hierarchy_info)

        assert reconciled.shape == base_forecast.shape
        assert torch.allclose(reconciled, base_forecast)


class TestLossFunctions:
    """Test loss functions."""

    def test_coherence_loss(self):
        """Test coherence loss."""
        loss_fn = CoherenceLoss()

        batch_size = 4
        pred_len = 10
        forecasts = torch.randn(batch_size, pred_len, 1)
        hierarchy_info = torch.randn(batch_size, 4)

        loss = loss_fn(forecasts, hierarchy_info)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_quantile_loss(self):
        """Test quantile loss."""
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        batch_size = 4
        pred_len = 10
        predictions = torch.randn(batch_size, pred_len, 3)
        targets = torch.randn(batch_size, pred_len, 1)

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = CombinedLoss(
            coherence_weight=0.3,
            quantile_weight=0.7,
            quantiles=[0.1, 0.5, 0.9],
        )

        batch_size = 4
        pred_len = 10
        predictions = torch.randn(batch_size, pred_len, 3)
        targets = torch.randn(batch_size, pred_len, 1)
        hierarchy_info = torch.randn(batch_size, 4)

        loss, loss_dict = loss_fn(predictions, targets, hierarchy_info)

        assert loss.ndim == 0
        assert 'total' in loss_dict
        assert 'quantile' in loss_dict
        assert 'coherence' in loss_dict


class TestHierarchicalAttentionForecaster:
    """Test main forecaster model."""

    def test_initialization(self, sample_config):
        """Test model initialization."""
        model = HierarchicalAttentionForecaster(sample_config['model'])

        assert model.feature_dim == sample_config['model']['feature_dim']
        assert model.num_heads == sample_config['model']['attention_heads']

    def test_forward(self, sample_config):
        """Test forward pass."""
        model = HierarchicalAttentionForecaster(sample_config['model'])

        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 1)
        hierarchy_info = torch.randn(batch_size, 4)

        outputs = model(x, hierarchy_info)

        assert 'predictions' in outputs
        assert 'reconciled' in outputs
        assert 'coherence_score' in outputs
        assert 'attention_weights' in outputs

        assert outputs['predictions'].shape[0] == batch_size
        assert outputs['predictions'].shape[-1] == len(sample_config['model']['quantiles'])

    def test_predict(self, sample_config):
        """Test prediction method."""
        model = HierarchicalAttentionForecaster(sample_config['model'])
        model.eval()

        batch_size = 4
        seq_len = 10
        pred_len = sample_config['model']['prediction_length']
        x = torch.randn(batch_size, seq_len, 1)
        hierarchy_info = torch.randn(batch_size, 4)

        predictions = model.predict(x, hierarchy_info)

        assert predictions.shape == (batch_size, pred_len, 1)

    def test_model_parameters(self, sample_config):
        """Test model has trainable parameters."""
        model = HierarchicalAttentionForecaster(sample_config['model'])

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert num_params > 0
        assert num_trainable > 0
        assert num_params == num_trainable
