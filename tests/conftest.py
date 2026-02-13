"""Pytest configuration and fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'sequence_length': 10,
            'prediction_length': 5,
            'batch_size': 16,
            'num_workers': 0,
            'train_split': 0.7,
            'val_split': 0.15,
            'hierarchy_levels': ['store', 'department', 'category', 'item'],
        },
        'model': {
            'attention_hidden_dim': 32,
            'attention_heads': 4,
            'attention_dropout': 0.1,
            'feature_dim': 64,
            'num_encoder_layers': 2,
            'num_decoder_layers': 1,
            'reconciliation_hidden_dim': 32,
            'learnable_reconciliation': True,
            'reconciliation_method': 'neural',
            'output_dim': 1,
            'quantiles': [0.1, 0.5, 0.9],
            'prediction_length': 5,
        },
        'training': {
            'epochs': 1,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'gradient_clip': 1.0,
            'scheduler': 'cosine',
            'warmup_epochs': 0,
            'min_lr': 0.000001,
            'early_stopping': False,
            'patience': 5,
            'coherence_weight': 0.3,
            'quantile_weight': 0.7,
            'mixed_precision': False,
            'accumulation_steps': 1,
        },
        'evaluation': {
            'metrics': ['WRMSSE', 'hierarchy_coherence_violation'],
            'save_predictions': True,
            'generate_plots': False,
            'plot_samples': 2,
        },
        'reproducibility': {
            'seed': 42,
            'deterministic': True,
        },
    }


@pytest.fixture
def sample_time_series():
    """Sample time series data for testing."""
    np.random.seed(42)
    num_series = 20
    time_steps = 50
    data = np.random.randn(num_series, time_steps).astype(np.float32)
    return data


@pytest.fixture
def sample_hierarchy_info():
    """Sample hierarchy information for testing."""
    num_series = 20
    return {
        'store': np.random.randint(0, 3, num_series),
        'department': np.random.randint(0, 5, num_series),
        'category': np.random.randint(0, 7, num_series),
        'item': np.arange(num_series),
    }


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device('cpu')
