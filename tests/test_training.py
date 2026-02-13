"""Tests for training functionality."""

import tempfile
from pathlib import Path

import pytest
import torch

from hierarchical_attention_reconciliation_retail_forecasting.data.loader import create_dataloaders
from hierarchical_attention_reconciliation_retail_forecasting.models.model import HierarchicalAttentionForecaster
from hierarchical_attention_reconciliation_retail_forecasting.training.trainer import Trainer


class TestTrainer:
    """Test Trainer class."""

    def test_initialization(self, sample_config, device):
        """Test trainer initialization."""
        model = HierarchicalAttentionForecaster(sample_config['model'])
        trainer = Trainer(model, sample_config, device)

        assert trainer.epochs == sample_config['training']['epochs']
        assert trainer.device == device

    def test_train_epoch(self, sample_config, device):
        """Test single training epoch."""
        model = HierarchicalAttentionForecaster(sample_config['model'])
        trainer = Trainer(model, sample_config, device)

        train_loader, _, _ = create_dataloaders(sample_config, use_synthetic=True)

        metrics = trainer.train_epoch(train_loader)

        assert 'loss' in metrics
        assert metrics['loss'] >= 0

    def test_validate(self, sample_config, device):
        """Test validation."""
        model = HierarchicalAttentionForecaster(sample_config['model'])
        trainer = Trainer(model, sample_config, device)

        _, val_loader, _ = create_dataloaders(sample_config, use_synthetic=True)

        metrics = trainer.validate(val_loader)

        assert 'loss' in metrics
        assert metrics['loss'] >= 0

    def test_fit(self, sample_config, device):
        """Test full training loop."""
        model = HierarchicalAttentionForecaster(sample_config['model'])
        trainer = Trainer(model, sample_config, device)

        train_loader, val_loader, _ = create_dataloaders(sample_config, use_synthetic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.fit(
                train_loader, val_loader, checkpoint_dir=Path(tmpdir)
            )

            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) > 0
            assert len(history['val_loss']) > 0

            # Check checkpoint saved
            checkpoint_path = Path(tmpdir) / "best_model.pt"
            assert checkpoint_path.exists()

    def test_load_checkpoint(self, sample_config, device):
        """Test checkpoint loading."""
        model = HierarchicalAttentionForecaster(sample_config['model'])
        trainer = Trainer(model, sample_config, device)

        train_loader, val_loader, _ = create_dataloaders(sample_config, use_synthetic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            trainer.fit(train_loader, val_loader, checkpoint_dir=Path(tmpdir))

            # Create new trainer and load
            new_model = HierarchicalAttentionForecaster(sample_config['model'])
            new_trainer = Trainer(new_model, sample_config, device)

            checkpoint_path = Path(tmpdir) / "best_model.pt"
            new_trainer.load_checkpoint(checkpoint_path)

            # Models should have same parameters
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_gradient_clipping(self, sample_config, device):
        """Test gradient clipping."""
        sample_config['training']['gradient_clip'] = 0.5

        model = HierarchicalAttentionForecaster(sample_config['model'])
        trainer = Trainer(model, sample_config, device)

        train_loader, _, _ = create_dataloaders(sample_config, use_synthetic=True)

        # Should not raise error
        metrics = trainer.train_epoch(train_loader)
        assert 'loss' in metrics

    def test_learning_rate_scheduler(self, sample_config, device):
        """Test learning rate scheduling."""
        model = HierarchicalAttentionForecaster(sample_config['model'])
        trainer = Trainer(model, sample_config, device)

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        train_loader, val_loader, _ = create_dataloaders(sample_config, use_synthetic=True)

        # Train for one epoch
        trainer.train_epoch(train_loader)
        trainer.validate(val_loader)
        trainer.scheduler.step()

        # Learning rate may have changed
        # (depends on scheduler type, but should not raise error)
        final_lr = trainer.optimizer.param_groups[0]['lr']
        assert final_lr > 0
