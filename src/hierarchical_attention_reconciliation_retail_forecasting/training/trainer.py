"""Training loop with learning rate scheduling and early stopping."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import CombinedLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Model trainer with advanced features.

    Args:
        model: PyTorch model to train
        config: Training configuration
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training parameters
        self.epochs = config['training']['epochs']
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5),
        )

        # Loss function
        self.criterion = CombinedLoss(
            coherence_weight=config['training'].get('coherence_weight', 0.3),
            quantile_weight=config['training'].get('quantile_weight', 0.7),
            quantiles=config['model'].get('quantiles', [0.1, 0.5, 0.9]),
        ).to(device)

        # Learning rate scheduler
        self.scheduler = self._create_scheduler(config['training'])

        # Early stopping
        self.early_stopping = config['training'].get('early_stopping', True)
        self.patience = config['training'].get('patience', 15)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        logger.info(
            f"Initialized Trainer: epochs={self.epochs}, "
            f"lr={config['training']['learning_rate']}, "
            f"device={device}, mixed_precision={self.use_amp}"
        )

    def _create_scheduler(self, config: Dict):
        """Create learning rate scheduler.

        Args:
            config: Training configuration

        Returns:
            Learning rate scheduler
        """
        scheduler_type = config.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=config.get('min_lr', 1e-6),
            )
        elif scheduler_type == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1),
            )
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=config.get('min_lr', 1e-6),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        logger.info(f"Created {scheduler_type} scheduler")
        return scheduler

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            hierarchy_info = batch['hierarchy_level'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x, hierarchy_info)
                    loss, loss_dict = self.criterion(
                        outputs['predictions'], y, hierarchy_info
                    )
            else:
                outputs = self.model(x, hierarchy_info)
                loss, loss_dict = self.criterion(
                    outputs['predictions'], y, hierarchy_info
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                hierarchy_info = batch['hierarchy_level'].to(self.device)

                outputs = self.model(x, hierarchy_info)
                loss, loss_dict = self.criterion(
                    outputs['predictions'], y, hierarchy_info
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, list]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics['loss']

            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if checkpoint_dir:
                    checkpoint_path = checkpoint_dir / "best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'config': self.config,
                    }, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.early_stopping and self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={self.patience})"
                )
                break

        logger.info("Training completed")
        return self.history

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
