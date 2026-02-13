#!/usr/bin/env python
"""Training script for hierarchical attention forecasting model."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hierarchical_attention_reconciliation_retail_forecasting.data.loader import create_dataloaders
from hierarchical_attention_reconciliation_retail_forecasting.models.model import HierarchicalAttentionForecaster
from hierarchical_attention_reconciliation_retail_forecasting.training.trainer import Trainer
from hierarchical_attention_reconciliation_retail_forecasting.utils.config import load_config, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log'),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train hierarchical attention forecasting model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models',
        help='Directory to save model checkpoints',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)',
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    logger.info("="*80)
    logger.info("Hierarchical Attention Reconciliation Retail Forecasting")
    logger.info("="*80)

    # Load configuration
    try:
        config = load_config(Path(args.config))
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Set random seed for reproducibility
    seed = config.get('reproducibility', {}).get('seed', 42)
    set_seed(seed)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MLflow tracking (optional)
    try:
        import mlflow
        mlflow.set_experiment("hierarchical_attention_forecasting")
        mlflow.start_run()
        mlflow.log_params({
            'learning_rate': config['training']['learning_rate'],
            'epochs': config['training']['epochs'],
            'batch_size': config['data']['batch_size'],
            'feature_dim': config['model']['feature_dim'],
            'attention_heads': config['model']['attention_heads'],
            'learnable_reconciliation': config['model']['learnable_reconciliation'],
        })
        logger.info("MLflow tracking enabled")
        mlflow_enabled = True
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")
        mlflow_enabled = False

    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            config, use_synthetic=True
        )
        logger.info(
            f"Data loaded - Train: {len(train_loader)} batches, "
            f"Val: {len(val_loader)} batches, "
            f"Test: {len(test_loader)} batches"
        )

        # Create model
        logger.info("Initializing model...")
        # Merge prediction_length from data config into model config
        model_config = config['model'].copy()
        model_config['prediction_length'] = config['data']['prediction_length']
        model = HierarchicalAttentionForecaster(model_config)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model initialized - Total params: {num_params:,}, "
            f"Trainable: {num_trainable:,}"
        )

        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(model, config, device)

        # Train model
        logger.info("Starting training...")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
        )

        # Save training history
        import json
        history_path = results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Log metrics to MLflow
        if mlflow_enabled:
            try:
                for epoch, (train_loss, val_loss) in enumerate(
                    zip(history['train_loss'], history['val_loss'])
                ):
                    mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, step=epoch)
                mlflow.log_artifact(str(checkpoint_dir / "best_model.pt"))
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if mlflow_enabled:
            try:
                mlflow.end_run()
            except:
                pass


if __name__ == "__main__":
    main()
