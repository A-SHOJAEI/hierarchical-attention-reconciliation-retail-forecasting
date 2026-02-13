#!/usr/bin/env python
"""Evaluation script for hierarchical attention forecasting model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from hierarchical_attention_reconciliation_retail_forecasting.data.loader import create_dataloaders
from hierarchical_attention_reconciliation_retail_forecasting.evaluation.analysis import ResultsAnalyzer
from hierarchical_attention_reconciliation_retail_forecasting.evaluation.metrics import HierarchicalMetrics
from hierarchical_attention_reconciliation_retail_forecasting.models.model import HierarchicalAttentionForecaster
from hierarchical_attention_reconciliation_retail_forecasting.utils.config import load_config, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate hierarchical attention forecasting model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate',
    )
    return parser.parse_args()


def evaluate_model(model, data_loader, device, config):
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to use
        config: Configuration dictionary

    Returns:
        Tuple of (predictions, targets, quantile_predictions)
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_quantile_predictions = {q: [] for q in config['model']['quantiles']}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            hierarchy_info = batch['hierarchy_level'].to(device)

            # Get predictions
            outputs = model(x, hierarchy_info)
            predictions = outputs['predictions']

            # Store predictions and targets
            all_predictions.append(predictions[:, :, len(config['model']['quantiles'])//2].cpu().numpy())
            all_targets.append(y.cpu().numpy())

            # Store quantile predictions
            for i, q in enumerate(config['model']['quantiles']):
                all_quantile_predictions[q].append(predictions[:, :, i].cpu().numpy())

    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    for q in all_quantile_predictions:
        all_quantile_predictions[q] = np.concatenate(all_quantile_predictions[q], axis=0)

    return predictions, targets, all_quantile_predictions


def main():
    """Main evaluation function."""
    args = parse_args()

    logger.info("="*80)
    logger.info("Model Evaluation")
    logger.info("="*80)

    # Load configuration
    config = load_config(Path(args.config))
    logger.info(f"Loaded configuration from {args.config}")

    # Set random seed
    seed = config.get('reproducibility', {}).get('seed', 42)
    set_seed(seed)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create dataloaders
        logger.info("Loading data...")
        train_loader, val_loader, test_loader = create_dataloaders(
            config, use_synthetic=True
        )

        # Select appropriate loader
        if args.split == 'train':
            data_loader = train_loader
        elif args.split == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader

        logger.info(f"Evaluating on {args.split} split ({len(data_loader)} batches)")

        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Merge prediction_length from data config into model config
        model_config = config['model'].copy()
        model_config['prediction_length'] = config['data']['prediction_length']
        model = HierarchicalAttentionForecaster(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        logger.info("Model loaded successfully")

        # Evaluate
        logger.info("Running evaluation...")
        predictions, targets, quantile_predictions = evaluate_model(
            model, data_loader, device, config
        )

        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Targets shape: {targets.shape}")

        # Compute metrics
        logger.info("Computing metrics...")
        metrics_calculator = HierarchicalMetrics(
            hierarchy_levels=config['data']['hierarchy_levels']
        )

        # Ensure proper shape (predictions already 2D from median extraction)
        if predictions.ndim == 3:
            predictions_2d = predictions.squeeze(-1)
        else:
            predictions_2d = predictions

        if targets.ndim == 3:
            targets_2d = targets.squeeze(-1)
        else:
            targets_2d = targets

        metrics = metrics_calculator.compute_all_metrics(
            predictions=predictions_2d,
            targets=targets_2d,
            quantile_predictions={
                q: quantile_predictions[q].squeeze(-1) if quantile_predictions[q].ndim == 3 else quantile_predictions[q]
                for q in config['model']['quantiles']
            },
        )

        # Print metrics summary
        logger.info("="*80)
        logger.info("Evaluation Results")
        logger.info("="*80)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name:30s}: {value:.4f}")
        logger.info("="*80)

        # Save metrics (convert numpy types to Python types for JSON)
        metrics_json = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        metrics_path = results_dir / f"metrics_{args.split}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        # Save predictions
        if config['evaluation'].get('save_predictions', True):
            predictions_path = results_dir / f"predictions_{args.split}.npz"
            np.savez(
                predictions_path,
                predictions=predictions,
                targets=targets,
                **{f'quantile_{q}': quantile_predictions[q] for q in config['model']['quantiles']}
            )
            logger.info(f"Saved predictions to {predictions_path}")

        # Generate visualizations
        if config['evaluation'].get('generate_plots', True):
            logger.info("Generating visualizations...")
            analyzer = ResultsAnalyzer(results_dir)

            # Plot predictions
            analyzer.plot_predictions(
                predictions_2d,
                targets_2d,
                num_samples=config['evaluation'].get('plot_samples', 10),
                filename=f"predictions_{args.split}.png",
            )

            # Plot error distribution
            analyzer.plot_error_distribution(
                predictions_2d,
                targets_2d,
                filename=f"error_distribution_{args.split}.png",
            )

            # Plot quantile predictions
            analyzer.plot_quantile_predictions(
                {q: quantile_predictions[q].squeeze(-1) if quantile_predictions[q].ndim == 3 else quantile_predictions[q]
                 for q in config['model']['quantiles']},
                targets_2d,
                series_idx=0,
                filename=f"quantile_predictions_{args.split}.png",
            )

            logger.info("Visualizations saved")

        logger.info("="*80)
        logger.info("Evaluation completed successfully!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
