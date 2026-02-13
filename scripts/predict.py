#!/usr/bin/env python
"""Prediction script for hierarchical attention forecasting model."""

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

from hierarchical_attention_reconciliation_retail_forecasting.models.model import HierarchicalAttentionForecaster
from hierarchical_attention_reconciliation_retail_forecasting.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make predictions with hierarchical attention forecasting model'
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
        '--input',
        type=str,
        required=True,
        help='Path to input data (npy file with shape [num_series, seq_len])',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Path to save predictions',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)',
    )
    parser.add_argument(
        '--hierarchy-level',
        type=str,
        default='item',
        choices=['store', 'department', 'category', 'item'],
        help='Hierarchy level of input data',
    )
    return parser.parse_args()


def prepare_input(input_path, hierarchy_level, sequence_length):
    """Prepare input data for prediction.

    Args:
        input_path: Path to input data file
        hierarchy_level: Hierarchy level name
        sequence_length: Required sequence length

    Returns:
        Tuple of (input_tensor, hierarchy_info_tensor)
    """
    # Load data
    data = np.load(input_path)

    if data.ndim == 1:
        # Single series
        data = data.reshape(1, -1)

    num_series = data.shape[0]

    # Take last sequence_length points
    if data.shape[1] < sequence_length:
        raise ValueError(
            f"Input data too short: {data.shape[1]} < {sequence_length}"
        )

    input_data = data[:, -sequence_length:]

    # Create hierarchy info
    hierarchy_levels = ['store', 'department', 'category', 'item']
    level_idx = hierarchy_levels.index(hierarchy_level)

    hierarchy_info = np.zeros((num_series, 4), dtype=np.float32)
    hierarchy_info[:, level_idx] = 1.0  # One-hot encoding

    # Convert to tensors
    input_tensor = torch.FloatTensor(input_data).unsqueeze(-1)
    hierarchy_tensor = torch.FloatTensor(hierarchy_info)

    return input_tensor, hierarchy_tensor


def main():
    """Main prediction function."""
    args = parse_args()

    logger.info("="*80)
    logger.info("Model Prediction")
    logger.info("="*80)

    # Load configuration
    config = load_config(Path(args.config))
    logger.info(f"Loaded configuration from {args.config}")

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Merge prediction_length from data config into model config
        model_config = config['model'].copy()
        model_config['prediction_length'] = config['data']['prediction_length']
        model = HierarchicalAttentionForecaster(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        logger.info("Model loaded successfully")

        # Prepare input
        logger.info(f"Loading input from {args.input}")
        input_tensor, hierarchy_tensor = prepare_input(
            args.input,
            args.hierarchy_level,
            config['data']['sequence_length'],
        )

        input_tensor = input_tensor.to(device)
        hierarchy_tensor = hierarchy_tensor.to(device)

        logger.info(f"Input shape: {input_tensor.shape}")

        # Make predictions
        logger.info("Generating predictions...")
        with torch.no_grad():
            outputs = model(input_tensor, hierarchy_tensor)

        predictions = outputs['predictions'].cpu().numpy()
        coherence_scores = outputs['coherence_score'].cpu().numpy()

        logger.info(f"Predictions shape: {predictions.shape}")

        # Extract quantile predictions
        quantiles = config['model']['quantiles']
        quantile_predictions = {}

        for i, q in enumerate(quantiles):
            quantile_predictions[f'quantile_{q}'] = predictions[:, :, i].tolist()

        # Prepare output
        output_data = {
            'input_shape': list(input_tensor.shape),
            'hierarchy_level': args.hierarchy_level,
            'quantiles': quantiles,
            'predictions': quantile_predictions,
            'median_forecast': predictions[:, :, len(quantiles)//2].tolist(),
            'coherence_scores': coherence_scores.tolist(),
            'prediction_length': predictions.shape[1],
        }

        # Save predictions
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved predictions to {output_path}")

        # Print summary
        logger.info("="*80)
        logger.info("Prediction Summary")
        logger.info("="*80)
        logger.info(f"Number of series: {predictions.shape[0]}")
        logger.info(f"Prediction horizon: {predictions.shape[1]} steps")
        logger.info(f"Quantiles: {quantiles}")

        median_idx = len(quantiles) // 2
        median_forecast = predictions[:, :, median_idx]

        logger.info(f"Median forecast range: [{median_forecast.min():.4f}, {median_forecast.max():.4f}]")
        logger.info(f"Mean coherence score: {coherence_scores.mean():.4f}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
