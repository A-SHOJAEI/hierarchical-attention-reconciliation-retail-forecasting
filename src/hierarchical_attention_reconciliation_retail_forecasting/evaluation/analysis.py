"""Results analysis and visualization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and visualize forecasting results.

    Args:
        results_dir: Directory to save results
    """

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ResultsAnalyzer: results_dir={results_dir}")

    def save_metrics(self, metrics: Dict[str, float], filename: str = "metrics.json") -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {filepath}")

    def plot_training_history(
        self, history: Dict[str, List[float]], filename: str = "training_history.png"
    ) -> None:
        """Plot training history.

        Args:
            history: Training history dictionary
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Learning rate
        axes[1].plot(history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True)

        plt.tight_layout()
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training history plot to {filepath}")

    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        num_samples: int = 10,
        filename: str = "predictions.png",
    ) -> None:
        """Plot sample predictions vs targets.

        Args:
            predictions: Predicted values (num_series, time_steps)
            targets: True values (num_series, time_steps)
            num_samples: Number of samples to plot
            filename: Output filename
        """
        num_samples = min(num_samples, predictions.shape[0])

        fig, axes = plt.subplots(
            num_samples, 1, figsize=(12, 2 * num_samples), squeeze=False
        )

        for i in range(num_samples):
            ax = axes[i, 0]
            ax.plot(targets[i], label='True', alpha=0.7)
            ax.plot(predictions[i], label='Predicted', alpha=0.7)
            ax.set_ylabel('Value')
            ax.set_title(f'Series {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel('Time Step')

        plt.tight_layout()
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved predictions plot to {filepath}")

    def plot_quantile_predictions(
        self,
        predictions: Dict[float, np.ndarray],
        targets: np.ndarray,
        series_idx: int = 0,
        filename: str = "quantile_predictions.png",
    ) -> None:
        """Plot quantile predictions.

        Args:
            predictions: Dictionary mapping quantiles to predictions
            targets: True values
            series_idx: Index of series to plot
            filename: Output filename
        """
        plt.figure(figsize=(12, 6))

        # Plot true values
        plt.plot(targets[series_idx], label='True', color='black', linewidth=2)

        # Plot quantiles
        quantiles = sorted(predictions.keys())
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(quantiles)))

        for q, color in zip(quantiles, colors):
            plt.plot(
                predictions[q][series_idx],
                label=f'Q{int(q*100)}',
                color=color,
                alpha=0.6,
            )

        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Quantile Predictions - Series {series_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved quantile predictions plot to {filepath}")

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        filename: str = "error_distribution.png",
    ) -> None:
        """Plot error distribution.

        Args:
            predictions: Predicted values
            targets: True values
            filename: Output filename
        """
        errors = predictions - targets

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(errors.flatten(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved error distribution plot to {filepath}")

    def generate_report(
        self,
        metrics: Dict[str, float],
        predictions: np.ndarray,
        targets: np.ndarray,
        history: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Generate complete analysis report.

        Args:
            metrics: Evaluation metrics
            predictions: Predicted values
            targets: True values
            history: Optional training history
        """
        logger.info("Generating analysis report")

        # Save metrics
        self.save_metrics(metrics)

        # Plot predictions
        self.plot_predictions(predictions, targets)

        # Plot error distribution
        self.plot_error_distribution(predictions, targets)

        # Plot training history if available
        if history:
            self.plot_training_history(history)

        logger.info("Analysis report generated successfully")
