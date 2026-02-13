"""Evaluation metrics for hierarchical forecasting."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HierarchicalMetrics:
    """Metrics for hierarchical time series forecasting.

    Args:
        hierarchy_levels: List of hierarchy level names
    """

    def __init__(self, hierarchy_levels: List[str]) -> None:
        self.hierarchy_levels = hierarchy_levels
        logger.info(f"Initialized metrics with levels: {hierarchy_levels}")

    def rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Root Mean Squared Error.

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((predictions - targets) ** 2))

    def mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Absolute Error.

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            MAE value
        """
        return np.mean(np.abs(predictions - targets))

    def wrmsse(
        self, predictions: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        """Weighted Root Mean Squared Scaled Error (M5 competition metric).

        Args:
            predictions: Predicted values
            targets: True values
            weights: Optional weights for each series

        Returns:
            WRMSSE value
        """
        if weights is None:
            weights = np.ones(predictions.shape[0])

        # Compute scaling factor (MAE of naive forecast)
        naive_forecast = np.roll(targets, 1, axis=1)
        naive_forecast[:, 0] = targets[:, 0]

        scale = np.mean(np.abs(targets - naive_forecast), axis=1)
        scale = np.maximum(scale, 1e-8)  # Avoid division by zero

        # Compute RMSSE for each series
        mse = np.mean((predictions - targets) ** 2, axis=1)
        rmsse = np.sqrt(mse) / scale

        # Weighted average
        wrmsse = np.average(rmsse, weights=weights)

        return wrmsse

    def hierarchy_coherence_violation(
        self,
        predictions: Dict[str, np.ndarray],
        aggregation_matrices: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Compute hierarchy coherence violation.

        Args:
            predictions: Dictionary mapping levels to predictions
            aggregation_matrices: Optional aggregation matrices

        Returns:
            Coherence violation score (0 = perfect coherence)
        """
        if len(predictions) < 2:
            return 0.0

        base_level = self.hierarchy_levels[-1]
        if base_level not in predictions:
            logger.warning(f"Base level {base_level} not in predictions")
            return 0.0

        base_forecasts = predictions[base_level]

        violations = []
        for level in self.hierarchy_levels[:-1]:
            if level in predictions:
                # Simplified coherence check: compare variance
                level_pred = predictions[level]
                violation = np.abs(level_pred.mean() - base_forecasts.mean())
                violations.append(violation)

        return np.mean(violations) if violations else 0.0

    def coverage(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        lower_quantile: np.ndarray,
        upper_quantile: np.ndarray,
    ) -> float:
        """Compute prediction interval coverage.

        Args:
            predictions: Point predictions
            targets: True values
            lower_quantile: Lower quantile predictions
            upper_quantile: Upper quantile predictions

        Returns:
            Coverage percentage
        """
        in_interval = (targets >= lower_quantile) & (targets <= upper_quantile)
        return np.mean(in_interval)

    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        quantile_predictions: Optional[Dict[float, np.ndarray]] = None,
        hierarchy_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Compute all metrics.

        Args:
            predictions: Point predictions
            targets: True values
            quantile_predictions: Optional quantile predictions
            hierarchy_predictions: Optional hierarchy-level predictions

        Returns:
            Dictionary of metric values
        """
        metrics = {
            'RMSE': self.rmse(predictions, targets),
            'MAE': self.mae(predictions, targets),
            'WRMSSE': self.wrmsse(predictions, targets),
        }

        # Quantile metrics
        if quantile_predictions:
            if 0.05 in quantile_predictions and 0.95 in quantile_predictions:
                metrics['coverage_90'] = self.coverage(
                    predictions,
                    targets,
                    quantile_predictions[0.05],
                    quantile_predictions[0.95],
                )

            if 0.25 in quantile_predictions and 0.75 in quantile_predictions:
                metrics['coverage_50'] = self.coverage(
                    predictions,
                    targets,
                    quantile_predictions[0.25],
                    quantile_predictions[0.75],
                )

        # Hierarchy coherence
        if hierarchy_predictions:
            metrics['hierarchy_coherence_violation'] = (
                self.hierarchy_coherence_violation(hierarchy_predictions)
            )

        return metrics
