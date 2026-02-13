"""Data preprocessing utilities."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HierarchyProcessor:
    """Process hierarchical structure information.

    Args:
        hierarchy_levels: List of hierarchy level names
    """

    def __init__(self, hierarchy_levels: List[str]) -> None:
        self.hierarchy_levels = hierarchy_levels
        self.aggregation_matrices = {}

        logger.info(f"Initialized HierarchyProcessor with levels: {hierarchy_levels}")

    def build_aggregation_matrix(
        self, hierarchy_info: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Build aggregation matrices for each hierarchy level.

        Args:
            hierarchy_info: Dictionary mapping level names to series indices

        Returns:
            Dictionary of aggregation matrices
        """
        num_series = len(hierarchy_info[self.hierarchy_levels[-1]])
        matrices = {}

        for level in self.hierarchy_levels[:-1]:  # Exclude item level
            unique_groups = np.unique(hierarchy_info[level])
            matrix = np.zeros((len(unique_groups), num_series))

            for i, group in enumerate(unique_groups):
                mask = hierarchy_info[level] == group
                matrix[i, mask] = 1.0

            matrices[level] = matrix
            logger.debug(f"Built aggregation matrix for {level}: {matrix.shape}")

        self.aggregation_matrices = matrices
        return matrices

    def aggregate_forecasts(
        self, base_forecasts: np.ndarray, level: str
    ) -> np.ndarray:
        """Aggregate base-level forecasts to a higher level.

        Args:
            base_forecasts: Base level forecasts of shape (num_series, time_steps)
            level: Target hierarchy level

        Returns:
            Aggregated forecasts
        """
        if level not in self.aggregation_matrices:
            raise ValueError(f"Unknown level: {level}")

        matrix = self.aggregation_matrices[level]
        return matrix @ base_forecasts

    def check_coherence(
        self, forecasts: Dict[str, np.ndarray]
    ) -> Tuple[bool, float]:
        """Check if forecasts are coherent across hierarchy.

        Args:
            forecasts: Dictionary mapping levels to forecast arrays

        Returns:
            Tuple of (is_coherent, violation_score)
        """
        violation_score = 0.0
        num_checks = 0

        base_forecasts = forecasts[self.hierarchy_levels[-1]]

        for level in self.hierarchy_levels[:-1]:
            expected = self.aggregate_forecasts(base_forecasts, level)
            actual = forecasts[level]

            diff = np.abs(expected - actual)
            violation_score += np.mean(diff)
            num_checks += 1

        avg_violation = violation_score / num_checks if num_checks > 0 else 0.0
        is_coherent = avg_violation < 0.01

        return is_coherent, avg_violation


class TimeSeriesPreprocessor:
    """Preprocess time series data.

    Args:
        normalize: Whether to normalize the data
        handle_missing: How to handle missing values ('drop', 'forward_fill', 'zero')
    """

    def __init__(
        self, normalize: bool = True, handle_missing: str = 'forward_fill'
    ) -> None:
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.scaler = StandardScaler() if normalize else None

        logger.info(
            f"Initialized TimeSeriesPreprocessor: "
            f"normalize={normalize}, handle_missing={handle_missing}"
        )

    def fit(self, data: np.ndarray) -> 'TimeSeriesPreprocessor':
        """Fit preprocessor on training data.

        Args:
            data: Training data of shape (num_series, time_steps)

        Returns:
            Self
        """
        if self.scaler is not None:
            # Fit on flattened data
            self.scaler.fit(data.reshape(-1, 1))
            logger.info("Fitted scaler on training data")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data.

        Args:
            data: Data to transform of shape (num_series, time_steps)

        Returns:
            Transformed data
        """
        # Handle missing values
        if np.isnan(data).any():
            logger.warning(f"Found {np.isnan(data).sum()} missing values")

            if self.handle_missing == 'zero':
                data = np.nan_to_num(data, nan=0.0)
            elif self.handle_missing == 'forward_fill':
                data = pd.DataFrame(data.T).ffill().fillna(0).values.T
            elif self.handle_missing == 'drop':
                raise ValueError("Cannot drop missing values in array format")

        # Normalize
        if self.scaler is not None:
            original_shape = data.shape
            data = self.scaler.transform(data.reshape(-1, 1)).reshape(original_shape)

        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data.

        Args:
            data: Transformed data

        Returns:
            Original scale data
        """
        if self.scaler is not None:
            original_shape = data.shape
            data = self.scaler.inverse_transform(
                data.reshape(-1, 1)
            ).reshape(original_shape)

        return data
