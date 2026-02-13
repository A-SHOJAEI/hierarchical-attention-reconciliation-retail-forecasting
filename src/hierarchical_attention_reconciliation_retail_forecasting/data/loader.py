"""Data loading utilities for M5 forecasting dataset."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class M5Dataset(Dataset):
    """PyTorch Dataset for M5 forecasting data.

    Args:
        data: Time series data array of shape (num_series, time_steps)
        hierarchy_info: Hierarchy level information for each series
        sequence_length: Input sequence length
        prediction_length: Output prediction length
        quantiles: Quantiles for probabilistic forecasting
    """

    def __init__(
        self,
        data: np.ndarray,
        hierarchy_info: Dict[str, np.ndarray],
        sequence_length: int,
        prediction_length: int,
        quantiles: List[float],
    ) -> None:
        self.data = data
        self.hierarchy_info = hierarchy_info
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles

        # Calculate valid sequence positions
        self.num_series = data.shape[0]
        self.max_idx = data.shape[1] - sequence_length - prediction_length

        if self.max_idx < 0:
            raise ValueError(
                f"Time series too short: {data.shape[1]} < "
                f"{sequence_length + prediction_length}"
            )

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.num_series * self.max_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing input sequence, target, and hierarchy info
        """
        series_idx = idx // self.max_idx
        time_idx = idx % self.max_idx

        # Extract sequence
        x = self.data[series_idx, time_idx:time_idx + self.sequence_length]
        y = self.data[
            series_idx,
            time_idx + self.sequence_length:time_idx + self.sequence_length + self.prediction_length
        ]

        # Get hierarchy level for this series
        hierarchy_level = np.zeros(4, dtype=np.float32)
        for i, level in enumerate(['store', 'department', 'category', 'item']):
            if level in self.hierarchy_info:
                hierarchy_level[i] = self.hierarchy_info[level][series_idx]

        return {
            'x': torch.FloatTensor(x).unsqueeze(-1),
            'y': torch.FloatTensor(y).unsqueeze(-1),
            'hierarchy_level': torch.FloatTensor(hierarchy_level),
            'series_idx': torch.LongTensor([series_idx]),
        }


class M5DataLoader:
    """Loader for M5 forecasting dataset with hierarchy structure.

    Args:
        data_dir: Directory to store/load data
        sequence_length: Input sequence length
        prediction_length: Output prediction length
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        sequence_length: int = 28,
        prediction_length: int = 28,
    ) -> None:
        self.data_dir = data_dir or Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        logger.info(f"Initialized M5DataLoader with data_dir={self.data_dir}")

    def generate_synthetic_data(
        self, num_series: int = 100, time_steps: int = 365
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate synthetic M5-like data for testing.

        Args:
            num_series: Number of time series
            time_steps: Number of time steps

        Returns:
            Tuple of (data array, hierarchy info dict)
        """
        logger.info(f"Generating synthetic data: {num_series} series, {time_steps} steps")

        # Generate synthetic sales data with trend and seasonality
        np.random.seed(42)
        t = np.arange(time_steps)

        data = []
        for i in range(num_series):
            # Trend
            trend = 10 + 0.01 * i * t
            # Weekly seasonality
            seasonality = 5 * np.sin(2 * np.pi * t / 7)
            # Noise
            noise = np.random.normal(0, 2, time_steps)
            # Combine
            series = np.maximum(0, trend + seasonality + noise)
            data.append(series)

        data = np.array(data)

        # Create hierarchy structure
        hierarchy_info = {
            'store': np.random.randint(0, 10, num_series),
            'department': np.random.randint(0, 7, num_series),
            'category': np.random.randint(0, 3, num_series),
            'item': np.arange(num_series),
        }

        return data, hierarchy_info

    def load_data(
        self, use_synthetic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load M5 dataset or generate synthetic data.

        Args:
            use_synthetic: If True, generate synthetic data

        Returns:
            Tuple of (data array, hierarchy info dict)
        """
        if use_synthetic:
            return self.generate_synthetic_data()
        else:
            raise NotImplementedError(
                "Real M5 dataset loading not implemented. "
                "Set use_synthetic=True to generate synthetic data."
            )


def create_dataloaders(
    config: Dict,
    use_synthetic: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary
        use_synthetic: Whether to use synthetic data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating dataloaders")

    # Load data
    loader = M5DataLoader(
        sequence_length=config['data']['sequence_length'],
        prediction_length=config['data']['prediction_length'],
    )
    data, hierarchy_info = loader.load_data(use_synthetic=use_synthetic)

    # Split data
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']

    split_idx1 = int(data.shape[1] * train_split)
    split_idx2 = int(data.shape[1] * (train_split + val_split))

    train_data = data[:, :split_idx1]
    val_data = data[:, :split_idx2]
    test_data = data

    logger.info(
        f"Data splits - Train: {train_data.shape}, "
        f"Val: {val_data.shape}, Test: {test_data.shape}"
    )

    # Create datasets
    quantiles = config['model'].get('quantiles', [0.1, 0.5, 0.9])

    train_dataset = M5Dataset(
        train_data,
        hierarchy_info,
        config['data']['sequence_length'],
        config['data']['prediction_length'],
        quantiles,
    )
    val_dataset = M5Dataset(
        val_data,
        hierarchy_info,
        config['data']['sequence_length'],
        config['data']['prediction_length'],
        quantiles,
    )
    test_dataset = M5Dataset(
        test_data,
        hierarchy_info,
        config['data']['sequence_length'],
        config['data']['prediction_length'],
        quantiles,
    )

    # Create dataloaders
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Created dataloaders - Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
