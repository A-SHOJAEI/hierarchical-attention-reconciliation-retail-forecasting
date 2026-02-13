"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from hierarchical_attention_reconciliation_retail_forecasting.data.loader import (
    M5Dataset,
    M5DataLoader,
    create_dataloaders,
)
from hierarchical_attention_reconciliation_retail_forecasting.data.preprocessing import (
    HierarchyProcessor,
    TimeSeriesPreprocessor,
)


class TestM5DataLoader:
    """Test M5DataLoader class."""

    def test_initialization(self):
        """Test loader initialization."""
        loader = M5DataLoader(sequence_length=10, prediction_length=5)
        assert loader.sequence_length == 10
        assert loader.prediction_length == 5

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = M5DataLoader()
        data, hierarchy_info = loader.generate_synthetic_data(
            num_series=50, time_steps=100
        )

        assert data.shape == (50, 100)
        assert 'store' in hierarchy_info
        assert 'department' in hierarchy_info
        assert 'category' in hierarchy_info
        assert 'item' in hierarchy_info
        assert len(hierarchy_info['store']) == 50

    def test_load_data(self):
        """Test data loading."""
        loader = M5DataLoader()
        data, hierarchy_info = loader.load_data(use_synthetic=True)

        assert data.shape[0] > 0
        assert data.shape[1] > 0
        assert len(hierarchy_info) > 0


class TestM5Dataset:
    """Test M5Dataset class."""

    def test_dataset_creation(self, sample_time_series, sample_hierarchy_info):
        """Test dataset creation."""
        dataset = M5Dataset(
            data=sample_time_series,
            hierarchy_info=sample_hierarchy_info,
            sequence_length=10,
            prediction_length=5,
            quantiles=[0.1, 0.5, 0.9],
        )

        assert len(dataset) > 0

    def test_dataset_getitem(self, sample_time_series, sample_hierarchy_info):
        """Test dataset item retrieval."""
        dataset = M5Dataset(
            data=sample_time_series,
            hierarchy_info=sample_hierarchy_info,
            sequence_length=10,
            prediction_length=5,
            quantiles=[0.1, 0.5, 0.9],
        )

        sample = dataset[0]

        assert 'x' in sample
        assert 'y' in sample
        assert 'hierarchy_level' in sample
        assert sample['x'].shape == (10, 1)
        assert sample['y'].shape == (5, 1)

    def test_dataset_too_short(self, sample_hierarchy_info):
        """Test dataset with too short time series."""
        short_data = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(ValueError):
            M5Dataset(
                data=short_data,
                hierarchy_info=sample_hierarchy_info,
                sequence_length=10,
                prediction_length=5,
                quantiles=[0.1, 0.5, 0.9],
            )


class TestCreateDataloaders:
    """Test dataloader creation."""

    def test_create_dataloaders(self, sample_config):
        """Test dataloader creation."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_config, use_synthetic=True
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_dataloader_batch(self, sample_config):
        """Test dataloader batch retrieval."""
        train_loader, _, _ = create_dataloaders(sample_config, use_synthetic=True)

        batch = next(iter(train_loader))

        assert 'x' in batch
        assert 'y' in batch
        assert 'hierarchy_level' in batch
        assert batch['x'].shape[0] == sample_config['data']['batch_size']


class TestHierarchyProcessor:
    """Test HierarchyProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = HierarchyProcessor(['store', 'department', 'category', 'item'])
        assert len(processor.hierarchy_levels) == 4

    def test_build_aggregation_matrix(self, sample_hierarchy_info):
        """Test aggregation matrix building."""
        processor = HierarchyProcessor(['store', 'department', 'category', 'item'])
        matrices = processor.build_aggregation_matrix(sample_hierarchy_info)

        assert 'store' in matrices
        assert 'department' in matrices
        assert 'category' in matrices
        assert 'item' not in matrices  # Base level has no aggregation

    def test_aggregate_forecasts(self, sample_hierarchy_info):
        """Test forecast aggregation."""
        processor = HierarchyProcessor(['store', 'department', 'category', 'item'])
        processor.build_aggregation_matrix(sample_hierarchy_info)

        num_series = 20
        time_steps = 10
        base_forecasts = np.random.randn(num_series, time_steps)

        aggregated = processor.aggregate_forecasts(base_forecasts, 'store')
        assert aggregated.shape[0] < num_series
        assert aggregated.shape[1] == time_steps


class TestTimeSeriesPreprocessor:
    """Test TimeSeriesPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TimeSeriesPreprocessor(normalize=True)
        assert preprocessor.normalize is True
        assert preprocessor.scaler is not None

    def test_fit_transform(self, sample_time_series):
        """Test fit and transform."""
        preprocessor = TimeSeriesPreprocessor(normalize=True)
        preprocessor.fit(sample_time_series)
        transformed = preprocessor.transform(sample_time_series)

        assert transformed.shape == sample_time_series.shape
        assert not np.allclose(transformed, sample_time_series)

    def test_inverse_transform(self, sample_time_series):
        """Test inverse transform."""
        preprocessor = TimeSeriesPreprocessor(normalize=True)
        preprocessor.fit(sample_time_series)
        transformed = preprocessor.transform(sample_time_series)
        recovered = preprocessor.inverse_transform(transformed)

        assert np.allclose(recovered, sample_time_series, rtol=1e-5)

    def test_handle_missing_values(self):
        """Test missing value handling."""
        data_with_nan = np.array([[1, 2, np.nan, 4], [5, np.nan, 7, 8]], dtype=np.float32)

        preprocessor = TimeSeriesPreprocessor(
            normalize=False, handle_missing='zero'
        )
        preprocessor.fit(data_with_nan)
        transformed = preprocessor.transform(data_with_nan)

        assert not np.isnan(transformed).any()
