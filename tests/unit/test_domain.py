"""
Unit tests for domain layer components.

Tests domain entities, value objects, and services in isolation.
"""

import unittest
from unittest.mock import Mock

from src.core.domain.entities import Dataset, RegressionModel
from src.core.domain.value_objects import DatasetConfig, RegressionParameters, ModelMetrics
from src.core.domain.services import RegressionAnalysisService


class TestDataset(unittest.TestCase):
    """Test Dataset entity."""

    def test_dataset_creation(self):
        """Test creating a dataset."""
        config = DatasetConfig(
            name="test_dataset",
            dataset_type="synthetic",
            source="test",
            description="Test dataset",
            variables={"x": "predictor", "y": "response"},
            n_observations=100
        )

        data = {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}
        dataset = Dataset(id="test_id", config=config, data=data)

        self.assertEqual(dataset.id, "test_id")
        self.assertEqual(dataset.config.name, "test_dataset")
        self.assertEqual(len(dataset.data), 2)

    def test_dataset_validation(self):
        """Test dataset validation."""
        config = DatasetConfig(
            name="test_dataset",
            dataset_type="synthetic",
            source="test",
            description="Test dataset",
            variables={"x": "predictor", "y": "response"},
            n_observations=3
        )

        # Valid dataset
        data = {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}
        dataset = Dataset(id="test_id", config=config, data=data)
        issues = dataset.validate()
        self.assertEqual(len(issues), 0)

        # Invalid dataset - inconsistent lengths
        data_invalid = {"x": [1.0, 2.0], "y": [2.0, 4.0, 6.0]}
        dataset_invalid = Dataset(id="test_id", config=config, data=data_invalid)
        issues = dataset_invalid.validate()
        self.assertGreater(len(issues), 0)


class TestValueObjects(unittest.TestCase):
    """Test value objects."""

    def test_regression_parameters_immutable(self):
        """Test that RegressionParameters is immutable."""
        params = RegressionParameters(
            intercept=1.0,
            coefficients={"x": 2.0},
            noise_level=0.1,
            seed=42
        )

        with self.assertRaises(AttributeError):
            params.intercept = 2.0

    def test_model_metrics_validation(self):
        """Test ModelMetrics creation."""
        metrics = ModelMetrics(
            r_squared=0.8,
            adj_r_squared=0.75,
            mse=1.0,
            rmse=1.0,
            mae=0.8,
            f_statistic=20.0,
            f_p_value=0.01
        )

        self.assertEqual(metrics.r_squared, 0.8)
        self.assertTrue(metrics.is_significant())


class TestRegressionAnalysisService(unittest.TestCase):
    """Test RegressionAnalysisService."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_repo = Mock()
        self.service = RegressionAnalysisService(self.mock_repo)

    def test_create_regression_model_valid_dataset(self):
        """Test creating a regression model with valid dataset."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.validate.return_value = []
        mock_dataset.get_variable.side_effect = lambda name: {
            "y": [2.0, 4.0, 6.0],
            "x": [1.0, 2.0, 3.0]
        }[name]

        self.mock_repo.find_by_id.return_value = mock_dataset

        # Test parameters
        params = RegressionParameters(
            intercept=0.0,
            coefficients={"x": 2.0},
            noise_level=0.0,
            seed=42
        )

        model = self.service.create_regression_model(
            dataset_id="test_dataset",
            target_variable="y",
            feature_variables=["x"],
            parameters=params
        )

        self.assertIsInstance(model, RegressionModel)
        self.assertEqual(model.model_type, "simple")

    def test_create_regression_model_invalid_dataset(self):
        """Test creating a regression model with invalid dataset."""
        self.mock_repo.find_by_id.return_value = None

        params = RegressionParameters(
            intercept=0.0,
            coefficients={"x": 2.0},
            noise_level=0.0,
            seed=42
        )

        with self.assertRaises(ValueError):
            self.service.create_regression_model(
                dataset_id="nonexistent",
                target_variable="y",
                feature_variables=["x"],
                parameters=params
            )


if __name__ == '__main__':
    unittest.main()