"""
Pytest configuration and shared fixtures for Pipeline tests.
"""

import pytest
import numpy as np

from src.pipeline.get_data import DataFetcher, DataResult
from src.pipeline.calculate import StatisticsCalculator


@pytest.fixture
def data_fetcher():
    """Provide a DataFetcher instance."""
    return DataFetcher()


@pytest.fixture
def calculator():
    """Provide a StatisticsCalculator instance."""
    return StatisticsCalculator()


@pytest.fixture
def simple_data():
    """Provide simple regression test data."""
    np.random.seed(42)
    n = 50
    x = np.random.uniform(10, 100, n)
    y = 5 + 2 * x + np.random.normal(0, 10, n)
    
    return DataResult(
        x=x,
        y=y,
        x_label="X",
        y_label="Y",
        context_title="Test Data",
        context_description="Test dataset for unit tests"
    )
