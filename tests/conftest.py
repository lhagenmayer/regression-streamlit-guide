"""
Pytest configuration and shared fixtures for Linear Regression Guide tests.

This file contains:
- Pytest configuration and hooks
- Shared fixtures for common test data
- Test utilities and helpers
- Setup/teardown functionality
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root


@pytest.fixture(scope="session")
def sample_regression_data():
    """Generate sample regression data for testing."""
    np.random.seed(42)
    n = 50
    x = np.random.normal(10, 2, n)
    # Create linear relationship with some noise
    y = 2.5 * x + 5 + np.random.normal(0, 3, n)

    return {
        "x": x,
        "y": y,
        "n": n,
        "slope_true": 2.5,
        "intercept_true": 5,
        "noise_std": 3
    }


@pytest.fixture(scope="session")
def sample_multiple_regression_data():
    """Generate sample multiple regression data for testing."""
    np.random.seed(123)
    n = 100
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.normal(25, 5, n)
    # Create multiple regression relationship
    y = 1.5 * x1 - 0.8 * x2 + 10 + np.random.normal(0, 5, n)

    return {
        "x1": x1,
        "x2": x2,
        "y": y,
        "n": n,
        "coef_x1": 1.5,
        "coef_x2": -0.8,
        "intercept": 10,
        "noise_std": 5
    }


@pytest.fixture(scope="session")
def sample_swiss_canton_data():
    """Generate sample Swiss canton data for testing."""
    from src.data import generate_swiss_canton_regression_data
    return generate_swiss_canton_regression_data()


@pytest.fixture(scope="session")
def sample_config_data():
    """Return sample configuration data."""
    from src.config import CITIES_DATASET, HOUSES_DATASET, SIMPLE_REGRESSION

    return {
        "cities": CITIES_DATASET,
        "houses": HOUSES_DATASET,
        "simple": SIMPLE_REGRESSION
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    from unittest.mock import MagicMock
    logger = MagicMock()
    return logger


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for data-related tests."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test to ensure reproducibility."""
    np.random.seed(42)


# Pytest hooks for basic markers
def pytest_configure(config):
    """Configure pytest with simple markers for unit and integration tests."""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "streamlit: Tests using Streamlit AppTest framework")


def pytest_collection_modifyitems(config, items):
    """Tag integration tests based on file path."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)