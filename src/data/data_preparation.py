"""
Data preparation module for the Linear Regression Guide.

This module provides a simplified interface to the data and model services,
maintaining backward compatibility while delegating to the refactored services.
"""

from typing import Dict, Any
import numpy as np


def prepare_multiple_regression_data(
    dataset_choice: str,
    n: int,
    noise_level: float,
    seed: int
) -> Dict[str, Any]:
    """
    Prepare data for multiple regression with caching.

    Args:
        dataset_choice: Name of the selected dataset
        n: Number of observations
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing prepared data and model results
    """
    # Lazy import to avoid circular dependencies
    from ..core.services import DataService
    return DataService.load_multiple_regression_data(dataset_choice, n, noise_level, seed)


def prepare_simple_regression_data(
    dataset_choice: str,
    x_variable: str = None,
    n: int = 12,
    true_intercept: float = 0.6,
    true_beta: float = 0.52,
    noise_level: float = 0.4,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Prepare data for simple regression with caching.

    Args:
        dataset_choice: Name of the selected dataset
        x_variable: Selected X variable (for non-simulated datasets)
        n: Number of observations
        true_intercept: True intercept value (for simulated data)
        true_beta: True slope value (for simulated data)
        noise_level: Standard deviation of noise (for simulated data)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing prepared data
    """
    # Lazy import to avoid circular dependencies
    from ..core.services import DataService
    return DataService.load_simple_regression_data(
        dataset_choice, x_variable, n, true_intercept, true_beta, noise_level, seed
    )


def compute_simple_model(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str
) -> Dict[str, Any]:
    """
    Compute simple regression model with caching.

    Args:
        x: Predictor variable
        y: Response variable
        x_label: Label for X variable
        y_label: Label for Y variable

    Returns:
        Dictionary containing model and statistics
    """
    # Lazy import to avoid circular dependencies
    from ..core.services import DataService
    return DataService.prepare_simple_regression_model_data(x, y, x_label, y_label)
