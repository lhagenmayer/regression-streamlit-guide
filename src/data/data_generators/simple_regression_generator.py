"""
Simple regression data generator for the Linear Regression Guide.

This module provides functions for generating synthetic datasets
for simple regression analysis demonstrations.
"""

from typing import Dict, Union, Any
import numpy as np
import pandas as pd

from ...config import get_logger
from . import BaseDataGenerator

logger = get_logger(__name__)


class SimpleRegressionGenerator(BaseDataGenerator):
    """Generator for simple regression datasets."""

    def generate(
        self,
        dataset_choice: str,
        n_simple: int,
        noise_simple_level: float,
        seed_simple: int
    ) -> Dict[str, Union[np.ndarray, str]]:
        """
        Generate data for simple regression based on dataset choice.

        Args:
            dataset_choice: Name of the dataset
            n_simple: Number of observations
            noise_simple_level: Noise standard deviation
            seed_simple: Random seed

        Returns:
            Dictionary with x_simple, y_simple, x_name, y_name
        """
        # Check cache first
        cache_key = self._get_cache_key(
            dataset_choice=dataset_choice,
            n_simple=n_simple,
            noise_simple_level=noise_simple_level,
            seed_simple=seed_simple
        )

        if self._is_cached(cache_key):
            logger.debug("Returning cached simple regression data")
            return self._get_cached_result(cache_key)

        # Validate inputs using base class method
        self._validate_common_params(n_simple, noise_simple_level, seed_simple)

        logger.info(
            f"Generating simple regression data: dataset={dataset_choice}, n={n_simple}, noise={noise_simple_level}, seed={seed_simple}"
        )
        start_time = __import__('time').time()

        # Set random seed for reproducibility
        np.random.seed(seed_simple)

        if dataset_choice == "study_hours":
            # Study hours vs exam scores
            x_simple = np.random.uniform(0, 40, n_simple)  # hours studied
            y_simple = 50 + 1.5 * x_simple + np.random.normal(0, noise_simple_level, n_simple)
            x_name = "Study Hours"
            y_name = "Exam Score"

        elif dataset_choice == "temperature":
            # Temperature vs ice cream sales
            x_simple = np.random.uniform(15, 35, n_simple)  # temperature in Celsius
            y_simple = 20 + 3.0 * x_simple + np.random.normal(0, noise_simple_level, n_simple)
            x_name = "Temperature (°C)"
            y_name = "Ice Cream Sales"

        elif dataset_choice == "advertising":
            # Advertising spend vs sales
            x_simple = np.random.uniform(1000, 10000, n_simple)  # advertising spend
            y_simple = 50000 + 5.0 * x_simple + np.random.normal(0, noise_simple_level, n_simple)
            x_name = "Advertising Spend ($)"
            y_name = "Sales ($)"

        else:
            # Default: generic linear relationship
            x_simple = np.random.uniform(0, 100, n_simple)
            y_simple = 10 + 2.0 * x_simple + np.random.normal(0, noise_simple_level, n_simple)
            x_name = "X Variable"
            y_name = "Y Variable"

        result = {
            "x_simple": x_simple,
            "y_simple": y_simple,
            "x_name": x_name,
            "y_name": y_name
        }

        end_time = __import__('time').time()
        logger.info(".2f")

        # Cache the result
        self._cache_result(cache_key, result)

        return result

    def generate_custom(
        self,
        n_obs: int,
        slope: float,
        intercept: float,
        noise_level: float,
        seed: int,
        x_range: tuple = (0, 100)
    ) -> Dict[str, Union[np.ndarray, str]]:
        """
        Generate custom simple regression data with user-specified parameters.

        Args:
            n_obs: Number of observations
            slope: Slope coefficient
            intercept: Intercept coefficient
            noise_level: Standard deviation of noise
            seed: Random seed
            x_range: Range for predictor variable (min, max)

        Returns:
            Dictionary with predictor and response variables
        """
        # Check cache first
        cache_key = self._get_cache_key(
            n_obs=n_obs, slope=slope, intercept=intercept,
            noise_level=noise_level, seed=seed, x_range=x_range
        )

        if self._is_cached(cache_key):
            logger.debug("Returning cached custom simple regression data")
            return self._get_cached_result(cache_key)

        # Validate common parameters
        self._validate_common_params(n_obs, noise_level, seed)

        np.random.seed(seed)

        # Generate predictor
        x = np.random.uniform(x_range[0], x_range[1], n_obs)

        # Generate response
        y = intercept + slope * x + np.random.normal(0, noise_level, n_obs)

        result = {
            "x": x,
            "y": y,
            "x_name": "Predictor",
            "y_name": "Response"
        }

        # Cache the result
        self._cache_result(cache_key, result)

        return result


# Create default instance for backward compatibility
_default_generator = SimpleRegressionGenerator()


def generate_simple_regression_data(
    dataset_choice: str, n_simple: int, noise_simple_level: float, seed_simple: int
) -> Dict[str, Union[np.ndarray, str]]:
    """Backward compatibility function."""
    return _default_generator.generate(
        dataset_choice=dataset_choice,
        n_simple=n_simple,
        noise_simple_level=noise_simple_level,
        seed_simple=seed_simple
    )


def generate_custom_simple_regression_data(
    n_obs: int,
    slope: float,
    intercept: float,
    noise_level: float,
    seed: int,
    x_range: tuple = (0, 100)
) -> Dict[str, Union[np.ndarray, str]]:
    """Backward compatibility function."""
    return _default_generator.generate_custom(
        n_obs=n_obs,
        slope=slope,
        intercept=intercept,
        noise_level=noise_level,
        seed=seed,
        x_range=x_range
    )