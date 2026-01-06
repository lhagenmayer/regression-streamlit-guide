"""
Multiple regression data generator for the Linear Regression Guide.

This module provides functions for generating synthetic datasets
for multiple regression analysis demonstrations.
"""

from typing import Dict, Union, Any
import numpy as np
import pandas as pd
import streamlit as st

from ...config import CITIES_DATASET, HOUSES_DATASET
from ...config import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_multiple_regression_data(
    dataset_choice_mult: str, n_mult: int, noise_mult_level: float, seed_mult: int
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Generate data for multiple regression based on dataset choice.

    Args:
        dataset_choice_mult: Name of the dataset
        n_mult: Number of observations
        noise_mult_level: Noise standard deviation
        seed_mult: Random seed

    Returns:
        Dictionary with x2_preis, x3_werbung, y_mult, x1_name, x2_name, y_name
    """
    # Validate inputs
    if not isinstance(n_mult, int) or n_mult <= 0:
        raise ValueError(f"Sample size n_mult must be a positive integer, got {n_mult}")

    if not isinstance(seed_mult, int):
        raise ValueError(f"Seed seed_mult must be an integer, got {seed_mult}")

    if not isinstance(noise_mult_level, (int, float)) or noise_mult_level < 0:
        raise ValueError(f"Noise level noise_mult_level must be a non-negative number, got {noise_mult_level}")

    logger.info(
        f"Generating multiple regression data: dataset={dataset_choice_mult}, n={n_mult}, noise={noise_mult_level}, seed={seed_mult}"
    )
    start_time = __import__('time').time()

    # Set random seed for reproducibility
    np.random.seed(seed_mult)

    if dataset_choice_mult == "Cities":
        # Generate cities dataset
        # x1: population (in 1000s)
        x1_population = np.random.normal(500, 150, n_mult)
        x1_population = np.clip(x1_population, 50, 2000)  # reasonable bounds

        # x2: average income (in 1000s)
        x2_income = np.random.normal(50, 15, n_mult)
        x2_income = np.clip(x2_income, 20, 150)  # reasonable bounds

        # x3: unemployment rate (percentage)
        x3_unemployment = np.random.normal(5, 2, n_mult)
        x3_unemployment = np.clip(x3_unemployment, 1, 15)  # reasonable bounds

        # True relationship: crime_rate = 20 + 0.02*population + 0.5*income - 2*unemployment + noise
        true_crime_rate = (
            20 +
            0.02 * x1_population +
            0.5 * x2_income +
            -2 * x3_unemployment
        )

        # Add noise
        noise = np.random.normal(0, noise_mult_level, n_mult)
        y_crime = true_crime_rate + noise
        y_crime = np.maximum(y_crime, 0)  # crime rate can't be negative

        # Variable names for display
        x1_name = "Population (1000s)"
        x2_name = "Avg Income (1000s)"
        y_name = "Crime Rate"

        result = {
            "x1_population": x1_population,
            "x2_income": x2_income,
            "x3_unemployment": x3_unemployment,
            "y_crime": y_crime,
            "x1_name": x1_name,
            "x2_name": x2_name,
            "x3_name": "Unemployment Rate (%)",
            "y_name": y_name
        }

    elif dataset_choice_mult == "Houses":
        # Generate houses dataset
        # x1: size (sq ft)
        x1_size = np.random.normal(2000, 500, n_mult)
        x1_size = np.clip(x1_size, 800, 5000)  # reasonable bounds

        # x2: bedrooms
        x2_bedrooms = np.random.normal(3, 0.8, n_mult)
        x2_bedrooms = np.round(np.clip(x2_bedrooms, 1, 6))  # integer bedrooms

        # x3: age (years)
        x3_age = np.random.normal(20, 10, n_mult)
        x3_age = np.clip(x3_age, 0, 100)  # reasonable bounds

        # True relationship: price = 50000 + 50*size + 20000*bedrooms - 1000*age + noise
        true_price = (
            50000 +
            50 * x1_size +
            20000 * x2_bedrooms +
            -1000 * x3_age
        )

        # Add noise
        noise = np.random.normal(0, noise_mult_level, n_mult)
        y_price = true_price + noise
        y_price = np.maximum(y_price, 20000)  # minimum house price

        # Variable names for display
        x1_name = "Size (sq ft)"
        x2_name = "Bedrooms"
        y_name = "Price ($)"

        result = {
            "x1_size": x1_size,
            "x2_bedrooms": x2_bedrooms,
            "x3_age": x3_age,
            "y_price": y_price,
            "x1_name": x1_name,
            "x2_name": x2_name,
            "x3_name": "Age (years)",
            "y_name": y_name
        }

    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice_mult}")

    end_time = __import__('time').time()
    logger.info(".2f")

    return result


def generate_custom_multiple_regression_data(
    n_obs: int,
    coefficients: list,
    noise_level: float,
    seed: int,
    x_ranges: list = None
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Generate custom multiple regression data with user-specified coefficients.

    Args:
        n_obs: Number of observations
        coefficients: List of coefficients [intercept, beta1, beta2, ...]
        noise_level: Standard deviation of noise
        seed: Random seed
        x_ranges: Optional ranges for predictors [[min1, max1], [min2, max2], ...]

    Returns:
        Dictionary with predictors and response variable
    """
    if len(coefficients) < 2:
        raise ValueError("At least intercept and one coefficient required")

    np.random.seed(seed)
    n_predictors = len(coefficients) - 1

    # Set default ranges if not provided
    if x_ranges is None:
        x_ranges = [[0, 100] for _ in range(n_predictors)]

    # Generate predictors
    X = []
    for i, (min_val, max_val) in enumerate(x_ranges):
        x = np.random.uniform(min_val, max_val, n_obs)
        X.append(x)

    # Generate response variable
    y = np.full(n_obs, coefficients[0])  # intercept
    for i, coeff in enumerate(coefficients[1:], 1):
        y += coeff * X[i-1]

    # Add noise
    y += np.random.normal(0, noise_level, n_obs)

    result = {"y": y}
    for i, x in enumerate(X, 1):
        result[f"x{i}"] = x

    # Add variable names
    result.update({
        "y_name": "Response Variable",
        "x_names": [f"Predictor {i}" for i in range(1, n_predictors + 1)]
    })

    return result