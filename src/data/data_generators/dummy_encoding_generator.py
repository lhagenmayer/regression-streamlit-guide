"""
Dummy encoding data generator for the Linear Regression Guide.

This module provides functions for generating datasets with categorical variables
that need dummy encoding for regression analysis.
"""

from typing import Dict, Union, Any, List
import numpy as np
import pandas as pd

from ...config import get_logger

logger = get_logger(__name__)


def create_dummy_encoded_dataset(
    n_obs: int = 100,
    categories: List[str] = None,
    effects: Dict[str, float] = None,
    noise_level: float = 5.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create a dataset with categorical variables for dummy encoding demonstration.

    Args:
        n_obs: Number of observations
        categories: List of category names
        effects: Dictionary mapping categories to their effects
        noise_level: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with original and dummy-encoded data
    """
    if categories is None:
        categories = ["A", "B", "C"]

    if effects is None:
        effects = {"A": 10.0, "B": 15.0, "C": 8.0}

    np.random.seed(seed)

    # Generate categorical variable
    category_data = np.random.choice(categories, size=n_obs)

    # Generate continuous predictor
    x_continuous = np.random.normal(50, 10, n_obs)

    # Generate response variable
    y = 20 + 0.5 * x_continuous  # base relationship

    # Add categorical effects
    for i, cat in enumerate(category_data):
        y[i] += effects.get(cat, 0)

    # Add noise
    y += np.random.normal(0, noise_level, n_obs)

    # Create dummy variables
    dummy_data = {}
    for cat in categories[1:]:  # Skip first category (reference)
        dummy_data[f"dummy_{cat.lower()}"] = (category_data == cat).astype(int)

    return {
        "category": category_data,
        "x_continuous": x_continuous,
        "y": y,
        "categories": categories,
        "effects": effects,
        "dummy_variables": dummy_data,
        "reference_category": categories[0]
    }


def generate_categorical_regression_data(
    n_obs: int = 200,
    n_categories: int = 3,
    n_continuous: int = 2,
    interaction_effects: bool = False,
    noise_level: float = 3.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate comprehensive categorical regression data.

    Args:
        n_obs: Number of observations
        n_categories: Number of categories
        n_continuous: Number of continuous predictors
        interaction_effects: Whether to include category-continuous interactions
        noise_level: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with complete dataset
    """
    np.random.seed(seed)

    # Generate categorical variable
    categories = [f"Cat_{i}" for i in range(n_categories)]
    category_data = np.random.choice(categories, size=n_obs)

    # Generate continuous predictors
    continuous_data = {}
    for i in range(n_continuous):
        var_name = f"x_cont_{i}"
        continuous_data[var_name] = np.random.normal(50, 15, n_obs)

    # Generate response variable
    y = 25.0  # intercept

    # Add continuous effects
    for i, (name, data) in enumerate(continuous_data.items()):
        coeff = 0.8 + i * 0.3  # Different coefficients
        y += coeff * data

    # Add categorical effects
    category_effects = {}
    for cat in categories:
        effect = np.random.normal(0, 5)  # Random effects
        category_effects[cat] = effect

    for i, cat in enumerate(category_data):
        y[i] += category_effects[cat]

    # Add interaction effects if requested
    if interaction_effects and n_continuous > 0:
        for cat in categories[1:]:  # Skip reference
            for cont_name, cont_data in continuous_data.items():
                interaction_effect = np.random.normal(0, 0.5)
                mask = category_data == cat
                y[mask] += interaction_effect * cont_data[mask]

    # Add noise
    y += np.random.normal(0, noise_level, n_obs)

    # Create dummy variables
    dummy_vars = {}
    for cat in categories[1:]:
        dummy_vars[f"dummy_{cat.lower()}"] = (category_data == cat).astype(int)

    return {
        "category": category_data,
        "continuous": continuous_data,
        "y": y,
        "categories": categories,
        "category_effects": category_effects,
        "continuous_coefficients": [0.8 + i * 0.3 for i in range(n_continuous)],
        "dummy_variables": dummy_vars,
        "reference_category": categories[0],
        "has_interactions": interaction_effects
    }