"""
Statistical modeling functions for the Linear Regression Guide.

This module contains functions for fitting statistical models using external libraries.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Any
from functools import lru_cache
from ..config.config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=128)
def execute_ols_regression_modeling(
    X: np.ndarray, y: np.ndarray
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, np.ndarray]:
    """
    Execute OLS regression model fitting with caching.

    Args:
        X: Design matrix (with constant column)
        y: Response variable

    Returns:
        Tuple of (fitted model, predictions)
    """
    logger.debug(f"Executing OLS regression modeling with X shape {X.shape}, y shape {y.shape}")
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    return model, predictions


@lru_cache(maxsize=128)
def execute_multiple_ols_regression_modeling(
    X: np.ndarray, y: np.ndarray
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, np.ndarray]:
    """
    Execute multiple OLS regression model fitting with caching.

    Args:
        X: Design matrix (with constant column for multiple predictors)
        y: Response variable

    Returns:
        Tuple of (fitted model, predictions)
    """
    logger.debug(f"Executing multiple OLS regression modeling with X shape {X.shape}, y shape {y.shape}")
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    return model, predictions


def create_design_matrix(*columns: np.ndarray) -> np.ndarray:
    """
    Create design matrix from column arrays.

    Args:
        *columns: Variable arrays to include in design matrix

    Returns:
        Design matrix with intercept column
    """
    if not columns:
        raise ValueError("At least one column must be provided")

    # Create design matrix starting with intercept column
    X = np.ones((len(columns[0]), 1))

    # Add each variable column
    for col in columns:
        X = np.column_stack([X, col])

    return X