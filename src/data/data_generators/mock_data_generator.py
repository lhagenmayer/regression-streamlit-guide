"""
Mock data utilities and helper functions.

This module provides utility functions for data manipulation and conversion.
"""

from typing import Union
import numpy as np
import pandas as pd


def safe_scalar(val: Union[pd.Series, np.ndarray, float, int]) -> float:
    """
    Convert Series/ndarray to scalar, if necessary.

    Args:
        val: Input value that might be a Series, ndarray, or scalar

    Returns:
        Float scalar value
    """
    if isinstance(val, (pd.Series, np.ndarray)):
        return float(val.iloc[0] if hasattr(val, "iloc") else val[0])
    return float(val)