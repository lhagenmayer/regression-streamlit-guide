"""
Plot utility functions for the Linear Regression Guide.

This module contains helper functions for plot styling and data preparation.
"""

from typing import Optional, Union, List
import numpy as np


def get_signif_stars(p: float) -> str:
    """Signifikanz-Codes wie in R"""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return " "


def get_signif_color(p: float) -> str:
    """Farbe basierend auf Signifikanz"""
    if p < 0.001:
        return "#006400"
    if p < 0.01:
        return "#228B22"
    if p < 0.05:
        return "#32CD32"
    if p < 0.1:
        return "#FFA500"
    return "#DC143C"


def calculate_residual_sizes(residuals: np.ndarray, base_size: float = 3, scale_factor: float = 5) -> np.ndarray:
    """
    Calculate residual marker sizes for visualization.

    Args:
        residuals: Array of residual values
        base_size: Base marker size
        scale_factor: Scaling factor for size variation

    Returns:
        Array of marker sizes
    """
    if len(residuals) == 0:
        return np.array([])

    # Normalize residuals to [0, 1] range
    abs_residuals = np.abs(residuals)
    if np.max(abs_residuals) > 0:
        normalized = abs_residuals / np.max(abs_residuals)
    else:
        normalized = np.zeros_like(abs_residuals)

    # Calculate sizes (base_size to base_size + scale_factor)
    sizes = base_size + normalized * scale_factor
    return sizes


def standardize_residuals(residuals: np.ndarray) -> np.ndarray:
    """
    Standardize residuals for better visualization.

    Args:
        residuals: Array of residual values

    Returns:
        Standardized residuals
    """
    if len(residuals) == 0:
        return residuals

    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals, ddof=1)

    if std_resid > 0:
        return (residuals - mean_resid) / std_resid
    else:
        return residuals - mean_resid