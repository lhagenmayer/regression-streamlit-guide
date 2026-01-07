"""
Tab components for the Linear Regression Guide application.

This package contains modular tab implementations for better code organization:
- simple_regression.py: Simple linear regression tab
- multiple_regression.py: Multiple regression tab  
- datasets.py: Datasets overview tab
"""

from .simple_regression import render_simple_regression_tab
from .multiple_regression import render_multiple_regression_tab
from .datasets import render_datasets_tab

__all__ = [
    "render_simple_regression_tab",
    "render_multiple_regression_tab",
    "render_datasets_tab",
]
