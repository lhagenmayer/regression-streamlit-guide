"""
Tab components for the Linear Regression Guide application.

This package contains modular tab implementations:
- simple_regression_educational.py: Full educational content for simple regression
- multiple_regression_educational.py: Full educational content for multiple regression
- datasets.py: Datasets overview tab

Legacy modules (for backward compatibility):
- simple_regression.py: Original detailed tab
- multiple_regression.py: Original detailed tab
"""

# New educational tabs (for Pipeline integration)
from .simple_regression_educational import render_simple_regression_educational
from .multiple_regression_educational import render_multiple_regression_educational

# Legacy tabs (backward compatibility)
from .simple_regression import render_simple_regression_tab
from .multiple_regression import render_multiple_regression_tab
from .datasets import render_datasets_tab

__all__ = [
    # New educational tabs
    "render_simple_regression_educational",
    "render_multiple_regression_educational",
    # Legacy tabs
    "render_simple_regression_tab",
    "render_multiple_regression_tab",
    "render_datasets_tab",
]
