"""
Plotting functions for the Linear Regression Guide.

This module provides a unified interface to all plotting functions,
organized into specialized sub-modules for better maintainability.
"""

# Re-export all plotting functions from specialized modules
from .plot_utils import (
    get_signif_stars,
    get_signif_color,
    calculate_residual_sizes,
    standardize_residuals
)

from .plot_basic import (
    create_plotly_scatter,
    create_plotly_scatter_with_line,
    create_plotly_residual_plot,
    create_plotly_bar,
    create_plotly_distribution
)

from .plot_3d import (
    create_regression_mesh,
    get_3d_layout_config,
    create_zero_plane,
    create_plotly_3d_scatter,
    create_plotly_3d_surface
)

from .plot_r import (
    create_r_output_display,
    create_r_output_figure
)

# Keep backward compatibility by importing all functions
__all__ = [
    # Utils
    'get_signif_stars',
    'get_signif_color',
    'calculate_residual_sizes',
    'standardize_residuals',

    # Basic plots
    'create_plotly_scatter',
    'create_plotly_scatter_with_line',
    'create_plotly_residual_plot',
    'create_plotly_bar',
    'create_plotly_distribution',

    # 3D plots
    'create_regression_mesh',
    'get_3d_layout_config',
    'create_zero_plane',
    'create_plotly_3d_scatter',
    'create_plotly_3d_surface',

    # R-style plots
    'create_r_output_display',
    'create_r_output_figure',
]