"""
User interface package for the Linear Regression Guide.

This package contains plotting functions, UI components, and display logic.

Modules:
- plots: Unified plotting interface (re-exports from plot_basic, plot_3d, plot_r)
- sidebar: Sidebar component for parameter selection
- tabs: Tab rendering components (simple_regression, multiple_regression, datasets)
- accessibility: Accessibility utilities
- ui_config: UI configuration constants
- r_output: R-style statistical output display
"""

# Plotting functions
from .plots import (
    get_signif_stars,
    get_signif_color,
    create_plotly_scatter,
    create_plotly_scatter_with_line,
    create_plotly_residual_plot,
    create_plotly_bar,
    create_plotly_3d_surface,
    create_regression_mesh,
    create_r_output_figure,
)

# Tab renderers
from .tabs import (
    render_simple_regression_tab,
    render_multiple_regression_tab,
    render_datasets_tab,
)

# R output
from .r_output import create_r_output_display

__all__ = [
    # Plots
    'get_signif_stars',
    'get_signif_color',
    'create_plotly_scatter',
    'create_plotly_scatter_with_line',
    'create_plotly_residual_plot',
    'create_plotly_bar',
    'create_plotly_3d_surface',
    'create_regression_mesh',
    'create_r_output_figure',
    # Tabs
    'render_simple_regression_tab',
    'render_multiple_regression_tab',
    'render_datasets_tab',
    # R output
    'create_r_output_display',
]