"""
Basic plotting functions for the Linear Regression Guide.

This module contains fundamental 2D plotting functions.
"""

from typing import Optional, Union, List, Callable, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ..config import get_logger

logger = get_logger(__name__)


def create_plotly_scatter(
    x: np.ndarray,
    y: np.ndarray,
    name: str = "Data",
    mode: str = "markers",
    color: Optional[str] = None,
    size: Optional[Union[int, np.ndarray]] = None,
    opacity: float = 0.8,
    show_legend: bool = True
) -> go.Scatter:
    """
    Create a basic scatter plot.

    Args:
        x, y: Data arrays
        name: Legend name
        mode: Plot mode ("markers", "lines", "markers+lines")
        color: Point/line color
        size: Marker sizes
        opacity: Element opacity
        show_legend: Whether to show in legend

    Returns:
        Plotly Scatter object
    """
    marker_config = {"opacity": opacity}
    if color is not None:
        marker_config["color"] = color
    if size is not None:
        marker_config["size"] = size

    return go.Scatter(
        x=x, y=y,
        mode=mode,
        name=name,
        marker=marker_config,
        showlegend=show_legend
    )


def create_plotly_scatter_with_line(
    x: np.ndarray,
    y: np.ndarray,
    line_x: Optional[np.ndarray] = None,
    line_y: Optional[np.ndarray] = None,
    name_scatter: str = "Data Points",
    name_line: str = "Regression Line",
    scatter_color: str = "blue",
    line_color: str = "red",
    show_legend: bool = True
) -> List[go.Scatter]:
    """
    Create scatter plot with regression line.

    Args:
        x, y: Scatter data
        line_x, line_y: Line data (if None, uses x, y)
        name_scatter: Scatter legend name
        name_line: Line legend name
        scatter_color: Scatter color
        line_color: Line color
        show_legend: Whether to show legend

    Returns:
        List of Plotly Scatter objects
    """
    plots = []

    # Scatter plot
    scatter = create_plotly_scatter(
        x=x, y=y,
        name=name_scatter,
        mode="markers",
        color=scatter_color,
        show_legend=show_legend
    )
    plots.append(scatter)

    # Regression line
    if line_x is not None and line_y is not None:
        line = create_plotly_scatter(
            x=line_x, y=line_y,
            name=name_line,
            mode="lines",
            color=line_color,
            show_legend=show_legend
        )
        plots.append(line)

    return plots


def create_plotly_residual_plot(
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    name: str = "Residuals",
    color: str = "red",
    show_legend: bool = True
) -> go.Scatter:
    """
    Create residual plot (fitted values vs residuals).

    Args:
        fitted_values: Fitted/predicted values
        residuals: Residual values
        name: Legend name
        color: Point color
        show_legend: Whether to show in legend

    Returns:
        Plotly Scatter object
    """
    return create_plotly_scatter(
        x=fitted_values,
        y=residuals,
        name=name,
        mode="markers",
        color=color,
        show_legend=show_legend
    )


def create_plotly_bar(
    x: Union[List[str], np.ndarray],
    y: np.ndarray,
    name: str = "Bar Chart",
    color: str = "lightblue",
    show_legend: bool = True
) -> go.Bar:
    """
    Create a bar chart.

    Args:
        x: X-axis categories
        y: Bar heights
        name: Legend name
        color: Bar color
        show_legend: Whether to show in legend

    Returns:
        Plotly Bar object
    """
    return go.Bar(
        x=x, y=y,
        name=name,
        marker_color=color,
        showlegend=show_legend
    )


def create_plotly_distribution(
    data: Union[np.ndarray, pd.Series],
    name: str = "Distribution",
    color: str = "lightblue",
    show_legend: bool = True,
    histnorm: Optional[str] = None
) -> go.Histogram:
    """
    Create a distribution histogram.

    Args:
        data: Data to plot
        name: Legend name
        color: Bar color
        show_legend: Whether to show in legend
        histnorm: Normalization mode

    Returns:
        Plotly Histogram object
    """
    return go.Histogram(
        x=data,
        name=name,
        marker_color=color,
        showlegend=show_legend,
        histnorm=histnorm
    )