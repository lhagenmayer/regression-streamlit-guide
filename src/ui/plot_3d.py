"""
3D plotting functions for the Linear Regression Guide.

This module contains functions for creating 3D visualizations.
"""

from typing import Optional, Union, List, Tuple, Any
import numpy as np
import plotly.graph_objects as go
from ..config import get_logger

logger = get_logger(__name__)


def create_regression_mesh(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh grid for 3D regression surface plotting.

    Args:
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
        resolution: Number of points in each dimension

    Returns:
        Tuple of (X_mesh, Y_mesh) for plotting
    """
    x_lin = np.linspace(x_range[0], x_range[1], resolution)
    y_lin = np.linspace(y_range[0], y_range[1], resolution)
    X_mesh, Y_mesh = np.meshgrid(x_lin, y_lin)
    return X_mesh, Y_mesh


def get_3d_layout_config(x_title: str, y_title: str, z_title: str, height: int = 600) -> dict:
    """
    Get standardized 3D layout configuration.

    Args:
        x_title: X-axis title
        y_title: Y-axis title
        z_title: Z-axis title
        height: Plot height

    Returns:
        Plotly layout configuration
    """
    return {
        "scene": {
            "xaxis": {"title": x_title},
            "yaxis": {"title": y_title},
            "zaxis": {"title": z_title},
        },
        "height": height,
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "showlegend": True,
    }


def create_zero_plane(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_value: float = 0,
    color: str = "lightgray",
    opacity: float = 0.3
) -> go.Surface:
    """
    Create a zero plane for 3D plots.

    Args:
        x_range: X-axis range
        y_range: Y-axis range
        z_value: Z-value for the plane
        color: Plane color
        opacity: Plane opacity

    Returns:
        Plotly Surface object
    """
    x_vals = np.linspace(x_range[0], x_range[1], 10)
    y_vals = np.linspace(y_range[0], y_range[1], 10)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.full_like(X, z_value)

    return go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        name="Zero Plane"
    )


def create_plotly_3d_scatter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str = "Data Points",
    color: str = "blue",
    size: Optional[Union[int, np.ndarray]] = None,
    opacity: float = 0.8
) -> go.Scatter3d:
    """
    Create a 3D scatter plot.

    Args:
        x, y, z: Coordinate arrays
        name: Legend name
        color: Point color
        size: Point sizes
        opacity: Point opacity

    Returns:
        Plotly Scatter3d object
    """
    marker_config = {
        "color": color,
        "opacity": opacity,
    }

    if size is not None:
        if np.isscalar(size):
            marker_config["size"] = size
        else:
            marker_config["size"] = size

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        name=name,
        marker=marker_config
    )


def create_plotly_3d_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    name: str = "Regression Surface",
    colorscale: str = "Viridis",
    opacity: float = 0.7
) -> go.Surface:
    """
    Create a 3D surface plot.

    Args:
        X, Y, Z: Mesh grid arrays
        name: Legend name
        colorscale: Color scale
        opacity: Surface opacity

    Returns:
        Plotly Surface object
    """
    return go.Surface(
        x=X, y=Y, z=Z,
        name=name,
        colorscale=colorscale,
        opacity=opacity,
        showscale=False
    )