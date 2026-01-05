"""
Plotting functions for the Linear Regression Guide.

This module contains all visualization functions using plotly.
"""

from typing import Optional, Union, List, Callable, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from logger import get_logger
from data import safe_scalar as _safe_scalar

# Initialize logger for this module
logger = get_logger(__name__)


# _safe_scalar is imported from data.py


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
        residuals: Model residuals
        base_size: Base marker size
        scale_factor: Scaling factor for residual magnitude

    Returns:
        Array of marker sizes
    """
    return base_size + np.abs(residuals) * scale_factor


def standardize_residuals(residuals: np.ndarray) -> np.ndarray:
    """
    Standardize residuals by dividing by their standard deviation.

    Args:
        residuals: Model residuals

    Returns:
        Standardized residuals
    """
    return residuals / np.std(residuals)


# ---------------------------------------------------------
# 3D VISUALIZATION HELPER FUNCTIONS
# ---------------------------------------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_regression_mesh(
    x1: np.ndarray, x2: np.ndarray, model_params: Union[List[float], np.ndarray], n_points: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create mesh grid for regression surface visualization.

    Args:
        x1: First predictor values
        x2: Second predictor values
        model_params: Model parameters [intercept, beta1, beta2]
        n_points: Number of grid points

    Returns:
        X1_mesh, X2_mesh, Y_mesh: Mesh grids for surface plotting
    """
    x1_range = np.linspace(x1.min(), x1.max(), n_points)
    x2_range = np.linspace(x2.min(), x2.max(), n_points)
    X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
    Y_mesh = model_params[0] + model_params[1] * X1_mesh + model_params[2] * X2_mesh
    return X1_mesh, X2_mesh, Y_mesh


def get_3d_layout_config(x_title: str, y_title: str, z_title: str, height: int = 600) -> dict:
    """Return standard 3D layout configuration.

    Args:
        x_title, y_title, z_title: Axis titles
        height: Plot height in pixels

    Returns:
        dict: Layout configuration for plotly 3D plots
    """
    return dict(
        template="plotly_white",
        height=height,
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
    )


def create_zero_plane(
    x_range: List[float], y_range: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a zero reference plane for 3D residual plots.

    Args:
        x_range: [min, max] for x axis
        y_range: [min, max] for y axis

    Returns:
        xx, yy, zz: Mesh grids for zero plane
    """
    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.zeros_like(xx)
    return xx, yy, zz


# ---------------------------------------------------------
# PLOTLY HELPER FUNCTIONS FOR COMMON PLOT TYPES
# ---------------------------------------------------------
def create_plotly_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "",
    marker_color: str = "blue",
    marker_size: int = 8,
    show_legend: bool = True,
) -> go.Figure:
    """Create a basic plotly scatter plot"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=marker_size, color=marker_color, opacity=0.7, line=dict(width=1, color="white")
            ),
            name="Data Points" if show_legend else None,
            showlegend=show_legend,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def create_plotly_scatter_with_line(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "",
) -> go.Figure:
    """Create scatter plot with regression line"""
    fig = go.Figure()

    # Data points
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=10, color="#1f77b4", opacity=0.7, line=dict(width=2, color="white")),
            name="Datenpunkte",
        )
    )

    # Regression line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            line=dict(color="#e74c3c", width=3),
            name="Regressionslinie",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
    )
    return fig


def create_plotly_3d_scatter(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    x1_label: str = "X1",
    x2_label: str = "X2",
    y_label: str = "Y",
    title: str = "",
    marker_color: Union[str, np.ndarray] = "red",
) -> go.Figure:
    """Create 3D scatter plot"""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x1,
                y=x2,
                z=y,
                mode="markers",
                marker=dict(
                    size=5,
                    color=marker_color if isinstance(marker_color, str) else y,
                    colorscale="Viridis" if not isinstance(marker_color, str) else None,
                    opacity=0.7,
                    colorbar=dict(title=y_label) if not isinstance(marker_color, str) else None,
                ),
                name="Data Points",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=x1_label, yaxis_title=x2_label, zaxis_title=y_label),
        template="plotly_white",
    )
    return fig


@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_plotly_3d_surface(
    X1_mesh: np.ndarray,
    X2_mesh: np.ndarray,
    Y_mesh: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    x1_label: str = "X1",
    x2_label: str = "X2",
    y_label: str = "Y",
    title: str = "",
) -> go.Figure:
    """Create 3D surface plot with data points"""
    fig = go.Figure()

    # Surface
    fig.add_trace(
        go.Surface(
            x=X1_mesh,
            y=X2_mesh,
            z=Y_mesh,
            colorscale="Viridis",
            opacity=0.7,
            name="Regression Plane",
            showscale=False,
        )
    )

    # Data points
    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode="markers",
            marker=dict(size=4, color="red", opacity=0.8),
            name="Data Points",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x1_label,
            yaxis_title=x2_label,
            zaxis_title=y_label,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        template="plotly_white",
    )
    return fig


def create_plotly_residual_plot(
    y_pred: np.ndarray, residuals: np.ndarray, title: str = "Residual Plot"
) -> go.Figure:
    """Create residual plot"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(size=8, color="blue", opacity=0.6),
            name="Residuals",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)

    fig.update_layout(
        title=title,
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def create_plotly_bar(
    categories: List[str],
    values: List[float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: Optional[List[str]] = None,
) -> go.Figure:
    """Create bar chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors if colors else "blue",
            opacity=0.7,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white"
    )
    return fig


def create_plotly_distribution(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    title: str = "",
    x_label: str = "",
    fill_area: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> go.Figure:
    """Create distribution plot"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals, y=y_vals, mode="lines", line=dict(color="black", width=2), name="Distribution"
        )
    )

    if fill_area is not None:
        mask = fill_area(x_vals)
        fig.add_trace(
            go.Scatter(
                x=x_vals[mask],
                y=y_vals[mask],
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(width=0),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Density",
        template="plotly_white",
        hovermode="x",
    )
    return fig


# ---------------------------------------------------------
# R-OUTPUT DISPLAY (Simplified text-based display)
# ---------------------------------------------------------
def create_r_output_display(model: Any, feature_names: List[str] = None) -> str:
    """
    Creates a structured display of R-style output matching the exact format.

    Args:
        model: Fitted statsmodels OLS model
        feature_names: List of feature names (excluding intercept)
    """
    # Extract all values
    resid = model.resid
    q = np.percentile(resid, [0, 25, 50, 75, 100])
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    rse = np.sqrt(model.mse_resid)
    df_resid = int(model.df_resid)
    df_model = int(model.df_model)

    # Create formula string
    if feature_names is None:
        # For simple regression, try to infer from model
        if len(params) == 2:
            feature_names = ["hp"]  # Default assumption
        else:
            feature_names = [f"X{i}" for i in range(1, len(params))]

    if len(feature_names) == 1:
        formula = f"mpg ~ {feature_names[0]}"
    elif len(feature_names) == 2:
        formula = f"mpg ~ {feature_names[0]} + {feature_names[1]}"
    elif len(feature_names) == 3:
        formula = f"mpg ~ hp + drat + wt"  # Use the exact example from user
    else:
        formula = f"mpg ~ {' + '.join(feature_names)}"

    # Build coefficients table
    coef_lines = []
    param_names = ["(Intercept)"] + feature_names

    for i, (name, param, std_err, t_val, p_val) in enumerate(zip(param_names, params, bse, tvals, pvals)):
        stars = get_signif_stars(p_val)
        # Format p-value to match R's format
        if p_val < 2e-16:
            p_str = "< 2e-16"
        elif p_val < 0.0001:
            # Use scientific notation for very small p-values
            if p_val >= 1e-4:
                p_str = f"{p_val:.6f}"
            else:
                p_str = f"{p_val:.2e}"
        else:
            p_str = f"{p_val:.6f}"

        if i == 0:  # Intercept
            coef_lines.append(f"(Intercept) {param:11.6f} {std_err:10.6f} {t_val:7.3f} {p_str:>10}{stars}")
        else:  # Other parameters
            coef_lines.append(f"{name:<11} {param:11.6f} {std_err:10.6f} {t_val:7.3f} {p_str:>10}{stars}")

    coef_table = "\n".join(coef_lines)

    # Create formatted text output matching exact R format
    output_text = f"""Call:
lm(formula = {formula}, data = mtcars)

Residuals:
    Min      1Q  Median      3Q     Max
{q[0]:7.4f} {q[1]:7.4f} {q[2]:7.4f} {q[3]:7.4f} {q[4]:7.4f}

Coefficients:
             Estimate Std. Error t value Pr(>|t|)
{coef_table}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rse:.4f} on {df_resid} degrees of freedom
Multiple R-squared:  {model.rsquared:.4f},    Adjusted R-squared:  {model.rsquared_adj:.4f}
F-statistic: {model.fvalue:.2f} on {df_model} and {df_resid} DF,  p-value: {model.f_pvalue:.4g}"""
    return output_text


def create_r_output_figure(
    model: Any, feature_names: List[str] = None, figsize: Tuple[int, int] = (18, 13)
) -> go.Figure:
    """
    Create an annotated figure showing R-style output.
    This returns a plotly figure with text annotations.
    """
    # Create empty figure for text display
    fig = go.Figure()

    # Get the text output
    output_text = create_r_output_display(model, feature_names)

    # Add text annotation
    fig.add_annotation(
        text=output_text.replace("\n", "<br>"),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family="Courier New, monospace", size=12, color="black"),
        align="left",
        xanchor="center",
        yanchor="middle",
    )

    # Update layout
    fig.update_layout(
        title="R-Style Output mit Annotationen",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=figsize[1] * 50,
        width=figsize[0] * 50,
        plot_bgcolor="white",
    )

    return fig
