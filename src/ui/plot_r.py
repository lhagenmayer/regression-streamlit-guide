"""
R-style plotting functions for the Linear Regression Guide.

This module contains functions that mimic R's statistical plotting capabilities.
"""

from typing import Optional, List, Any
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ..config import get_logger
from ..data import safe_scalar as _safe_scalar

logger = get_logger(__name__)


def create_r_output_display(model: Any, feature_names: List[str] = None) -> str:
    """
    Create R-style model summary output text. (Legacy helper)
    """
    if not hasattr(model, 'summary'):
        return "Model summary not available"
    try:
        summary = model.summary()
        return summary.as_text() if hasattr(summary, 'as_text') else str(summary)
    except Exception as e:
        logger.warning(f"Could not create R-style summary text: {e}")
        return f"Model summary unavailable: {str(e)}"


def create_r_output_figure(
    model: Any,
    figure_type: str = "r_output",
    feature_names: Optional[List[str]] = None,
    figsize: tuple = (18, 13),
    **kwargs
) -> go.Figure:
    """
    Create R-style diagnostic figures or the summary output with annotations.

    Args:
        model: Statistical model object (statsmodels expected)
        figure_type: Type of figure ("r_output", "residuals", "qqplot", etc.)
        feature_names: Feature names for labeling
        figsize: Figure size
        **kwargs: Additional arguments

    Returns:
        Plotly figure object
    """
    if figure_type == "r_output":
        return _create_annotated_r_output(model, feature_names, figsize)
    
    try:
        if figure_type == "residuals":
            return _create_residuals_vs_fitted_plot(model, **kwargs)
        elif figure_type == "qqplot":
            from scipy import stats
            return _create_qq_plot(model, **kwargs)
        elif figure_type == "scale_location":
            return _create_scale_location_plot(model, **kwargs)
        elif figure_type == "residuals_leverage":
            return _create_residuals_leverage_plot(model, **kwargs)
        else:
            return _create_annotated_r_output(model, feature_names, figsize)

    except Exception as e:
        logger.error(f"Could not create {figure_type} plot: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating plot: {e}", showarrow=False)
        return fig


def _create_annotated_r_output(model: Any, feature_names: List[str] = None, figsize: tuple = (18, 13)) -> go.Figure:
    """
    Create an interactive Plotly figure mimics R output with educational annotations.
    """
    # 1. Get Summary Text
    summary_text = model.summary().as_text()
    lines = summary_text.split('\n')
    
    # Create the figure
    fig = go.Figure()
    
    # Add the text as a pseudo-terminal effect
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="markers", marker=dict(opacity=0), showlegend=False
    ))
    
    # Render text lines
    for i, line in enumerate(lines[:45]): # Limit lines for visibility
        fig.add_annotation(
            x=0, y=100 - i*2.2,
            text=f"<code>{line}</code>",
            showarrow=False,
            xanchor="left",
            font=dict(family="Courier New, monospace", size=14, color="white"),
            bgcolor="rgba(0,0,0,0)"
        )

    # 2. Add Educational Annotations (Boxes)
    # Box 1: Call
    fig.add_shape(type="rect", x0=-2, y0=95, x1=80, y1=100, line=dict(color="cyan", width=2), fillcolor="rgba(0, 255, 255, 0.1)")
    fig.add_annotation(x=82, y=97, text="<b>Call:</b> Die Modellformel.", showarrow=True, arrowhead=2, ax=40, ay=0, font=dict(color="cyan"))

    # Box 2: Residuals
    fig.add_shape(type="rect", x0=-2, y0=85, x1=50, y1=93, line=dict(color="yellow", width=2), fillcolor="rgba(255, 255, 0, 0.1)")
    fig.add_annotation(x=52, y=89, text="<b>Residuals:</b> Verteilung der Fehler.", showarrow=True, arrowhead=2, ax=40, ay=0, font=dict(color="yellow"))

    # Box 3: Coefficients
    fig.add_shape(type="rect", x0=-2, y0=55, x1=90, y1=83, line=dict(color="magenta", width=2), fillcolor="rgba(255, 0, 255, 0.1)")
    fig.add_annotation(x=92, y=70, text="<b>Coefficients:</b> Die Herzstücke.<br>Estimate = Steigung/Schnittpunkt<br>Pr(>|t|) = Signifikanz", showarrow=True, arrowhead=2, ax=40, ay=0, font=dict(color="magenta"))

    # Box 4: Quality Metrics
    fig.add_shape(type="rect", x0=-2, y0=40, x1=60, y1=50, line=dict(color="lime", width=2), fillcolor="rgba(0, 255, 0, 0.1)")
    fig.add_annotation(x=62, y=45, text="<b>Stats:</b> R-squared (Güte) & F-Statistik.", showarrow=True, arrowhead=2, ax=40, ay=0, font=dict(color="lime"))

    fig.update_layout(
        plot_bgcolor="rgb(30, 30, 30)",
        paper_bgcolor="rgb(30, 30, 30)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 120]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[35, 105]),
        width=figsize[0]*50, height=figsize[1]*50,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig


def _create_residuals_vs_fitted_plot(model: Any, **kwargs) -> go.Figure:
    """Create Residuals vs Fitted plot (R-style diagnostic plot 1)."""
    fitted = model.fittedvalues
    residuals = model.resid
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers', name='Residuals', marker=dict(color='rgba(31, 119, 180, 0.6)', size=8)))
    fig.add_trace(go.Scatter(x=[min(fitted), max(fitted)], y=[0, 0], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))
    # Lowess-like line
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(residuals, fitted)
    fig.add_trace(go.Scatter(x=smooth[:, 0], y=smooth[:, 1], mode='lines', line=dict(color='red', width=2), name='Lowess'))
    fig.update_layout(title="Residuals vs Fitted", xaxis_title="Fitted values", yaxis_title="Residuen", template="plotly_white")
    return fig


def _create_qq_plot(model: Any, **kwargs) -> go.Figure:
    """Create Q-Q plot (R-style diagnostic plot 2)."""
    import scipy.stats as stats
    residuals = model.resid
    osm, osr = stats.probplot(residuals, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm[0], y=osm[1], mode='markers', name='Residuals', marker=dict(color='blue', opacity=0.6)))
    # Add reference line
    line_x = np.array([min(osm[0]), max(osm[0])])
    line_y = line_x * np.std(residuals) + np.mean(residuals)
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', dash='dash'), name='Normal Reference'))
    fig.update_layout(title="Normal Q-Q", xaxis_title="Theoretical Quantiles", yaxis_title="Standardized Residuals", template="plotly_white")
    return fig


def _create_scale_location_plot(model: Any, **kwargs) -> go.Figure:
    """Create Scale-Location plot (R-style diagnostic plot 3)."""
    fitted = model.fittedvalues
    resid_std = np.sqrt(np.abs(stats.zscore(model.resid)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fitted, y=resid_std, mode='markers', name='Residuals'))
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(resid_std, fitted)
    fig.add_trace(go.Scatter(x=smooth[:, 0], y=smooth[:, 1], mode='lines', line=dict(color='red')))
    fig.update_layout(title="Scale-Location", xaxis_title="Fitted values", yaxis_title="√|Standardized residuals|", template="plotly_white")
    return fig


def _create_residuals_leverage_plot(model: Any, **kwargs) -> go.Figure:
    """Create Residuals vs Leverage plot (R-style diagnostic plot 5)."""
    inf = model.get_influence()
    leverage = inf.hat_matrix_diag
    resid_std = stats.zscore(model.resid)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=leverage, y=resid_std, mode='markers', name='Residuals'))
    # Add Cook's distance lines (simplified)
    # ...
    fig.update_layout(title="Residuals vs Leverage", xaxis_title="Leverage", yaxis_title="Standardized Residuals", template="plotly_white")
    return fig