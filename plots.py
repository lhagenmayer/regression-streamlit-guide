"""
Plotting functions for the Linear Regression Guide.

This module contains all visualization functions using plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def _safe_scalar(val):
    """Helper: Konvertiert Series/ndarray zu Skalar, falls nötig."""
    import pandas as pd
    if isinstance(val, (pd.Series, np.ndarray)):
        return float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
    return float(val)


# ---------------------------------------------------------
# 3D VISUALIZATION HELPER FUNCTIONS
# ---------------------------------------------------------
def create_regression_mesh(x1, x2, model_params, n_points=20):
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
    Y_mesh = model_params[0] + model_params[1]*X1_mesh + model_params[2]*X2_mesh
    return X1_mesh, X2_mesh, Y_mesh


def get_3d_layout_config(x_title, y_title, z_title, height=600):
    """Return standard 3D layout configuration.
    
    Args:
        x_title, y_title, z_title: Axis titles
        height: Plot height in pixels
    
    Returns:
        dict: Layout configuration for plotly 3D plots
    """
    return dict(
        template='plotly_white',
        height=height,
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        )
    )


def create_zero_plane(x_range, y_range):
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
def create_plotly_scatter(x, y, x_label='X', y_label='Y', title='', 
                         marker_color='blue', marker_size=8, show_legend=True):
    """Create a basic plotly scatter plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=marker_size, color=marker_color, opacity=0.7,
                   line=dict(width=1, color='white')),
        name='Data Points' if show_legend else None,
        showlegend=show_legend
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest'
    )
    return fig


def create_plotly_scatter_with_line(x, y, y_pred, x_label='X', y_label='Y', title=''):
    """Create scatter plot with regression line"""
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=10, color='#1f77b4', opacity=0.7,
                   line=dict(width=2, color='white')),
        name='Datenpunkte'
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=x, y=y_pred, mode='lines',
        line=dict(color='#e74c3c', width=3),
        name='Regressionslinie'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest',
        showlegend=True
    )
    return fig


def create_plotly_3d_scatter(x1, x2, y, x1_label='X1', x2_label='X2', y_label='Y', 
                            title='', marker_color='red'):
    """Create 3D scatter plot"""
    fig = go.Figure(data=[go.Scatter3d(
        x=x1, y=x2, z=y,
        mode='markers',
        marker=dict(
            size=5,
            color=marker_color if isinstance(marker_color, str) else y,
            colorscale='Viridis' if not isinstance(marker_color, str) else None,
            opacity=0.7,
            colorbar=dict(title=y_label) if not isinstance(marker_color, str) else None
        ),
        name='Data Points'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x1_label,
            yaxis_title=x2_label,
            zaxis_title=y_label
        ),
        template='plotly_white'
    )
    return fig


def create_plotly_3d_surface(X1_mesh, X2_mesh, Y_mesh, x1, x2, y,
                             x1_label='X1', x2_label='X2', y_label='Y', title=''):
    """Create 3D surface plot with data points"""
    fig = go.Figure()
    
    # Surface
    fig.add_trace(go.Surface(
        x=X1_mesh, y=X2_mesh, z=Y_mesh,
        colorscale='Viridis',
        opacity=0.7,
        name='Regression Plane',
        showscale=False
    ))
    
    # Data points
    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=y,
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.8),
        name='Data Points'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x1_label,
            yaxis_title=x2_label,
            zaxis_title=y_label,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        template='plotly_white'
    )
    return fig


def create_plotly_residual_plot(y_pred, residuals, title='Residual Plot'):
    """Create residual plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.6),
        name='Residuals'
    ))
    
    fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2)
    
    fig.update_layout(
        title=title,
        xaxis_title='Fitted Values',
        yaxis_title='Residuals',
        template='plotly_white',
        hovermode='closest'
    )
    return fig


def create_plotly_bar(categories, values, title='', x_label='', y_label='',
                     colors=None):
    """Create bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors if colors else 'blue',
        opacity=0.7,
        text=[f'{v:.2f}' for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white'
    )
    return fig


def create_plotly_distribution(x_vals, y_vals, title='', x_label='', fill_area=None):
    """Create distribution plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines',
        line=dict(color='black', width=2),
        name='Distribution'
    ))
    
    if fill_area is not None:
        mask = fill_area(x_vals)
        fig.add_trace(go.Scatter(
            x=x_vals[mask], y=y_vals[mask],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(width=0),
            showlegend=False
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='Density',
        template='plotly_white',
        hovermode='x'
    )
    return fig


# ---------------------------------------------------------
# R-OUTPUT DISPLAY (Simplified text-based display)
# ---------------------------------------------------------
def create_r_output_display(model, feature_name="X"):
    """
    Creates a structured display of R-style output using Streamlit components
    instead of matplotlib figure. This provides better interactivity.
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
    
    def get_signif_stars(p):
        """Signifikanz-Codes wie in R"""
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        if p < 0.1:   return '.'
        return ' '
    
    # Create formatted text output
    output_text = f"""
Python Replikation des R-Outputs: summary(lm_model)
===================================================

Residuals:
    Min      1Q  Median      3Q     Max
{q[0]:7.4f} {q[1]:7.4f} {q[2]:7.4f} {q[3]:7.4f} {q[4]:7.4f}

Coefficients:
             Estimate Std.Err  t val  Pr(>|t|)    
(Intercept)  {params[0]:9.4f} {bse[0]:8.4f} {tvals[0]:7.2f} {pvals[0]:10.4g} {get_signif_stars(pvals[0])}
{feature_name:<13}{params[1]:9.4f} {bse[1]:8.4f} {tvals[1]:7.2f} {pvals[1]:10.4g} {get_signif_stars(pvals[1])}
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rse:.4f} on {df_resid} degrees of freedom
Multiple R-squared:  {model.rsquared:.4f},    Adjusted R-squared:  {model.rsquared_adj:.4f}
F-statistic: {model.fvalue:.1f} on {df_model} and {df_resid} DF,  p-value: {model.f_pvalue:.4g}
"""
    return output_text


def create_r_output_figure(model, feature_name="X", figsize=(18, 13)):
    """
    Create an annotated figure showing R-style output.
    This returns a plotly figure with text annotations.
    """
    # Create empty figure for text display
    fig = go.Figure()
    
    # Get the text output
    output_text = create_r_output_display(model, feature_name)
    
    # Add text annotation
    fig.add_annotation(
        text=output_text.replace('\n', '<br>'),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(family="Courier New, monospace", size=12, color="black"),
        align="left",
        xanchor="center",
        yanchor="middle"
    )
    
    # Update layout
    fig.update_layout(
        title="R-Style Output mit Annotationen",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=figsize[1] * 50,
        width=figsize[0] * 50,
        plot_bgcolor='white'
    )
    
    return fig
