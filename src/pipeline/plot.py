"""
Step 3: PLOT

This module creates all visualizations.
It generates Plotly figures from data and calculation results.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import get_logger
from .get_data import DataResult, MultipleRegressionDataResult
from .calculate import RegressionResult, MultipleRegressionResult

logger = get_logger(__name__)


@dataclass
class PlotCollection:
    """Collection of plots for display."""
    scatter: go.Figure
    residuals: go.Figure
    diagnostics: Optional[go.Figure] = None
    extra: Dict[str, go.Figure] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class PlotBuilder:
    """
    Step 3: PLOT
    
    Creates visualizations from data and regression results.
    
    Example:
        plotter = PlotBuilder()
        plots = plotter.simple_regression_plots(data, regression_result)
    """
    
    # Color scheme
    COLORS = {
        "data": "#3498db",      # Blue for data points
        "regression": "#e74c3c", # Red for regression line
        "residual": "#2ecc71",   # Green for residuals
        "surface": "Blues",      # Colorscale for 3D
        "positive": "#27ae60",
        "negative": "#c0392b",
    }
    
    def __init__(self):
        logger.info("PlotBuilder initialized")
    
    def simple_regression_plots(
        self,
        data: DataResult,
        result: RegressionResult,
        show_true_line: bool = False,
        true_intercept: float = 0,
        true_slope: float = 0,
    ) -> PlotCollection:
        """
        Create all plots for simple regression.
        
        Args:
            data: Original data
            result: Regression calculation result
            show_true_line: Whether to show true regression line
            true_intercept: True β₀ (if known)
            true_slope: True β₁ (if known)
        
        Returns:
            PlotCollection with scatter, residuals, and diagnostic plots
        """
        logger.info("Creating simple regression plots")
        
        scatter = self._create_scatter_with_regression(
            data, result, show_true_line, true_intercept, true_slope
        )
        residuals = self._create_residual_plot(result)
        diagnostics = self._create_diagnostic_plots(result)
        
        return PlotCollection(
            scatter=scatter,
            residuals=residuals,
            diagnostics=diagnostics,
            extra={
                "histogram": self._create_residual_histogram(result),
            }
        )
    
    def multiple_regression_plots(
        self,
        data: MultipleRegressionDataResult,
        result: MultipleRegressionResult,
    ) -> PlotCollection:
        """
        Create all plots for multiple regression.
        
        Args:
            data: Original data
            result: Multiple regression result
        
        Returns:
            PlotCollection with 3D scatter, residuals, etc.
        """
        logger.info("Creating multiple regression plots")
        
        scatter_3d = self._create_3d_surface(data, result)
        residuals = self._create_residual_plot_multiple(result)
        diagnostics = self._create_diagnostic_plots_multiple(result)
        
        return PlotCollection(
            scatter=scatter_3d,
            residuals=residuals,
            diagnostics=diagnostics,
        )
    
    # =========================================================
    # PRIVATE: Plot Creators
    # =========================================================
    
    def _create_scatter_with_regression(
        self,
        data: DataResult,
        result: RegressionResult,
        show_true_line: bool,
        true_intercept: float,
        true_slope: float,
    ) -> go.Figure:
        """Create scatter plot with regression line."""
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter(
            x=data.x,
            y=data.y,
            mode="markers",
            name="Datenpunkte",
            marker=dict(
                size=10,
                color=self.COLORS["data"],
                opacity=0.7,
                line=dict(width=1, color="white")
            ),
        ))
        
        # Regression line
        x_line = np.linspace(min(data.x), max(data.x), 100)
        y_line = result.intercept + result.slope * x_line
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"OLS: ŷ = {result.intercept:.3f} + {result.slope:.3f}x",
            line=dict(color=self.COLORS["regression"], width=3),
        ))
        
        # True line (if known)
        if show_true_line and (true_intercept != 0 or true_slope != 0):
            y_true = true_intercept + true_slope * x_line
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_true,
                mode="lines",
                name=f"Wahr: y = {true_intercept:.2f} + {true_slope:.2f}x",
                line=dict(color="green", width=2, dash="dash"),
                opacity=0.7,
            ))
        
        # Mean lines
        fig.add_hline(
            y=result.extra.get("y_mean", np.mean(data.y)),
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"ȳ = {result.extra.get('y_mean', np.mean(data.y)):.2f}",
        )
        
        fig.update_layout(
            title=f"Regression: {data.y_label} vs {data.x_label}<br><sub>R² = {result.r_squared:.4f}</sub>",
            xaxis_title=data.x_label,
            yaxis_title=data.y_label,
            template="plotly_white",
            hovermode="closest",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        
        return fig
    
    def _create_residual_plot(self, result: RegressionResult) -> go.Figure:
        """Create residual vs fitted plot."""
        fig = go.Figure()
        
        # Residuals as scatter
        fig.add_trace(go.Scatter(
            x=result.y_pred,
            y=result.residuals,
            mode="markers",
            name="Residuen",
            marker=dict(
                size=8,
                color=result.residuals,
                colorscale="RdYlGn",
                cmin=-max(abs(result.residuals)),
                cmax=max(abs(result.residuals)),
                showscale=True,
                colorbar=dict(title="Residuum"),
            ),
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        
        fig.update_layout(
            title="Residuen vs. Fitted Values<br><sub>Sollte zufällig um 0 streuen</sub>",
            xaxis_title="Fitted Values (ŷ)",
            yaxis_title="Residuen (y - ŷ)",
            template="plotly_white",
        )
        
        return fig
    
    def _create_diagnostic_plots(self, result: RegressionResult) -> go.Figure:
        """Create 2x2 diagnostic plot grid."""
        from scipy import stats
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Residuen vs Fitted",
                "Normal Q-Q",
                "Scale-Location",
                "Residuen Histogramm"
            )
        )
        
        # 1. Residuals vs Fitted (already done, but simplified)
        fig.add_trace(
            go.Scatter(x=result.y_pred, y=result.residuals, mode="markers", 
                      marker=dict(size=6, opacity=0.6), showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Q-Q Plot
        sorted_resid = np.sort(result.residuals)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(result.residuals)))
        fig.add_trace(
            go.Scatter(x=theoretical, y=sorted_resid, mode="markers",
                      marker=dict(size=6, opacity=0.6), showlegend=False),
            row=1, col=2
        )
        # Q-Q reference line
        fig.add_trace(
            go.Scatter(x=theoretical, y=theoretical * np.std(result.residuals),
                      mode="lines", line=dict(color="red", dash="dash"), showlegend=False),
            row=1, col=2
        )
        
        # 3. Scale-Location
        std_resid = result.residuals / np.std(result.residuals)
        fig.add_trace(
            go.Scatter(x=result.y_pred, y=np.sqrt(np.abs(std_resid)), mode="markers",
                      marker=dict(size=6, opacity=0.6), showlegend=False),
            row=2, col=1
        )
        
        # 4. Histogram
        fig.add_trace(
            go.Histogram(x=result.residuals, nbinsx=15, marker_color=self.COLORS["data"],
                        showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            title_text="Diagnose-Plots",
        )
        
        return fig
    
    def _create_residual_histogram(self, result: RegressionResult) -> go.Figure:
        """Create residual histogram with normal curve."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=result.residuals,
            nbinsx=20,
            name="Residuen",
            marker_color=self.COLORS["data"],
            opacity=0.7,
        ))
        
        # Normal curve overlay
        x_norm = np.linspace(min(result.residuals), max(result.residuals), 100)
        from scipy import stats
        y_norm = stats.norm.pdf(x_norm, 0, np.std(result.residuals)) * len(result.residuals) * (max(result.residuals) - min(result.residuals)) / 20
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode="lines",
            name="Normalverteilung",
            line=dict(color=self.COLORS["regression"], width=2),
        ))
        
        fig.update_layout(
            title="Residuen-Verteilung",
            xaxis_title="Residuum",
            yaxis_title="Häufigkeit",
            template="plotly_white",
        )
        
        return fig
    
    def _create_3d_surface(
        self,
        data: MultipleRegressionDataResult,
        result: MultipleRegressionResult,
    ) -> go.Figure:
        """Create 3D scatter with regression plane."""
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter3d(
            x=data.x1,
            y=data.x2,
            z=data.y,
            mode="markers",
            name="Datenpunkte",
            marker=dict(
                size=5,
                color=data.y,
                colorscale="Viridis",
                opacity=0.8,
            ),
        ))
        
        # Regression plane
        x1_range = np.linspace(min(data.x1), max(data.x1), 20)
        x2_range = np.linspace(min(data.x2), max(data.x2), 20)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        Y_mesh = result.intercept + result.coefficients[0] * X1_mesh + result.coefficients[1] * X2_mesh
        
        fig.add_trace(go.Surface(
            x=X1_mesh,
            y=X2_mesh,
            z=Y_mesh,
            name="Regressionsebene",
            colorscale=self.COLORS["surface"],
            opacity=0.6,
            showscale=False,
        ))
        
        fig.update_layout(
            title=f"Multiple Regression: R² = {result.r_squared:.4f}",
            scene=dict(
                xaxis_title=data.x1_label,
                yaxis_title=data.x2_label,
                zaxis_title=data.y_label,
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            ),
            template="plotly_white",
        )
        
        return fig
    
    def _create_residual_plot_multiple(
        self, result: MultipleRegressionResult
    ) -> go.Figure:
        """Create residual plot for multiple regression."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=result.y_pred,
            y=result.residuals,
            mode="markers",
            marker=dict(size=8, color=self.COLORS["data"], opacity=0.7),
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Residuen vs Fitted (Multiple Regression)",
            xaxis_title="Fitted Values",
            yaxis_title="Residuen",
            template="plotly_white",
        )
        
        return fig
    
    def _create_diagnostic_plots_multiple(
        self, result: MultipleRegressionResult
    ) -> go.Figure:
        """Create diagnostic plots for multiple regression."""
        from scipy import stats
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Residuen vs Fitted", "Q-Q Plot", "Scale-Location", "Histogramm")
        )
        
        # Similar to simple regression diagnostics
        fig.add_trace(
            go.Scatter(x=result.y_pred, y=result.residuals, mode="markers",
                      marker=dict(size=6, opacity=0.6), showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        sorted_resid = np.sort(result.residuals)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(result.residuals)))
        fig.add_trace(
            go.Scatter(x=theoretical, y=sorted_resid, mode="markers",
                      marker=dict(size=6, opacity=0.6), showlegend=False),
            row=1, col=2
        )
        
        std_resid = result.residuals / np.std(result.residuals)
        fig.add_trace(
            go.Scatter(x=result.y_pred, y=np.sqrt(np.abs(std_resid)), mode="markers",
                      marker=dict(size=6, opacity=0.6), showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=result.residuals, nbinsx=15, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=500, template="plotly_white")
        
        return fig
