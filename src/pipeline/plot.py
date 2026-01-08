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
        """Create 3D scatter plot with regression line (z=0)."""
        fig = go.Figure()
        
        z_zeros = np.zeros(len(data.x))
        
        # Data points
        fig.add_trace(go.Scatter3d(
            x=data.x,
            y=data.y,
            z=z_zeros,
            mode="markers",
            name="Datenpunkte",
            marker=dict(
                size=5,
                color=self.COLORS["data"],
                opacity=0.8,
                line=dict(width=1, color="white")
            ),
        ))
        
        # Regression line
        x_line = np.linspace(min(data.x), max(data.x), 100)
        y_line = result.intercept + result.slope * x_line
        z_line = np.zeros(len(x_line))
        
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode="lines",
            name=f"OLS: ŷ = {result.intercept:.3f} + {result.slope:.3f}x",
            line=dict(color=self.COLORS["regression"], width=5),
        ))
        
        # True line (if known)
        if show_true_line and (true_intercept != 0 or true_slope != 0):
            y_true = true_intercept + true_slope * x_line
            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=y_true,
                z=z_line,
                mode="lines",
                name=f"Wahr: y = {true_intercept:.2f} + {true_slope:.2f}x",
                line=dict(color="green", width=4, dash="dash"),
                opacity=0.7,
            ))
        
        # Mean lines (plane/line at y=mean)
        y_mean = result.extra.get("y_mean", np.mean(data.y))
        fig.add_trace(go.Scatter3d(
            x=[min(data.x), max(data.x)],
            y=[y_mean, y_mean],
            z=[0, 0],
            mode="lines",
            name=f"ȳ = {y_mean:.2f}",
            line=dict(color="gray", width=2, dash="dot"),
        ))
        
        fig.update_layout(
            title=f"Regression (3D View): {data.y_label} vs {data.x_label}<br><sub>R² = {result.r_squared:.4f}</sub>",
            scene=dict(
                xaxis_title=data.x_label,
                yaxis_title=data.y_label,
                zaxis_title="Z (Null)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            ),
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=40),
        )
        
        return fig
    
    def _create_residual_plot(self, result: RegressionResult) -> go.Figure:
        """Create residual vs fitted plot in 3D."""
        fig = go.Figure()
        
        z_zeros = np.zeros(len(result.residuals))
        
        # Residuals as scatter
        fig.add_trace(go.Scatter3d(
            x=result.y_pred,
            y=result.residuals,
            z=z_zeros,
            mode="markers",
            name="Residuen",
            marker=dict(
                size=5,
                color=result.residuals,
                colorscale="RdYlGn",
                cmin=-max(abs(result.residuals)),
                cmax=max(abs(result.residuals)),
                showscale=True,
                colorbar=dict(title="Residuum"),
            ),
        ))
        
        # Zero line
        x_range = np.linspace(min(result.y_pred), max(result.y_pred), 100)
        fig.add_trace(go.Scatter3d(
            x=x_range,
            y=np.zeros(len(x_range)),
            z=np.zeros(len(x_range)),
            mode="lines",
            name="Null-Linie",
            line=dict(color="red", width=4, dash="dash"),
        ))
        
        fig.update_layout(
            title="Residuen vs. Fitted Values (3D)<br><sub>Sollte zufällig um 0 streuen</sub>",
            scene=dict(
                xaxis_title="Fitted Values (ŷ)",
                yaxis_title="Residuen (y - ŷ)",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            ),
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=40),
        )
        
        return fig
    
    def _create_diagnostic_plots(self, result: RegressionResult) -> go.Figure:
        """Create 2x2 diagnostic plot grid in 3D."""
        from scipy import stats
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scene"}, {"type": "scene"}],
                [{"type": "scene"}, {"type": "scene"}]
            ],
            subplot_titles=(
                "Residuen vs Fitted (3D)",
                "Normal Q-Q (3D)",
                "Scale-Location (3D)",
                "Residuen Verteilung (3D)"
            )
        )
        
        z_zeros = np.zeros(len(result.residuals))
        
        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter3d(x=result.y_pred, y=result.residuals, z=z_zeros, mode="markers", 
                      marker=dict(size=4, opacity=0.6), showlegend=False),
            row=1, col=1
        )
        # Zero line for plot 1
        x_range_1 = np.linspace(min(result.y_pred), max(result.y_pred), 50)
        fig.add_trace(
            go.Scatter3d(x=x_range_1, y=np.zeros(50), z=np.zeros(50), mode="lines",
                        line=dict(color="red", dash="dash"), showlegend=False),
            row=1, col=1
        )
        
        # 2. Q-Q Plot
        sorted_resid = np.sort(result.residuals)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(result.residuals)))
        z_qq = np.zeros(len(theoretical))
        fig.add_trace(
            go.Scatter3d(x=theoretical, y=sorted_resid, z=z_qq, mode="markers",
                      marker=dict(size=4, opacity=0.6), showlegend=False),
            row=1, col=2
        )
        # Q-Q reference line
        fig.add_trace(
            go.Scatter3d(x=theoretical, y=theoretical * np.std(result.residuals), z=z_qq,
                      mode="lines", line=dict(color="red", dash="dash"), showlegend=False),
            row=1, col=2
        )
        
        # 3. Scale-Location
        std_resid = result.residuals / np.std(result.residuals)
        fig.add_trace(
            go.Scatter3d(x=result.y_pred, y=np.sqrt(np.abs(std_resid)), z=z_zeros, mode="markers",
                      marker=dict(size=4, opacity=0.6), showlegend=False),
            row=2, col=1
        )
        
        # 4. Histogram (simulated in 3D with scatter lines)
        # Using numpy to calculate histogram bins
        hist, bin_edges = np.histogram(result.residuals, bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        z_hist = np.zeros(len(bin_centers))
        
        # Bar representation in 3D is hard with Scatter3d, let's use lines
        for i in range(len(bin_centers)):
            fig.add_trace(
                go.Scatter3d(
                    x=[bin_centers[i], bin_centers[i]],
                    y=[0, hist[i]],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color=self.COLORS["data"], width=5),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            template="plotly_white",
            title_text="Diagnose-Plots (3D)",
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=40),
        )
        
        return fig
    
    def _create_residual_histogram(self, result: RegressionResult) -> go.Figure:
        """Create residual histogram with normal curve in 3D."""
        fig = go.Figure()
        
        # Calculate histogram data
        hist, bin_edges = np.histogram(result.residuals, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Draw bars as vertical lines in 3D
        for i in range(len(bin_centers)):
            fig.add_trace(go.Scatter3d(
                x=[bin_centers[i], bin_centers[i]],
                y=[0, hist[i]],
                z=[0, 0],
                mode="lines",
                line=dict(color=self.COLORS["data"], width=10),
                showlegend=False,
                name="Residuen"
            ))
            # Add top marker
            fig.add_trace(go.Scatter3d(
                x=[bin_centers[i]],
                y=[hist[i]],
                z=[0],
                mode="markers",
                marker=dict(size=4, color=self.COLORS["data"]),
                showlegend=False
            ))
        
        # Normal curve overlay
        x_norm = np.linspace(min(result.residuals), max(result.residuals), 100)
        from scipy import stats
        y_norm = stats.norm.pdf(x_norm, 0, np.std(result.residuals)) * len(result.residuals) * (max(result.residuals) - min(result.residuals)) / 20
        z_norm = np.zeros(len(x_norm))
        
        fig.add_trace(go.Scatter3d(
            x=x_norm,
            y=y_norm,
            z=z_norm,
            mode="lines",
            name="Normalverteilung",
            line=dict(color=self.COLORS["regression"], width=4),
        ))
        
        fig.update_layout(
            title="Residuen-Verteilung (3D)",
            scene=dict(
                xaxis_title="Residuum",
                yaxis_title="Häufigkeit",
                zaxis_title="",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            ),
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=40),
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
            margin=dict(l=0, r=0, b=0, t=40),
        )
        
        return fig
    
    def _create_residual_plot_multiple(
        self, result: MultipleRegressionResult
    ) -> go.Figure:
        """Create residual plot for multiple regression in 3D."""
        fig = go.Figure()
        
        z_zeros = np.zeros(len(result.residuals))
        
        fig.add_trace(go.Scatter3d(
            x=result.y_pred,
            y=result.residuals,
            z=z_zeros,
            mode="markers",
            marker=dict(size=5, color=self.COLORS["data"], opacity=0.7),
        ))
        
        # Zero line
        x_range = np.linspace(min(result.y_pred), max(result.y_pred), 100)
        fig.add_trace(go.Scatter3d(
            x=x_range,
            y=np.zeros(len(x_range)),
            z=np.zeros(len(x_range)),
            mode="lines",
            line=dict(color="red", dash="dash", width=4),
        ))
        
        fig.update_layout(
            title="Residuen vs Fitted (Multiple Regression - 3D)",
            scene=dict(
                xaxis_title="Fitted Values",
                yaxis_title="Residuen",
                zaxis_title="",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            ),
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=40),
        )
        
        return fig
    
    def _create_diagnostic_plots_multiple(
        self, result: MultipleRegressionResult
    ) -> go.Figure:
        """Create diagnostic plots for multiple regression in 3D."""
        from scipy import stats
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scene"}, {"type": "scene"}],
                [{"type": "scene"}, {"type": "scene"}]
            ],
            subplot_titles=("Residuen vs Fitted (3D)", "Q-Q Plot (3D)", "Scale-Location (3D)", "Histogramm (3D)")
        )
        
        z_zeros = np.zeros(len(result.residuals))
        
        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter3d(x=result.y_pred, y=result.residuals, z=z_zeros, mode="markers",
                      marker=dict(size=4, opacity=0.6), showlegend=False),
            row=1, col=1
        )
        x_range_1 = np.linspace(min(result.y_pred), max(result.y_pred), 50)
        fig.add_trace(
            go.Scatter3d(x=x_range_1, y=np.zeros(50), z=np.zeros(50), mode="lines",
                        line=dict(color="red", dash="dash"), showlegend=False),
            row=1, col=1
        )
        
        # 2. Q-Q Plot
        sorted_resid = np.sort(result.residuals)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(result.residuals)))
        z_qq = np.zeros(len(theoretical))
        fig.add_trace(
            go.Scatter3d(x=theoretical, y=sorted_resid, z=z_qq, mode="markers",
                      marker=dict(size=4, opacity=0.6), showlegend=False),
            row=1, col=2
        )
        
        # 3. Scale-Location
        std_resid = result.residuals / np.std(result.residuals)
        fig.add_trace(
            go.Scatter3d(x=result.y_pred, y=np.sqrt(np.abs(std_resid)), z=z_zeros, mode="markers",
                      marker=dict(size=4, opacity=0.6), showlegend=False),
            row=2, col=1
        )
        
        # 4. Histogram
        hist, bin_edges = np.histogram(result.residuals, bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for i in range(len(bin_centers)):
            fig.add_trace(
                go.Scatter3d(
                    x=[bin_centers[i], bin_centers[i]],
                    y=[0, hist[i]],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color=self.COLORS["data"], width=5),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=40),
        )
        
        return fig
