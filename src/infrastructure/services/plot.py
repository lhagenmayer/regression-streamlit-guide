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

from ...config import get_logger
from ..data.generators import DataResult, MultipleRegressionDataResult
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
    
    # Modern, professional color scheme for teaching
    COLORS = {
        "data": "#1f77b4",       # Distinct Blue
        "data_marker": "#155a8a", # Darker outline
        "model": "#ff7f0e",      # Safety Orange (High visibility)
        "true_model": "#2ca02c", # Green for "Truth"
        "residual_pos": "#d62728", # Red for error
        "residual_neg": "#d62728",
        "plane": "Blues",        # Surface colorscale
        "shadow": "rgba(0, 0, 0, 0.15)", # For floor projections
        "grid": "#E5E5E5",
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
        """Create 3D scatter plot with explicit residuals to visualize Error."""
        fig = go.Figure()
        
        # We use X axis for Feature, Z axis for Target (Height), Y axis fixed at 0
        # This creates a "Billboard" effect in 3D space which is readable but allows 3D interactions
        y_pos = np.zeros(len(data.x))
        
        # 1. Residual Lines (The "Sticks") - linking Data to Model
        # This gives immediate intuition about "Distance from Line"
        for i in range(len(data.x)):
            fig.add_trace(go.Scatter3d(
                x=[data.x[i], data.x[i]],
                y=[0, 0],
                z=[data.y[i], result.y_pred[i]],
                mode="lines",
                line=dict(color="rgba(200, 50, 50, 0.4)", width=2),
                showlegend=False,
                hoverinfo="skip"
            ))

        # 2. Data points (Spheres)
        fig.add_trace(go.Scatter3d(
            x=data.x,
            y=y_pos,
            z=data.y,
            mode="markers",
            name="Beobachtete Daten",
            marker=dict(
                size=6,
                color=self.COLORS["data"],
                opacity=0.9,
                line=dict(width=1, color="white"),
                symbol="circle"
            ),
            hovertemplate=f"{data.x_label}: %{{x:.1f}}<br>{data.y_label}: %{{z:.1f}}<extra></extra>"
        ))
        
        # 3. Regression Line (Thick Tube)
        x_line = np.linspace(min(data.x), max(data.x), 100)
        y_line = result.intercept + result.slope * x_line
        z_line = np.zeros(len(x_line))
        
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=z_line,
            z=y_line,
            mode="lines",
            name=f"Modell (OLS): ŷ = {result.intercept:.2f} + {result.slope:.2f}x",
            line=dict(color=self.COLORS["model"], width=6),
            hovertemplate="Modellvorhersage<extra></extra>"
        ))
        
        # 4. True line (if known)
        if show_true_line and (true_intercept != 0 or true_slope != 0):
            y_true = true_intercept + true_slope * x_line
            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=z_line,
                z=y_true,
                mode="lines",
                name=f"Wahre Population: y = {true_intercept:.2f} + {true_slope:.2f}x",
                line=dict(color=self.COLORS["true_model"], width=4, dash="dashdot"),
                opacity=0.7,
            ))
            
        # 5. Floor Projections (Shadows) to ground the data
        # Project points to Z=min to show X distribution
        z_min = min(min(data.y), min(y_line)) * 0.95
        fig.add_trace(go.Scatter3d(
            x=data.x,
            y=y_pos,
            z=[z_min] * len(data.x),
            mode="markers",
            marker=dict(size=4, color="gray", opacity=0.3),
            name="X-Verteilung",
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.update_layout(
            title=f"<b>Lineare Regression 3D</b><br><sup>Datenpunkte (Blau) vs. Modell (Orange) | Rote Linien = Fehler (Residuen)</sup>",
            scene=dict(
                xaxis=dict(title=data.x_label, backgroundcolor=self.COLORS["grid"], gridcolor="white"),
                yaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False), # Hide Y axis as it's depth 0
                zaxis=dict(title=data.y_label, backgroundcolor=self.COLORS["grid"], gridcolor="white"),
                camera=dict(
                    eye=dict(x=0.0, y=-2.0, z=0.5), # Front view initially
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.2, z=0.8) # Flatten the Y depth
            ),
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=60),
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.8)"),
        )
        
        return fig
    
    def _create_residual_plot(self, result: RegressionResult) -> go.Figure:
        """Create residual plot in 3D space."""
        fig = go.Figure()
        
        z_zeros = np.zeros(len(result.residuals))
        y_zeros = np.zeros(len(result.residuals))
        
        # Residual "Lollipops" (Stems from 0 to Residual)
        for i in range(len(result.residuals)):
            color = self.COLORS["residual_pos"] if result.residuals[i] >= 0 else self.COLORS["residual_neg"]
            fig.add_trace(go.Scatter3d(
                x=[result.y_pred[i], result.y_pred[i]],
                y=[0, 0],
                z=[0, result.residuals[i]],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False
            ))
        
        # Residual Heads
        fig.add_trace(go.Scatter3d(
            x=result.y_pred,
            y=y_zeros,
            z=result.residuals,
            mode="markers",
            name="Residuen",
            marker=dict(
                size=5,
                color=result.residuals,
                colorscale="RdBu",
                cmid=0,
                line=dict(width=1, color="white")
            ),
            hovertemplate="Fitted: %{x:.2f}<br>Residuum: %{z:.2f}<extra></extra>"
        ))
        
        # Zero Plane/Line
        x_range = np.linspace(min(result.y_pred), max(result.y_pred), 100)
        fig.add_trace(go.Scatter3d(
            x=x_range,
            y=np.zeros(len(x_range)),
            z=np.zeros(len(x_range)),
            mode="lines",
            name="Null-Linie (Perfekter Fit)",
            line=dict(color="black", width=2, dash="dash"),
        ))
        
        # Add visual bands for standard deviation
        std_resid = np.std(result.residuals)
        fig.add_trace(go.Scatter3d(
            x=x_range, y=np.zeros(len(x_range)), z=[2*std_resid]*len(x_range),
            mode="lines", line=dict(color="gray", width=1, dash="dot"), name="+2 SD"
        ))
        fig.add_trace(go.Scatter3d(
            x=x_range, y=np.zeros(len(x_range)), z=[-2*std_resid]*len(x_range),
            mode="lines", line=dict(color="gray", width=1, dash="dot"), name="-2 SD"
        ))
        
        fig.update_layout(
            title="<b>Residuen-Analyse</b><br><sup>Struktur der Fehler (Sollte zufälliges Rauschen um 0 sein)</sup>",
            scene=dict(
                xaxis_title="Vorhergesagte Werte (ŷ)",
                yaxis_title="",
                zaxis_title="Residuen (y - ŷ)",
                camera=dict(eye=dict(x=0, y=-2.0, z=0.5)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.2, z=0.6)
            ),
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    def _create_diagnostic_plots(self, result: RegressionResult) -> go.Figure:
        """Create diagnostic plots as 3D scenes."""
        from scipy import stats
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scene"}, {"type": "scene"}],
                [{"type": "scene"}, {"type": "scene"}]
            ],
            subplot_titles=(
                "Linearität (Residuen vs Fitted)",
                "Normalität (Q-Q Plot)",
                "Homoskedastizität (Scale-Location)",
                "Verteilung (Histogramm)"
            )
        )
        
        zeros = np.zeros(len(result.residuals))
        
        # 1. Residuals vs Fitted
        # Use simple scatter in 3D plane
        fig.add_trace(
            go.Scatter3d(x=result.y_pred, y=zeros, z=result.residuals, mode="markers", 
                      marker=dict(size=4, color=self.COLORS["data"], opacity=0.7), showlegend=False),
            row=1, col=1
        )
        fig.add_trace(go.Scatter3d(
            x=[min(result.y_pred), max(result.y_pred)], y=[0,0], z=[0,0], 
            mode="lines", line=dict(color="red", dash="dash"), showlegend=False), row=1, col=1
        )
        
        # 2. Q-Q Plot
        sorted_resid = np.sort(result.residuals)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(result.residuals)))
        
        fig.add_trace(
            go.Scatter3d(x=theoretical, y=zeros, z=sorted_resid, mode="markers",
                      marker=dict(size=4, color=self.COLORS["data"], opacity=0.7), showlegend=False),
            row=1, col=2
        )
        # Reference line
        slope_qq = np.std(result.residuals)
        fig.add_trace(
            go.Scatter3d(x=theoretical, y=zeros, z=theoretical * slope_qq,
                      mode="lines", line=dict(color="red", dash="dash"), showlegend=False),
            row=1, col=2
        )
        
        # 3. Scale-Location
        std_resid = result.residuals / np.std(result.residuals)
        sqrt_std_resid = np.sqrt(np.abs(std_resid))
        
        fig.add_trace(
            go.Scatter3d(x=result.y_pred, y=zeros, z=sqrt_std_resid, mode="markers",
                      marker=dict(size=4, color=self.COLORS["data"], opacity=0.7), showlegend=False),
            row=2, col=1
        )
        
        # 4. Histogram (3D Bars)
        hist, bin_edges = np.histogram(result.residuals, bins=15)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for i, (x, h) in enumerate(zip(bin_centers, hist)):
            fig.add_trace(
                go.Scatter3d(
                    x=[x, x], y=[0, 0], z=[0, h],
                    mode="lines",
                    line=dict(color=self.COLORS["data"], width=8),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout for all scenes to look "flat" (billboard style)
        flat_scene = dict(
            yaxis=dict(showticklabels=False, showgrid=False, title=""),
            camera=dict(eye=dict(x=0, y=-2.5, z=0.2)),
            aspectmode='manual', aspectratio=dict(x=1, y=0.1, z=0.8)
        )
        
        fig.update_layout(
            height=700,
            template="plotly_white",
            title_text="<b>Diagnose-Dashboard</b>",
            scene1=flat_scene, scene2=flat_scene,
            scene3=flat_scene, scene4=flat_scene
        )
        
        return fig
    
    def _create_residual_histogram(self, result: RegressionResult) -> go.Figure:
        """Create 3D residual histogram."""
        fig = go.Figure()
        
        hist, bin_edges = np.histogram(result.residuals, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Draw "Bars"
        for i, (x, h) in enumerate(zip(bin_centers, hist)):
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[0, 0], z=[0, h],
                mode="lines",
                line=dict(color=self.COLORS["data"], width=15),
                showlegend=False,
                name="Bin"
            ))
            
        # Normal Curve
        x_norm = np.linspace(min(result.residuals), max(result.residuals), 100)
        from scipy import stats
        y_norm = stats.norm.pdf(x_norm, 0, np.std(result.residuals)) * len(result.residuals) * (max(result.residuals) - min(result.residuals)) / 20
        
        fig.add_trace(go.Scatter3d(
            x=x_norm, y=np.zeros(len(x_norm)), z=y_norm,
            mode="lines",
            line=dict(color=self.COLORS["model"], width=4),
            name="Normalverteilung"
        ))
        
        fig.update_layout(
            title="<b>Residuen-Verteilung</b>",
            scene=dict(
                xaxis_title="Residuum",
                yaxis_title="",
                zaxis_title="Häufigkeit",
                camera=dict(eye=dict(x=0, y=-2.5, z=0.3)),
                aspectmode='manual', aspectratio=dict(x=1, y=0.1, z=0.8)
            ),
            template="plotly_white"
        )
        return fig
    
    def _create_3d_surface(
        self,
        data: MultipleRegressionDataResult,
        result: MultipleRegressionResult,
    ) -> go.Figure:
        """Create rich 3D multiple regression visualization."""
        fig = go.Figure()
        
        # 1. The Regression Plane (Surface)
        # Create a grid
        padding = 0.1
        x1_range = np.linspace(min(data.x1)-padding, max(data.x1)+padding, 30)
        x2_range = np.linspace(min(data.x2)-padding, max(data.x2)+padding, 30)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        Y_mesh = result.intercept + result.coefficients[0] * X1_mesh + result.coefficients[1] * X2_mesh
        
        fig.add_trace(go.Surface(
            x=X1_mesh,
            y=X2_mesh,
            z=Y_mesh,
            name="Regressionsebene",
            colorscale=self.COLORS["plane"],
            opacity=0.6,
            showscale=False,
            contours=dict(
                x=dict(show=True, color="white", width=1),
                y=dict(show=True, color="white", width=1),
            ),
            hoverinfo="skip"
        ))
        
        # 2. Residual Lines (Vertical drops to plane)
        for i in range(len(data.y)):
            fig.add_trace(go.Scatter3d(
                x=[data.x1[i], data.x1[i]],
                y=[data.x2[i], data.x2[i]],
                z=[data.y[i], result.y_pred[i]],
                mode="lines",
                line=dict(color=self.COLORS["residual_pos"], width=2),
                showlegend=False,
                hoverinfo="skip"
            ))
            
        # 3. Data Points
        fig.add_trace(go.Scatter3d(
            x=data.x1,
            y=data.x2,
            z=data.y,
            mode="markers",
            name="Datenpunkte",
            marker=dict(
                size=5,
                color=self.COLORS["data"],
                line=dict(width=1, color="white"),
                opacity=0.9
            ),
            hovertemplate=f"{data.x1_label}: %{{x:.1f}}<br>{data.x2_label}: %{{y:.1f}}<br>{data.y_label}: %{{z:.1f}}<extra></extra>"
        ))
        
        # 4. Floor Projections (Shadows)
        # Project points to the bottom of the plot to show predictor distribution
        z_floor = min(min(data.y), np.min(Y_mesh)) - (max(data.y) - min(data.y)) * 0.1
        
        fig.add_trace(go.Scatter3d(
            x=data.x1,
            y=data.x2,
            z=[z_floor] * len(data.y),
            mode="markers",
            marker=dict(color="gray", size=3, opacity=0.3),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.update_layout(
            title=f"<b>Multiple Regression</b><br><sup>z = {result.intercept:.1f} + {result.coefficients[0]:.2f}x₁ + {result.coefficients[1]:.2f}x₂ (R² = {result.r_squared:.3f})</sup>",
            scene=dict(
                xaxis=dict(title=data.x1_label, backgroundcolor=self.COLORS["grid"]),
                yaxis=dict(title=data.x2_label, backgroundcolor=self.COLORS["grid"]),
                zaxis=dict(title=data.y_label, backgroundcolor=self.COLORS["grid"]),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
            ),
            template="plotly_white",
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05),
            margin=dict(l=0, r=0, b=0, t=50),
        )
        
        return fig
    
    def _create_residual_plot_multiple(
        self, result: MultipleRegressionResult
    ) -> go.Figure:
        """Create 3D residual plot for multiple regression."""
        # Re-use simple regression style but with specific title
        fig = self._create_residual_plot(result) # Compatible duck-typing on result object parts
        fig.update_layout(title="<b>Residuen (Multiple Regression)</b>")
        return fig
    
    def _create_diagnostic_plots_multiple(
        self, result: MultipleRegressionResult
    ) -> go.Figure:
        """Create diagnostic plots for multiple regression."""
        return self._create_diagnostic_plots(result)
