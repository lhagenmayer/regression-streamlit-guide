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
from ..data.generators import DataResult, MultipleRegressionDataResult, ClassificationDataResult
from .calculate import RegressionResult, MultipleRegressionResult
from ...core.domain.value_objects import ClassificationResult
from .ml_bridge import LossSurfaceResult, OverfittingDemoResult

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

    def ml_bridge_plots(
        self,
        loss_data: Optional[LossSurfaceResult] = None,
        overfit_data: Optional[OverfittingDemoResult] = None,
    ) -> Dict[str, go.Figure]:
        """
        Create specialized plots for ML bridge chapters.
        Returns a dict of named figures.
        """
        plots = {}
        
        if loss_data:
            plots["loss_surface"] = self._create_loss_surface(loss_data)
        
        if overfit_data:
            plots["overfitting_demo"] = self._create_overfitting_plot(overfit_data)
            
        return plots
        
    def classification_plots(
        self,
        data: ClassificationDataResult,
        result: ClassificationResult,
    ) -> PlotCollection:
        """
        Create plots for classification (Logistic/KNN).
        """
        logger.info("Creating classification plots")
        
        # Main visualization depends on dimensions
        if data.X.shape[1] == 2:
            main_plot = self._create_classification_3d(data, result)
        else:
            # Fallback for > 2 dimensions (e.g. PCA or just 2D projection)
            # For now, we reuse the 3d creator but it might just pick first 2 features
            main_plot = self._create_classification_3d(data, result)
            
        # Confusion Matrix
        conf_matrix = self._create_confusion_matrix_plot(result)
        
        # ROC Curve (if probabilities available)
        roc_curve = self._create_roc_curve_plot(data, result)
        
        return PlotCollection(
            scatter=main_plot,  # We abuse 'scatter' for the main viz
            residuals=conf_matrix,  # Abuse 'residuals' for Confusion Matrix
            diagnostics=roc_curve,   # Abuse 'diagnostics' for ROC
            extra={}
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

    def _create_loss_surface(self, data: LossSurfaceResult) -> go.Figure:
        """Create 3D Loss Surface with Gradient Descent Path."""
        fig = go.Figure()

        # 1. Surface
        fig.add_trace(go.Surface(
            z=data.loss_grid,
            x=data.w_grid,
            y=data.b_grid,
            colorscale='Viridis',
            opacity=0.8,
            name='Loss Surface (MSE)'
        ))

        # 2. Path
        # Lift path slightly above surface to avoid clipping
        path_z = [z + 0.01 for z in data.path_loss]
        
        fig.add_trace(go.Scatter3d(
            x=data.path_w,
            y=data.path_b,
            z=path_z,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=4, color='yellow'),
            name='Gradient Descent Path'
        ))

        # 3. Optimum
        fig.add_trace(go.Scatter3d(
            x=[data.optimal_w],
            y=[data.optimal_b],
            z=[np.min(data.loss_grid)], # Approximate
            mode='markers',
            marker=dict(size=8, color='green', symbol='diamond'),
            name='Global Optimum (OLS)'
        ))

        fig.update_layout(
            title="<b>Loss Landschaft & Gradient Descent</b><br><sup>Suche nach dem Minimum des Fehlers (MSE)</sup>",
            scene=dict(
                xaxis_title='Gewicht (Slope)',
                yaxis_title='Intercept',
                zaxis_title='Loss (MSE)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            template="plotly_white"
        )
        return fig

    def _create_overfitting_plot(self, data: OverfittingDemoResult) -> go.Figure:
        """Create 3D plot showing Underfitting vs Optimal vs Overfitting (Stacked)."""
        fig = go.Figure()

        # We stack models along Y axis (Complexity)
        # Degree 1 -> y=1, Degree 3 -> y=3, Degree 12 -> y=12
        
        # 0. Ground Truth / Data (at Y=0)
        fig.add_trace(go.Scatter3d(
            x=data.x_train, y=[0]*len(data.x_train), z=data.y_train,
            mode='markers', marker=dict(color='blue', size=5),
            name='Training Data (y=0)'
        ))
        
        # 1. Models
        colors = {1: 'orange', 3: 'black', 12: 'red'}
        names = {1: 'Underfitting (d=1)', 3: 'Optimal (d=3)', 12: 'Overfitting (d=12)'}
        
        for degree in [1, 3, 12]:
            if degree in data.predictions:
                # Prediction Line
                y_pos = [degree] * len(data.x_plot)
                fig.add_trace(go.Scatter3d(
                    x=data.x_plot,
                    y=y_pos,
                    z=data.predictions[degree],
                    mode='lines',
                    line=dict(color=colors[degree], width=4),
                    name=names[degree]
                ))
                
                # Projection of data to this degree's plane for comparison
                fig.add_trace(go.Scatter3d(
                    x=data.x_train, y=[degree]*len(data.x_train), z=data.y_train,
                    mode='markers', marker=dict(color='lightgray', size=3, opacity=0.5),
                    showlegend=False
                ))

        fig.update_layout(
            title="<b>Modell-Komplexität 3D</b><br><sup>Separation nach Komplexität (Tiefe)</sup>",
            scene=dict(
                xaxis_title="Feature x",
                yaxis_title="Polynom Grad (d)",
                zaxis_title="Target y",
                camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
                xaxis=dict(backgroundcolor="white"),
                yaxis=dict(backgroundcolor="#f0f0f0"), # Highlight depth
                zaxis=dict(backgroundcolor="white"),
            ),
            template="plotly_white"
        )
        return fig

    def _create_classification_3d(
        self, 
        data: ClassificationDataResult, 
        result: ClassificationResult
    ) -> go.Figure:
        """Create 3D visualization for Classification."""
        fig = go.Figure()
        
        # 1. Data Points
        unique_classes = np.unique(data.y)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
        
        for i, c in enumerate(unique_classes):
            mask = (data.y == c)
            x1 = data.X[mask, 0]
            x2 = data.X[mask, 1] if data.X.shape[1] > 1 else np.zeros_like(x1)
            
            # 3D: Lift Class 1 up in Z axis? 
            # Or just keep points flat and show surface?
            # Let's separate them slightly in Z based on class for "3D" effect if requested,
            # but user said "no 2d". 
            # Better: Z = 0 for all, but Surface is 3D.
            
            fig.add_trace(go.Scatter3d(
                x=x1,
                y=x2,
                z=np.zeros_like(x1), 
                mode='markers',
                marker=dict(size=6, color=colors[i % len(colors)], line=dict(width=1, color='white')),
                name=f"Klasse {c}"
            ))

        # 2. Decision Surface (Probabilities)
        if result.model_params.get("coefficients") is not None and data.X.shape[1] == 2:
            x1_min, x1_max = data.X[:, 0].min(), data.X[:, 0].max()
            x2_min, x2_max = data.X[:, 1].min(), data.X[:, 1].max()
            
            grid_x1, grid_x2 = np.meshgrid(
                np.linspace(x1_min, x1_max, 30),
                np.linspace(x2_min, x2_max, 30)
            )
            
            coeffs = result.model_params["coefficients"]
            intercept = result.model_params.get("intercept", 0)
            
            c_vals = list(coeffs.values()) if isinstance(coeffs, dict) else coeffs
                
            if len(c_vals) >= 2:
                # Z = Probability
                logit = intercept + c_vals[0] * grid_x1 + c_vals[1] * grid_x2
                prob = 1 / (1 + np.exp(-logit))
                
                fig.add_trace(go.Surface(
                    x=grid_x1, y=grid_x2, z=prob,
                    colorscale='RdBu', showscale=False, opacity=0.4,
                    name='Probability Surface'
                ))
                
                # Add decision boundary at Proba=0.5 (Contour)
                # Hard in 3D surface trace directly, can add line trace
                
        fig.update_layout(
            title="<b>Klassifikations-Wahrscheinlichkeit</b>",
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="P(Class=1)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.6))
            ),
            template="plotly_white"
        )
        return fig

    def _create_confusion_matrix_plot(self, result: ClassificationResult) -> go.Figure:
        """Create 3D Lego Plot for Confusion Matrix."""
        cm = result.metrics.confusion_matrix
        classes = result.classes
        
        x_pos = []
        y_pos = []
        z_pos = []
        c_vals = []
        
        # Grid coordinates
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                # Center bars
                x_pos.append(j) # Pred on X
                y_pos.append(i) # True on Y
                z_pos.append(cm[i, j])
                
                # Color based on diagonal (Correct) vs Off-diagonal (Error)
                if i == j:
                    c_vals.append('green')
                else:
                    c_vals.append('red')

        fig = go.Figure()
        
        # 3D Bars
        for x, y, z, c in zip(x_pos, y_pos, z_pos, c_vals):
            if z > 0: # Only plot non-zero bars
                fig.add_trace(go.Scatter3d(
                    x=[x, x], y=[y, y], z=[0, z],
                    mode='lines',
                    line=dict(color=c, width=30), # Very thick lines look like bars
                    hovertemplate=f"True: {classes[y]}<br>Pred: {classes[x]}<br>Count: {z}<extra></extra>"
                ))
        
        # Ground plane for reference
        fig.add_trace(go.Scatter3d(
            x=[-0.5, len(classes)-0.5, len(classes)-0.5, -0.5, -0.5],
            y=[-0.5, -0.5, len(classes)-0.5, len(classes)-0.5, -0.5],
            z=[0,0,0,0,0],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))

        fig.update_layout(
            title="<b>3D Confusion Matrix</b>",
            scene=dict(
                xaxis=dict(title="Vorhergesagt", tickvals=list(range(len(classes))), ticktext=[str(c) for c in classes]),
                yaxis=dict(title="Wahr", tickvals=list(range(len(classes))), ticktext=[str(c) for c in classes]),
                zaxis_title="Anzahl",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            template="plotly_white"
        )
        return fig

    def _create_roc_curve_plot(
        self, 
        data: ClassificationDataResult,
        result: ClassificationResult
    ) -> go.Figure:
        """
        Create Educational 3D ROC Curve.
        
        Dimensions:
        X: False Positive Rate
        Y: True Positive Rate
        Z: Threshold (The decision boundary)
        """
        if result.probabilities is None or len(result.probabilities) == 0:
             return self._create_placeholder_3d("Keine Wahrscheinlichkeiten für ROC verfügbar")
             
        # from sklearn.metrics import roc_curve (REMOVED)
        
        # Needs binary 0/1 for ROC
        pos_label = data.y.max()
        y_true = (data.y == pos_label).astype(int)
        
        # We need probabilities for the Positive Class
        y_score = result.probabilities
        if y_score.ndim > 1:
            y_score = y_score[:, -1]
            
        # Manually compute ROC
        # 1. Sort by score descending
        desc_score_indices = np.argsort(y_score)[::-1]
        y_score_sorted = y_score[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]
        
        # 2. Compute TPR/FPR
        fps = np.cumsum(1 - y_true_sorted)
        tps = np.cumsum(y_true_sorted)
        
        # Add 0,0 point
        fps = np.r_[0, fps]
        tps = np.r_[0, tps]
        thresholds = np.r_[y_score_sorted[0] + 1e-3, y_score_sorted]
        
        # Normalize
        fpr = fps / fps[-1] if fps[-1] > 0 else fps
        tpr = tps / tps[-1] if tps[-1] > 0 else tps
        
        # Thresholds can go > 1, clip them
        thresholds = np.clip(thresholds, 0, 1)
        
        fig = go.Figure()
        
        # 3D ROC Curve
        fig.add_trace(go.Scatter3d(
            x=fpr,
            y=tpr,
            z=thresholds,
            mode='lines+markers',
            line=dict(color='purple', width=6),
            marker=dict(size=4, color='purple'),
            name='ROC Trajectory'
        ))
        
        # Floor Projection (Standard 2D ROC)
        fig.add_trace(go.Scatter3d(
            x=fpr,
            y=tpr,
            z=[0]*len(fpr),
            mode='lines',
            line=dict(color='gray', width=3, dash='dot'),
            name='Klassische 2D ROC'
        ))
        
        # Diagonal Reference
        fig.add_trace(go.Scatter3d(
            x=[0, 1], y=[0, 1], z=[0, 0],
            mode='lines',
            line=dict(color='lightgray', dash='dash'),
            showlegend=False
        ))

        fig.update_layout(
            title="<b>3D ROC Kurve</b><br><sup>Z-Achse = Threshold: Wie der Schwellenwert die Performance steuert</sup>",
            scene=dict(
                xaxis_title="False Positive Rate (Fehlalarm)",
                yaxis_title="True Positive Rate (Recall)",
                zaxis_title="Threshold (Schwellenwert)",
                camera=dict(eye=dict(x=1.8, y=0.5, z=0.5))
            ),
            template="plotly_white"
        )
        return fig

    def _create_placeholder_3d(self, text: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='text', text=[text]
        ))
        fig.update_layout(
             title="<b>Information</b>",
             scene=dict(
                 xaxis=dict(visible=False),
                 yaxis=dict(visible=False),
                 zaxis=dict(visible=False)
             )
        )
        return fig

