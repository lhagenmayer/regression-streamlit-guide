"""
Base Renderer - Abstract interface for framework-agnostic rendering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json

from ..infrastructure import DataResult, MultipleRegressionDataResult
from ..infrastructure import RegressionResult, MultipleRegressionResult
from ..infrastructure import PlotCollection


@dataclass
class RenderContext:
    """
    Framework-agnostic context for rendering.
    
    Contains all data needed to render the regression analysis
    without any framework-specific dependencies.
    """
    # Analysis type
    analysis_type: str  # "simple" or "multiple"
    
    # Data
    data: Any  # DataResult or MultipleRegressionDataResult
    stats: Any  # RegressionResult or MultipleRegressionResult
    
    # Plots as JSON (Plotly figures serialized)
    plots_json: Dict[str, str] = field(default_factory=dict)
    
    # Display options
    show_formulas: bool = True
    show_true_line: bool = False
    compact_mode: bool = False
    
    # Dynamic content
    content: Dict[str, Any] = field(default_factory=dict)
    formulas: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    dataset_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            "analysis_type": self.analysis_type,
            "dataset_name": self.dataset_name,
            "show_formulas": self.show_formulas,
            "compact_mode": self.compact_mode,
            "plots": self.plots_json,
            "content": self.content,
            "formulas": self.formulas,
            "stats": self._stats_to_dict(),
            "data": self._data_to_dict(),
        }
    
    def _stats_to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        if isinstance(self.stats, RegressionResult):
            return {
                "type": "simple",
                "intercept": self.stats.intercept,
                "slope": self.stats.slope,
                "r_squared": self.stats.r_squared,
                "r_squared_adj": self.stats.r_squared_adj,
                "p_slope": self.stats.p_slope,
                "t_slope": self.stats.t_slope,
                "se_slope": self.stats.se_slope,
                "n": self.stats.n,
                "df": self.stats.df,
                "mse": self.stats.mse,
                "sse": self.stats.sse,
                "sst": self.stats.sst,
                "ssr": self.stats.ssr,
            }
        elif isinstance(self.stats, MultipleRegressionResult):
            return {
                "type": "multiple",
                "intercept": self.stats.intercept,
                "coefficients": list(self.stats.coefficients),
                "r_squared": self.stats.r_squared,
                "r_squared_adj": self.stats.r_squared_adj,
                "f_statistic": self.stats.f_statistic,
                "f_pvalue": self.stats.f_pvalue,
                "p_values": list(self.stats.p_values),
                "t_values": list(self.stats.t_values),
                "n": self.stats.n,
                "k": self.stats.k,
            }
        return {}
    
    def _data_to_dict(self) -> Dict[str, Any]:
        """Convert data to dictionary."""
        if isinstance(self.data, DataResult):
            return {
                "type": "simple",
                "x_label": self.data.x_label,
                "y_label": self.data.y_label,
                "x_unit": self.data.x_unit,
                "y_unit": self.data.y_unit,
                "context_title": self.data.context_title,
                "context_description": self.data.context_description,
                "n": len(self.data.x),
            }
        elif isinstance(self.data, MultipleRegressionDataResult):
            return {
                "type": "multiple",
                "x1_label": self.data.x1_label,
                "x2_label": self.data.x2_label,
                "y_label": self.data.y_label,
                "n": len(self.data.y),
            }
        return {}


class BaseRenderer(ABC):
    """
    Abstract base class for framework-specific renderers.
    
    Subclasses implement the actual rendering logic for
    Streamlit, Flask, or other frameworks.
    """
    
    @abstractmethod
    def render(self, context: RenderContext) -> Any:
        """
        Render the regression analysis.
        
        Args:
            context: RenderContext with all data and options
            
        Returns:
            Framework-specific response (None for Streamlit, Response for Flask)
        """
        pass
    
    @abstractmethod
    def render_simple_regression(self, context: RenderContext) -> Any:
        """Render simple regression analysis."""
        pass
    
    @abstractmethod
    def render_multiple_regression(self, context: RenderContext) -> Any:
        """Render multiple regression analysis."""
        pass
    
    @abstractmethod
    def run(self, host: str = "0.0.0.0", port: int = 8501, debug: bool = False) -> None:
        """
        Run the application server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            debug: Enable debug mode
        """
        pass
    
    def serialize_plots(self, plots: PlotCollection) -> Dict[str, str]:
        """
        Serialize Plotly figures to JSON for framework-agnostic transport.
        
        Args:
            plots: PlotCollection from pipeline
            
        Returns:
            Dictionary of plot names to JSON strings
        """
        result = {}
        
        if plots.scatter is not None:
            result["scatter"] = plots.scatter.to_json()
        if plots.residuals is not None:
            result["residuals"] = plots.residuals.to_json()
        if plots.diagnostics is not None:
            result["diagnostics"] = plots.diagnostics.to_json()
        
        for name, fig in plots.extra.items():
            if fig is not None:
                result[name] = fig.to_json()
        
        return result
