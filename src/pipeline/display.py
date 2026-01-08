"""
Step 4: DISPLAY (Framework-Agnostic Data Preparation)

This module prepares pipeline results for rendering.
It does NOT contain any framework-specific code.

The actual rendering is done by framework adapters:
- src/adapters/streamlit_app.py
- src/adapters/flask_app.py
"""

from typing import Dict, Any
from dataclasses import dataclass, field

from ..config import get_logger
from .get_data import DataResult, MultipleRegressionDataResult
from .calculate import RegressionResult, MultipleRegressionResult
from .plot import PlotCollection

logger = get_logger(__name__)


@dataclass
class DisplayData:
    """
    Framework-agnostic display data container.
    
    Contains all information needed for rendering,
    without any framework-specific dependencies.
    """
    analysis_type: str  # "simple" or "multiple"
    
    # Core data
    data: Any  # DataResult or MultipleRegressionDataResult
    stats: Any  # RegressionResult or MultipleRegressionResult
    plots: PlotCollection
    
    # Options
    show_formulas: bool = True
    show_true_line: bool = False
    compact_mode: bool = False
    
    # Dynamic content
    content: Dict[str, Any] = field(default_factory=dict)
    formulas: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    dataset_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template engines."""
        return {
            "analysis_type": self.analysis_type,
            "show_formulas": self.show_formulas,
            "show_true_line": self.show_true_line,
            "compact_mode": self.compact_mode,
            "content": self.content,
            "formulas": self.formulas,
            "dataset_name": self.dataset_name,
            "data": self._serialize_data(),
            "stats": self._serialize_stats(),
        }
    
    def _serialize_data(self) -> Dict[str, Any]:
        """Serialize data to dictionary."""
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
    
    def _serialize_stats(self) -> Dict[str, Any]:
        """Serialize stats to dictionary."""
        if isinstance(self.stats, RegressionResult):
            return {
                "type": "simple",
                "intercept": float(self.stats.intercept),
                "slope": float(self.stats.slope),
                "r_squared": float(self.stats.r_squared),
                "r_squared_adj": float(self.stats.r_squared_adj),
                "se_intercept": float(self.stats.se_intercept),
                "se_slope": float(self.stats.se_slope),
                "t_intercept": float(self.stats.t_intercept),
                "t_slope": float(self.stats.t_slope),
                "p_intercept": float(self.stats.p_intercept),
                "p_slope": float(self.stats.p_slope),
                "sse": float(self.stats.sse),
                "sst": float(self.stats.sst),
                "ssr": float(self.stats.ssr),
                "mse": float(self.stats.mse),
                "n": self.stats.n,
                "df": self.stats.df,
            }
        elif isinstance(self.stats, MultipleRegressionResult):
            return {
                "type": "multiple",
                "intercept": float(self.stats.intercept),
                "coefficients": [float(c) for c in self.stats.coefficients],
                "r_squared": float(self.stats.r_squared),
                "r_squared_adj": float(self.stats.r_squared_adj),
                "f_statistic": float(self.stats.f_statistic),
                "f_pvalue": float(self.stats.f_pvalue),
                "se_coefficients": [float(se) for se in self.stats.se_coefficients],
                "t_values": [float(t) for t in self.stats.t_values],
                "p_values": [float(p) for p in self.stats.p_values],
                "n": self.stats.n,
                "k": self.stats.k,
            }
        return {}


class DisplayPreparer:
    """
    Prepares pipeline results for display.
    
    This class is framework-agnostic and only prepares data.
    The actual rendering is done by framework adapters.
    """
    
    def prepare_simple(
        self,
        data: DataResult,
        stats: RegressionResult,
        plots: PlotCollection,
        show_formulas: bool = True,
        show_true_line: bool = False,
        compact_mode: bool = False,
        dataset_name: str = "",
    ) -> DisplayData:
        """Prepare simple regression for display."""
        return DisplayData(
            analysis_type="simple",
            data=data,
            stats=stats,
            plots=plots,
            show_formulas=show_formulas,
            show_true_line=show_true_line,
            compact_mode=compact_mode,
            dataset_name=dataset_name,
        )
    
    def prepare_multiple(
        self,
        data: MultipleRegressionDataResult,
        stats: MultipleRegressionResult,
        plots: PlotCollection,
        content: Dict[str, Any] = None,
        formulas: Dict[str, str] = None,
        show_formulas: bool = True,
        compact_mode: bool = False,
        dataset_name: str = "",
    ) -> DisplayData:
        """Prepare multiple regression for display."""
        return DisplayData(
            analysis_type="multiple",
            data=data,
            stats=stats,
            plots=plots,
            show_formulas=show_formulas,
            compact_mode=compact_mode,
            content=content or {},
            formulas=formulas or {},
            dataset_name=dataset_name,
        )
