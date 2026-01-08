"""
Content Builder - Base class for building educational content.

This is framework-agnostic: it produces ContentStructures
that any renderer can interpret.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

from .structure import (
    ContentElement, Chapter, Section, EducationalContent,
    Markdown, Metric, MetricRow, Formula, Plot, Table,
    Columns, Expander, InfoBox, WarningBox, SuccessBox,
    CodeBlock, Divider
)


class ContentBuilder(ABC):
    """Abstract base class for content builders."""
    
    def __init__(self, stats: Dict[str, Any], plots: Dict[str, Any]):
        """
        Initialize content builder.
        
        Args:
            stats: Statistical results from pipeline
            plots: Plot keys/figures from pipeline
        """
        self.stats = stats
        self.plots = plots
    
    @abstractmethod
    def build(self) -> EducationalContent:
        """Build complete educational content."""
        pass
    
    # Helper methods for formatting
    def fmt(self, value: float, decimals: int = 4) -> str:
        """Format numeric value."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.{decimals}f}"
    
    def fmt_pct(self, value: float) -> str:
        """Format as percentage."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value*100:.2f}%"
    
    def interpret_r2(self, r2: float) -> str:
        """Interpret R² value."""
        if r2 >= 0.9:
            return "Exzellent"
        elif r2 >= 0.7:
            return "Gut"
        elif r2 >= 0.5:
            return "Moderat"
        elif r2 >= 0.3:
            return "Schwach"
        return "Sehr schwach"
    
    def interpret_p_value(self, p: float) -> str:
        """Interpret p-value."""
        if p < 0.001:
            return "Höchst signifikant (***)"
        elif p < 0.01:
            return "Sehr signifikant (**)"
        elif p < 0.05:
            return "Signifikant (*)"
        elif p < 0.1:
            return "Marginal signifikant"
        return "Nicht signifikant"
    
    def interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        abs_r = abs(r)
        direction = "positive" if r > 0 else "negative"
        if abs_r >= 0.9:
            return f"Sehr starke {direction} Korrelation"
        elif abs_r >= 0.7:
            return f"Starke {direction} Korrelation"
        elif abs_r >= 0.5:
            return f"Moderate {direction} Korrelation"
        elif abs_r >= 0.3:
            return f"Schwache {direction} Korrelation"
        return "Keine/sehr schwache Korrelation"
    
    def interpret_vif(self, vif: float) -> str:
        """Interpret VIF value."""
        if vif < 5:
            return "✅ Keine problematische Multikollinearität"
        elif vif < 10:
            return "⚠️ Moderate Multikollinearität"
        return "🚨 Starke Multikollinearität - problematisch!"
    
    def interpret_durbin_watson(self, dw: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if 1.5 <= dw <= 2.5:
            return "✅ Keine signifikante Autokorrelation"
        elif dw < 1.5:
            return "⚠️ Positive Autokorrelation möglich"
        return "⚠️ Negative Autokorrelation möglich"
    
    # Common content building blocks
    def make_metric_row(self, metrics: List[tuple]) -> MetricRow:
        """Create a row of metrics from (label, value, help) tuples."""
        return MetricRow([
            Metric(label=m[0], value=str(m[1]), help_text=m[2] if len(m) > 2 else "")
            for m in metrics
        ])
    
    def make_two_columns(
        self, 
        left: List[ContentElement], 
        right: List[ContentElement],
        widths: Optional[List[float]] = None
    ) -> Columns:
        """Create two-column layout."""
        return Columns([left, right], widths or [1.0, 1.0])
    
    def make_three_columns(
        self,
        col1: List[ContentElement],
        col2: List[ContentElement], 
        col3: List[ContentElement],
        widths: Optional[List[float]] = None
    ) -> Columns:
        """Create three-column layout."""
        return Columns([col1, col2, col3], widths or [1.0, 1.0, 1.0])
