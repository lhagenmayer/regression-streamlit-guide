"""
Content Module - Framework-Agnostic Educational Content.

This module defines educational content as DATA STRUCTURES,
not as UI code. Renderers then interpret these structures.

Structure:
    ContentBuilder → produces → ContentStructure (data)
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
          StreamlitRenderer                      FlaskRenderer
          (st.markdown, etc.)                   (HTML/Jinja2)
"""

from .structure import (
    ContentElement,
    Chapter,
    Section,
    Markdown,
    Metric,
    MetricRow,
    Formula,
    Plot,
    Table,
    Columns,
    Expander,
    InfoBox,
    WarningBox,
    SuccessBox,
    CodeBlock,
)

from .builder import ContentBuilder
from .simple_regression import SimpleRegressionContent
from .multiple_regression import MultipleRegressionContent

__all__ = [
    # Structure elements
    "ContentElement",
    "Chapter", 
    "Section",
    "Markdown",
    "Metric",
    "MetricRow",
    "Formula",
    "Plot",
    "Table",
    "Columns",
    "Expander",
    "InfoBox",
    "WarningBox",
    "SuccessBox",
    "CodeBlock",
    # Builders
    "ContentBuilder",
    "SimpleRegressionContent",
    "MultipleRegressionContent",
]
