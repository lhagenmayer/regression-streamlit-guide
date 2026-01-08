"""
Pipeline Package - Simple 4-step data processing pipeline.

The pipeline follows a clear flow:
    1. GET      → Fetch/generate data
    2. CALCULATE → Compute statistics & fit models
    3. PLOT     → Create visualizations
    4. DISPLAY  → Render in UI

Usage:
    from src.pipeline import RegressionPipeline
    
    pipeline = RegressionPipeline()
    result = pipeline.run(dataset="electronics", n=50)
"""

# Core components (no external UI dependencies)
from .get_data import DataFetcher
from .calculate import StatisticsCalculator

# Lazy imports for components with external dependencies
def get_plot_builder():
    """Lazy import PlotBuilder (requires plotly)."""
    from .plot import PlotBuilder
    return PlotBuilder

def get_ui_renderer():
    """Lazy import UIRenderer (requires streamlit)."""
    from .display import UIRenderer
    return UIRenderer

def get_pipeline():
    """Lazy import RegressionPipeline (requires plotly)."""
    from .regression_pipeline import RegressionPipeline, PipelineResult
    return RegressionPipeline, PipelineResult

# For convenience, try to import full pipeline
try:
    from .plot import PlotBuilder
    from .regression_pipeline import RegressionPipeline, PipelineResult
except ImportError:
    PlotBuilder = None
    RegressionPipeline = None
    PipelineResult = None

try:
    from .display import UIRenderer
except ImportError:
    UIRenderer = None

__all__ = [
    # Core pipeline
    'RegressionPipeline',
    'PipelineResult',
    # Individual steps
    'DataFetcher',
    'StatisticsCalculator', 
    'PlotBuilder',
    'UIRenderer',
    # Lazy loaders
    'get_plot_builder',
    'get_ui_renderer',
    'get_pipeline',
]
