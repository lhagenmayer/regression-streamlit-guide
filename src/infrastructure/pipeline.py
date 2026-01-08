"""
Pipeline Package - Simple 4-step data processing pipeline.
Migrated to infrastructure layer.

The pipeline follows a clear flow:
    1. GET      → Fetch/generate data
    2. CALCULATE → Compute statistics & fit models
    3. PLOT     → Create visualizations
    4. DISPLAY  → Render in UI

Usage:
    from src.infrastructure import RegressionPipeline
    
    pipeline = RegressionPipeline()
    result = pipeline.run(dataset="electronics", n=50)
"""

# Core components from migrated infrastructure
from .data.generators import DataFetcher
from .services.calculate import StatisticsCalculator

# Lazy imports for components with external dependencies
def get_plot_builder():
    """Lazy import PlotBuilder (requires plotly)."""
    from .services.plot import PlotBuilder
    return PlotBuilder

# For convenience, try to import full pipeline
try:
    from .services.plot import PlotBuilder, PlotCollection
    from .regression_pipeline import RegressionPipeline, PipelineResult
except ImportError:
    PlotBuilder = None
    PlotCollection = None
    RegressionPipeline = None
    PipelineResult = None

__all__ = [
    # Core pipeline
    'RegressionPipeline',
    'PipelineResult',
    # Individual steps
    'DataFetcher',
    'StatisticsCalculator', 
    'PlotBuilder',
    'PlotCollection',
    # Lazy loaders
    'get_plot_builder',
]
