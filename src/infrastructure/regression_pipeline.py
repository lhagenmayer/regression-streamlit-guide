"""
Regression Pipeline - Simple 4-step data processing.

This module provides a unified pipeline that orchestrates:
    1. GET      → Fetch data
    2. CALCULATE → Compute statistics
    3. PLOT     → Create visualizations
    4. DISPLAY  → Render in UI
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

from ..config import get_logger
from .data.generators import DataFetcher, DataResult, MultipleRegressionDataResult
from .services.calculate import StatisticsCalculator, RegressionResult, MultipleRegressionResult
from .services.plot import PlotBuilder, PlotCollection

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """
    Complete result from pipeline execution.
    
    Contains all data, calculations, and plots for display.
    """
    # Step 1: Data
    data: Union[DataResult, MultipleRegressionDataResult]
    
    # Step 2: Calculations
    stats: Union[RegressionResult, MultipleRegressionResult]
    
    # Step 3: Plots
    plots: PlotCollection
    
    # Metadata
    pipeline_type: str  # "simple" or "multiple"
    params: Dict[str, Any]


class RegressionPipeline:
    """
    Simple 4-step regression analysis pipeline.
    
    The pipeline follows a clear, linear flow:
    
        ┌─────────┐    ┌───────────┐    ┌──────┐    ┌─────────┐
        │   GET   │ → │ CALCULATE │ → │ PLOT │ → │ DISPLAY │
        └─────────┘    └───────────┘    └──────┘    └─────────┘
    
    Example:
        pipeline = RegressionPipeline()
        
        # Run complete pipeline
        result = pipeline.run_simple(
            dataset="electronics",
            n=50,
            noise=0.4,
            seed=42
        )
        
        # Display in Streamlit
        pipeline.display(result)
    """
    
    def __init__(self):
        """Initialize pipeline with all components."""
        self.fetcher = DataFetcher()
        self.calculator = StatisticsCalculator()
        self.plotter = PlotBuilder()
        
        # Import display lazily to avoid Streamlit import issues
        self._renderer = None
        
        logger.info("RegressionPipeline initialized")
    
    @property
    def renderer(self):
        """Lazy load UIRenderer to avoid import issues."""
        if self._renderer is None:
            from .display import UIRenderer
            self._renderer = UIRenderer()
        return self._renderer
    
    def run_simple(
        self,
        dataset: str = "electronics",
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
        true_intercept: float = 0.6,
        true_slope: float = 0.52,
        show_true_line: bool = True,
    ) -> PipelineResult:
        """
        Run complete simple regression pipeline.
        
        Args:
            dataset: Dataset name
            n: Sample size
            noise: Noise level
            seed: Random seed
            true_intercept: True β₀ (for simulated data)
            true_slope: True β₁ (for simulated data)
            show_true_line: Show true regression line
        
        Returns:
            PipelineResult with data, stats, and plots
        """
        logger.info(f"Running simple regression pipeline: {dataset}, n={n}")
        
        # Step 1: GET DATA
        data = self.fetcher.get_simple(
            dataset=dataset,
            n=n,
            noise=noise,
            seed=seed,
            true_intercept=true_intercept,
            true_slope=true_slope,
        )
        
        # Step 2: CALCULATE
        stats = self.calculator.simple_regression(data.x, data.y)
        
        # Step 3: PLOT
        plots = self.plotter.simple_regression_plots(
            data=data,
            result=stats,
            show_true_line=show_true_line,
            true_intercept=true_intercept,
            true_slope=true_slope,
        )
        
        params = {
            "dataset": dataset,
            "n": n,
            "noise": noise,
            "seed": seed,
            "true_intercept": true_intercept,
            "true_slope": true_slope,
            "show_true_line": show_true_line,
        }
        
        return PipelineResult(
            data=data,
            stats=stats,
            plots=plots,
            pipeline_type="simple",
            params=params,
        )
    
    def run_multiple(
        self,
        dataset: str = "cities",
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
    ) -> PipelineResult:
        """
        Run complete multiple regression pipeline.
        
        Args:
            dataset: Dataset name ("cities" or "houses")
            n: Sample size
            noise: Noise level
            seed: Random seed
        
        Returns:
            PipelineResult with data, stats, and plots
        """
        logger.info(f"Running multiple regression pipeline: {dataset}, n={n}")
        
        # Step 1: GET DATA
        data = self.fetcher.get_multiple(
            dataset=dataset,
            n=n,
            noise=noise,
            seed=seed,
        )
        
        # Step 2: CALCULATE
        stats = self.calculator.multiple_regression(data.x1, data.x2, data.y)
        
        # Step 3: PLOT
        plots = self.plotter.multiple_regression_plots(data=data, result=stats)
        
        params = {
            "dataset": dataset,
            "n": n,
            "noise": noise,
            "seed": seed,
        }
        
        return PipelineResult(
            data=data,
            stats=stats,
            plots=plots,
            pipeline_type="multiple",
            params=params,
        )
    
    def display(
        self,
        result: PipelineResult,
        show_formulas: bool = True,
    ) -> None:
        """
        Step 4: DISPLAY - Render results in Streamlit.
        
        Args:
            result: PipelineResult from run_simple or run_multiple
            show_formulas: Whether to show mathematical formulas
        """
        if result.pipeline_type == "simple":
            self.renderer.simple_regression(
                data=result.data,
                result=result.stats,
                plots=result.plots,
                show_formulas=show_formulas,
            )
        else:
            self.renderer.multiple_regression(
                data=result.data,
                result=result.stats,
                plots=result.plots,
                show_formulas=show_formulas,
            )
    
    # =========================================================
    # Convenience methods for individual steps
    # =========================================================
    
    def get_data(self, regression_type: str = "simple", **kwargs):
        """Step 1: Get data only."""
        if regression_type == "simple":
            return self.fetcher.get_simple(**kwargs)
        else:
            return self.fetcher.get_multiple(**kwargs)
    
    def calculate(self, data, regression_type: str = "simple"):
        """Step 2: Calculate only."""
        if regression_type == "simple":
            return self.calculator.simple_regression(data.x, data.y)
        else:
            return self.calculator.multiple_regression(data.x1, data.x2, data.y)
    
    def plot(self, data, stats, regression_type: str = "simple", **kwargs):
        """Step 3: Plot only."""
        if regression_type == "simple":
            return self.plotter.simple_regression_plots(data, stats, **kwargs)
        else:
            return self.plotter.multiple_regression_plots(data, stats)
