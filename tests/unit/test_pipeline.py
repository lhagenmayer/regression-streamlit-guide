"""
Tests for the Pipeline Module.

Tests the 4-step pipeline: GET → CALCULATE → PLOT → DISPLAY

Note: Display tests require streamlit and are skipped in CI environments.
"""

import pytest
import numpy as np

# Import only non-UI components for testing
from src.pipeline.get_data import DataFetcher, DataResult, MultipleRegressionDataResult
from src.pipeline.calculate import StatisticsCalculator, RegressionResult, MultipleRegressionResult

# Try to import plot components (may fail without plotly)
try:
    from src.pipeline.plot import PlotBuilder, PlotCollection
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    PlotBuilder = None
    PlotCollection = None

# Try to import full pipeline (may fail without streamlit)
try:
    from src.pipeline import RegressionPipeline, PipelineResult
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    RegressionPipeline = None
    PipelineResult = None


class TestDataFetcher:
    """Test Step 1: GET DATA"""
    
    def setup_method(self):
        self.fetcher = DataFetcher()
    
    def test_get_simple_electronics(self):
        """Test fetching electronics dataset."""
        data = self.fetcher.get_simple("electronics", n=50, seed=42)
        
        assert isinstance(data, DataResult)
        assert len(data.x) == 50
        assert len(data.y) == 50
        assert data.x_label == "Verkaufsfläche (100 qm)"
        assert data.context_title == "🏪 Elektronikmarkt"
    
    def test_get_simple_advertising(self):
        """Test fetching advertising dataset."""
        data = self.fetcher.get_simple("advertising", n=30, seed=42)
        
        assert isinstance(data, DataResult)
        assert len(data.x) == 30
        assert "Werbeausgaben" in data.x_label
    
    def test_get_simple_reproducibility(self):
        """Test that same seed produces same data."""
        data1 = self.fetcher.get_simple("electronics", n=20, seed=123)
        data2 = self.fetcher.get_simple("electronics", n=20, seed=123)
        
        np.testing.assert_array_equal(data1.x, data2.x)
        np.testing.assert_array_equal(data1.y, data2.y)
    
    def test_get_multiple_cities(self):
        """Test fetching cities dataset for multiple regression."""
        data = self.fetcher.get_multiple("cities", n=75, seed=42)
        
        assert isinstance(data, MultipleRegressionDataResult)
        assert len(data.x1) == 75
        assert len(data.x2) == 75
        assert len(data.y) == 75
        assert "Preis" in data.x1_label
        assert "Werbung" in data.x2_label
    
    def test_get_multiple_houses(self):
        """Test fetching houses dataset."""
        data = self.fetcher.get_multiple("houses", n=100, seed=42)
        
        assert isinstance(data, MultipleRegressionDataResult)
        assert len(data.y) == 100
        assert "Wohnfläche" in data.x1_label
        assert "Pool" in data.x2_label


class TestStatisticsCalculator:
    """Test Step 2: CALCULATE"""
    
    def setup_method(self):
        self.calc = StatisticsCalculator()
        np.random.seed(42)
    
    def test_simple_regression_basic(self):
        """Test basic simple regression calculation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])  # ~2x
        
        result = self.calc.simple_regression(x, y)
        
        assert isinstance(result, RegressionResult)
        assert result.n == 5
        assert result.slope > 1.5  # Should be close to 2
        assert result.r_squared > 0.95  # Good fit
    
    def test_simple_regression_perfect_fit(self):
        """Test regression with perfect linear relationship."""
        x = np.array([1, 2, 3, 4, 5])
        y = 2.0 + 3.0 * x  # Perfect line
        
        result = self.calc.simple_regression(x, y)
        
        assert abs(result.intercept - 2.0) < 0.001
        assert abs(result.slope - 3.0) < 0.001
        assert abs(result.r_squared - 1.0) < 0.001
    
    def test_simple_regression_statistics(self):
        """Test that all statistics are computed."""
        x = np.random.uniform(0, 10, 30)
        y = 1 + 2 * x + np.random.normal(0, 1, 30)
        
        result = self.calc.simple_regression(x, y)
        
        # Check all fields exist
        assert hasattr(result, 'intercept')
        assert hasattr(result, 'slope')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'r_squared_adj')
        assert hasattr(result, 'se_intercept')
        assert hasattr(result, 'se_slope')
        assert hasattr(result, 't_intercept')
        assert hasattr(result, 't_slope')
        assert hasattr(result, 'p_intercept')
        assert hasattr(result, 'p_slope')
        assert hasattr(result, 'sse')
        assert hasattr(result, 'sst')
        assert hasattr(result, 'ssr')
        
        # Check sums of squares relationship
        assert abs(result.sst - result.ssr - result.sse) < 0.001
    
    def test_multiple_regression_basic(self):
        """Test basic multiple regression calculation."""
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([5, 4, 3, 2, 1])
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, 5)
        
        result = self.calc.multiple_regression(x1, x2, y)
        
        assert isinstance(result, MultipleRegressionResult)
        assert result.n == 5
        assert result.k == 2
        assert len(result.coefficients) == 2
    
    def test_multiple_regression_statistics(self):
        """Test multiple regression statistics."""
        np.random.seed(42)
        x1 = np.random.uniform(0, 10, 50)
        x2 = np.random.uniform(0, 10, 50)
        y = 5 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 1, 50)
        
        result = self.calc.multiple_regression(x1, x2, y)
        
        assert result.r_squared > 0.8  # Should be good fit
        assert len(result.se_coefficients) == 3  # intercept + 2 coeffs
        assert len(result.t_values) == 3
        assert len(result.p_values) == 3
    
    def test_basic_stats(self):
        """Test basic statistics calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        stats = self.calc.basic_stats(data)
        
        assert stats["n"] == 10
        assert stats["mean"] == 5.5
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["median"] == 5.5


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not installed")
class TestPlotBuilder:
    """Test Step 3: PLOT"""
    
    def setup_method(self):
        self.plotter = PlotBuilder()
        self.fetcher = DataFetcher()
        self.calc = StatisticsCalculator()
    
    def test_simple_regression_plots(self):
        """Test creating simple regression plots."""
        data = self.fetcher.get_simple("electronics", n=50, seed=42)
        result = self.calc.simple_regression(data.x, data.y)
        
        plots = self.plotter.simple_regression_plots(data, result)
        
        assert isinstance(plots, PlotCollection)
        assert plots.scatter is not None
        assert plots.residuals is not None
        assert plots.diagnostics is not None
    
    def test_multiple_regression_plots(self):
        """Test creating multiple regression plots."""
        data = self.fetcher.get_multiple("cities", n=75, seed=42)
        result = self.calc.multiple_regression(data.x1, data.x2, data.y)
        
        plots = self.plotter.multiple_regression_plots(data, result)
        
        assert isinstance(plots, PlotCollection)
        assert plots.scatter is not None
        assert plots.residuals is not None


@pytest.mark.skipif(not HAS_STREAMLIT, reason="Streamlit not installed")
class TestRegressionPipeline:
    """Test complete pipeline integration."""
    
    def setup_method(self):
        self.pipeline = RegressionPipeline()
    
    def test_run_simple_pipeline(self):
        """Test complete simple regression pipeline."""
        result = self.pipeline.run_simple(
            dataset="electronics",
            n=50,
            noise=0.4,
            seed=42,
        )
        
        assert isinstance(result, PipelineResult)
        assert result.pipeline_type == "simple"
        assert isinstance(result.data, DataResult)
        assert isinstance(result.stats, RegressionResult)
        assert isinstance(result.plots, PlotCollection)
    
    def test_run_multiple_pipeline(self):
        """Test complete multiple regression pipeline."""
        result = self.pipeline.run_multiple(
            dataset="cities",
            n=75,
            noise=3.5,
            seed=42,
        )
        
        assert isinstance(result, PipelineResult)
        assert result.pipeline_type == "multiple"
        assert isinstance(result.data, MultipleRegressionDataResult)
        assert isinstance(result.stats, MultipleRegressionResult)
        assert isinstance(result.plots, PlotCollection)
    
    def test_pipeline_params_stored(self):
        """Test that pipeline parameters are stored in result."""
        result = self.pipeline.run_simple(
            dataset="temperature",
            n=30,
            noise=0.5,
            seed=123,
        )
        
        assert result.params["dataset"] == "temperature"
        assert result.params["n"] == 30
        assert result.params["noise"] == 0.5
        assert result.params["seed"] == 123
    
    def test_individual_steps(self):
        """Test using pipeline steps individually."""
        # Step 1: Get data
        data = self.pipeline.get_data("simple", dataset="electronics", n=30, seed=42)
        assert isinstance(data, DataResult)
        
        # Step 2: Calculate
        stats = self.pipeline.calculate(data, "simple")
        assert isinstance(stats, RegressionResult)
        
        # Step 3: Plot
        plots = self.pipeline.plot(data, stats, "simple")
        assert isinstance(plots, PlotCollection)


@pytest.mark.skipif(not HAS_STREAMLIT, reason="Streamlit not installed")
class TestPipelineConsistency:
    """Test that pipeline produces consistent results."""
    
    def test_same_seed_same_results(self):
        """Test reproducibility with same seed."""
        pipeline = RegressionPipeline()
        
        result1 = pipeline.run_simple(dataset="electronics", n=50, seed=42)
        result2 = pipeline.run_simple(dataset="electronics", n=50, seed=42)
        
        # Same data
        np.testing.assert_array_equal(result1.data.x, result2.data.x)
        np.testing.assert_array_equal(result1.data.y, result2.data.y)
        
        # Same statistics
        assert result1.stats.r_squared == result2.stats.r_squared
        assert result1.stats.slope == result2.stats.slope
    
    def test_different_seed_different_results(self):
        """Test that different seeds produce different results."""
        pipeline = RegressionPipeline()
        
        result1 = pipeline.run_simple(dataset="electronics", n=50, seed=42)
        result2 = pipeline.run_simple(dataset="electronics", n=50, seed=123)
        
        # Data should differ
        assert not np.array_equal(result1.data.x, result2.data.x)
