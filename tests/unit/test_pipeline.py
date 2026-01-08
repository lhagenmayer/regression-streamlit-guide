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
from src.pipeline.display import DisplayPreparer, DisplayData

# Try to import plot components (may fail without plotly)
try:
    from src.pipeline.plot import PlotBuilder, PlotCollection
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    PlotBuilder = None
    PlotCollection = None

# Try to import full pipeline
try:
    from src.pipeline import RegressionPipeline, PipelineResult
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
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
        data = self.fetcher.get_simple("advertising", n=75, seed=42)
        
        assert isinstance(data, DataResult)
        assert len(data.x) == 75
        assert "Werbeausgaben" in data.x_label
    
    def test_get_simple_reproducibility(self):
        """Test that same seed gives same data."""
        data1 = self.fetcher.get_simple("electronics", n=50, seed=42)
        data2 = self.fetcher.get_simple("electronics", n=50, seed=42)
        
        np.testing.assert_array_equal(data1.x, data2.x)
        np.testing.assert_array_equal(data1.y, data2.y)
    
    def test_get_multiple_cities(self):
        """Test fetching cities dataset."""
        data = self.fetcher.get_multiple("cities", n=75, seed=42)
        
        assert isinstance(data, MultipleRegressionDataResult)
        assert len(data.x1) == 75
        assert len(data.x2) == 75
        assert len(data.y) == 75
    
    def test_get_multiple_houses(self):
        """Test fetching houses dataset."""
        data = self.fetcher.get_multiple("houses", n=100, seed=42)
        
        assert isinstance(data, MultipleRegressionDataResult)
        assert len(data.y) == 100


class TestStatisticsCalculator:
    """Test Step 2: CALCULATE"""
    
    def setup_method(self):
        self.calc = StatisticsCalculator()
        self.fetcher = DataFetcher()
    
    def test_simple_regression_basic(self):
        """Test simple regression calculation."""
        data = self.fetcher.get_simple("electronics", n=50, seed=42)
        result = self.calc.simple_regression(data.x, data.y)
        
        assert isinstance(result, RegressionResult)
        assert 0 <= result.r_squared <= 1
        assert result.n == 50
    
    def test_simple_regression_perfect_fit(self):
        """Test with perfectly linear data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x
        
        result = self.calc.simple_regression(x, y)
        
        assert result.slope == pytest.approx(2.0, rel=1e-10)
        assert result.intercept == pytest.approx(0.0, abs=1e-10)
        assert result.r_squared == pytest.approx(1.0, rel=1e-10)
    
    def test_simple_regression_statistics(self):
        """Test that all statistics are computed."""
        data = self.fetcher.get_simple("electronics", n=100, seed=42)
        result = self.calc.simple_regression(data.x, data.y)
        
        # Check all required fields
        assert hasattr(result, 'slope')
        assert hasattr(result, 'intercept')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'r_squared_adj')
        assert hasattr(result, 'se_slope')
        assert hasattr(result, 't_slope')
        assert hasattr(result, 'p_slope')
        assert hasattr(result, 'sse')
        assert hasattr(result, 'sst')
        
        # Validate sum of squares relationship
        assert result.ssr == pytest.approx(result.sst - result.sse, rel=1e-5)
    
    def test_multiple_regression_basic(self):
        """Test multiple regression calculation."""
        data = self.fetcher.get_multiple("cities", n=75, seed=42)
        result = self.calc.multiple_regression(data.x1, data.x2, data.y)
        
        assert isinstance(result, MultipleRegressionResult)
        assert 0 <= result.r_squared <= 1
        assert result.n == 75
        assert result.k == 2
    
    def test_multiple_regression_statistics(self):
        """Test that all multiple regression statistics are computed."""
        data = self.fetcher.get_multiple("cities", n=100, seed=42)
        result = self.calc.multiple_regression(data.x1, data.x2, data.y)
        
        # Check all required fields
        assert len(result.coefficients) == 2
        assert len(result.se_coefficients) == 3  # Including intercept
        assert len(result.t_values) == 3
        assert len(result.p_values) == 3
        assert hasattr(result, 'f_statistic')
        assert hasattr(result, 'f_pvalue')
    
    def test_basic_stats(self):
        """Test basic statistics calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = self.calc.basic_stats(x)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['mean'] == pytest.approx(3.0)


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
class TestPlotBuilder:
    """Test Step 3: PLOT"""
    
    def setup_method(self):
        self.plotter = PlotBuilder()
        self.fetcher = DataFetcher()
        self.calc = StatisticsCalculator()
    
    def test_simple_regression_plots(self):
        """Test that simple regression plots are created."""
        data = self.fetcher.get_simple("electronics", n=50, seed=42)
        stats = self.calc.simple_regression(data.x, data.y)
        
        plots = self.plotter.simple_regression_plots(data, stats)
        
        assert isinstance(plots, PlotCollection)
        assert plots.scatter is not None
        assert plots.residuals is not None
    
    def test_multiple_regression_plots(self):
        """Test that multiple regression plots are created."""
        data = self.fetcher.get_multiple("cities", n=75, seed=42)
        stats = self.calc.multiple_regression(data.x1, data.x2, data.y)
        
        plots = self.plotter.multiple_regression_plots(data, stats)
        
        assert isinstance(plots, PlotCollection)
        assert plots.scatter is not None  # 3D plot


class TestDisplayPreparer:
    """Test Step 4: DISPLAY (data preparation)"""
    
    def setup_method(self):
        self.preparer = DisplayPreparer()
        self.fetcher = DataFetcher()
        self.calc = StatisticsCalculator()
    
    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
    def test_prepare_simple(self):
        """Test preparing simple regression for display."""
        data = self.fetcher.get_simple("electronics", n=50, seed=42)
        stats = self.calc.simple_regression(data.x, data.y)
        
        plotter = PlotBuilder()
        plots = plotter.simple_regression_plots(data, stats)
        
        display_data = self.preparer.prepare_simple(data, stats, plots)
        
        assert isinstance(display_data, DisplayData)
        assert display_data.analysis_type == "simple"
    
    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")  
    def test_prepare_multiple(self):
        """Test preparing multiple regression for display."""
        data = self.fetcher.get_multiple("cities", n=75, seed=42)
        stats = self.calc.multiple_regression(data.x1, data.x2, data.y)
        
        plotter = PlotBuilder()
        plots = plotter.multiple_regression_plots(data, stats)
        
        display_data = self.preparer.prepare_multiple(data, stats, plots)
        
        assert isinstance(display_data, DisplayData)
        assert display_data.analysis_type == "multiple"
    
    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
    def test_serialization(self):
        """Test that display data can be serialized."""
        data = self.fetcher.get_simple("electronics", n=50, seed=42)
        stats = self.calc.simple_regression(data.x, data.y)
        
        plotter = PlotBuilder()
        plots = plotter.simple_regression_plots(data, stats)
        
        display_data = self.preparer.prepare_simple(data, stats, plots)
        serialized = display_data.to_dict()
        
        assert isinstance(serialized, dict)
        assert "analysis_type" in serialized
        assert "stats" in serialized
        assert serialized["stats"]["r_squared"] > 0


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline components not available")
class TestRegressionPipeline:
    """Test the complete pipeline."""
    
    def setup_method(self):
        self.pipeline = RegressionPipeline()
    
    def test_run_simple_pipeline(self):
        """Test running simple regression pipeline."""
        result = self.pipeline.run_simple(
            dataset="electronics",
            n=50,
            seed=42
        )
        
        assert isinstance(result, PipelineResult)
        assert result.data is not None
        assert result.stats is not None
        assert result.plots is not None
        assert result.pipeline_type == "simple"
    
    def test_run_multiple_pipeline(self):
        """Test running multiple regression pipeline."""
        result = self.pipeline.run_multiple(
            dataset="cities",
            n=75,
            seed=42
        )
        
        assert isinstance(result, PipelineResult)
        assert result.pipeline_type == "multiple"
    
    def test_pipeline_params_stored(self):
        """Test that pipeline parameters are stored."""
        result = self.pipeline.run_simple(
            dataset="electronics",
            n=100,
            seed=123
        )
        
        assert result.params["dataset"] == "electronics"
        assert result.params["n"] == 100
        assert result.params["seed"] == 123
    
    def test_individual_steps(self):
        """Test running individual pipeline steps."""
        # Step 1: Get data
        data = self.pipeline.get_data("simple", dataset="electronics", n=50, seed=42)
        assert isinstance(data, DataResult)
        
        # Step 2: Calculate
        stats = self.pipeline.calculate(data, "simple")
        assert isinstance(stats, RegressionResult)
        
        # Step 3: Plot
        plots = self.pipeline.plot(data, stats, "simple")
        assert plots is not None


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline components not available")
class TestPipelineConsistency:
    """Test pipeline consistency and reproducibility."""
    
    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        pipeline1 = RegressionPipeline()
        pipeline2 = RegressionPipeline()
        
        result1 = pipeline1.run_simple("electronics", n=50, seed=42)
        result2 = pipeline2.run_simple("electronics", n=50, seed=42)
        
        assert result1.stats.r_squared == result2.stats.r_squared
        assert result1.stats.slope == result2.stats.slope
    
    def test_different_seed_different_results(self):
        """Test that different seeds produce different results."""
        pipeline = RegressionPipeline()
        
        result1 = pipeline.run_simple("electronics", n=50, seed=42)
        result2 = pipeline.run_simple("electronics", n=50, seed=123)
        
        # Results should be different (with high probability)
        assert result1.stats.slope != result2.stats.slope


class TestFrameworkDetection:
    """Test framework detection and adapters."""
    
    def test_detector_import(self):
        """Test that detector can be imported."""
        from src.adapters.detector import FrameworkDetector, Framework
        
        assert Framework.STREAMLIT.value == "streamlit"
        assert Framework.FLASK.value == "flask"
    
    def test_detector_methods(self):
        """Test detector methods exist."""
        from src.adapters.detector import FrameworkDetector
        
        FrameworkDetector.reset()
        
        # In test environment, should detect as unknown
        framework = FrameworkDetector.detect()
        assert framework is not None
        
        # Methods should work
        assert isinstance(FrameworkDetector.is_streamlit(), bool)
        assert isinstance(FrameworkDetector.is_flask(), bool)
    
    def test_render_context(self):
        """Test RenderContext creation."""
        from src.adapters.base import RenderContext
        from src.pipeline.get_data import DataFetcher
        from src.pipeline.calculate import StatisticsCalculator
        
        fetcher = DataFetcher()
        calc = StatisticsCalculator()
        
        data = fetcher.get_simple("electronics", n=30, seed=42)
        stats = calc.simple_regression(data.x, data.y)
        
        context = RenderContext(
            analysis_type="simple",
            data=data,
            stats=stats,
            plots_json={},
            dataset_name="test"
        )
        
        assert context.analysis_type == "simple"
        
        # Test serialization
        d = context.to_dict()
        assert "stats" in d
        assert d["stats"]["type"] == "simple"


class TestFlaskAdapter:
    """Test Flask adapter."""
    
    def test_flask_app_creation(self):
        """Test Flask app can be created."""
        try:
            from src.adapters.flask_app import create_flask_app
            
            app = create_flask_app()
            assert app is not None
            
            # Check routes
            rules = [rule.rule for rule in app.url_map.iter_rules()]
            assert "/" in rules
            assert "/simple" in rules
            assert "/multiple" in rules
            assert "/api/analyze" in rules
        except ImportError:
            pytest.skip("Flask not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
