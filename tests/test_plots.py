"""
Unit tests for plotting functions in plots.py

Tests cover:
- Helper functions (get_signif_stars, get_signif_color, safe_scalar)
- 3D visualization helper functions
- Plotly figure creation functions
- R-output display functions
- Edge cases in plot generation
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import Mock, MagicMock
from src.plots import (
    get_signif_stars,
    get_signif_color,
    create_regression_mesh,
    get_3d_layout_config,
    create_zero_plane,
    create_plotly_scatter,
    create_plotly_scatter_with_line,
    create_plotly_3d_scatter,
    create_plotly_3d_surface,
    create_plotly_residual_plot,
    create_plotly_bar,
    create_plotly_distribution,
    create_r_output_display,
    create_r_output_figure,
)


class TestSignificanceHelpers:
    """Test significance helper functions."""
    
    @pytest.mark.unit
    def test_get_signif_stars_highly_significant(self):
        """Test significance stars for p < 0.001."""
        assert get_signif_stars(0.0001) == '***'
        assert get_signif_stars(0.0009) == '***'
    
    @pytest.mark.unit
    def test_get_signif_stars_very_significant(self):
        """Test significance stars for p < 0.01."""
        assert get_signif_stars(0.001) == '**'
        assert get_signif_stars(0.009) == '**'
    
    @pytest.mark.unit
    def test_get_signif_stars_significant(self):
        """Test significance stars for p < 0.05."""
        assert get_signif_stars(0.01) == '*'
        assert get_signif_stars(0.049) == '*'
    
    @pytest.mark.unit
    def test_get_signif_stars_marginally_significant(self):
        """Test significance stars for p < 0.1."""
        assert get_signif_stars(0.05) == '.'
        assert get_signif_stars(0.09) == '.'
    
    @pytest.mark.unit
    def test_get_signif_stars_not_significant(self):
        """Test significance stars for p >= 0.1."""
        assert get_signif_stars(0.1) == ' '
        assert get_signif_stars(0.5) == ' '
        assert get_signif_stars(1.0) == ' '
    
    @pytest.mark.unit
    def test_get_signif_color_highly_significant(self):
        """Test color for p < 0.001."""
        color = get_signif_color(0.0001)
        assert isinstance(color, str)
        assert color == '#006400'
    
    @pytest.mark.unit
    def test_get_signif_color_not_significant(self):
        """Test color for p >= 0.1."""
        color = get_signif_color(0.5)
        assert isinstance(color, str)
        assert color == '#DC143C'


class TestRegressionMesh:
    """Test regression mesh creation for 3D visualization."""
    
    @pytest.mark.unit
    def test_create_regression_mesh_basic(self):
        """Test basic regression mesh creation."""
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([2, 3, 4, 5, 6])
        params = [1.0, 0.5, 0.3]  # intercept, beta1, beta2
        
        X1_mesh, X2_mesh, Y_mesh = create_regression_mesh(x1, x2, params, n_points=10)
        
        assert X1_mesh.shape == (10, 10)
        assert X2_mesh.shape == (10, 10)
        assert Y_mesh.shape == (10, 10)
        
        # Check that mesh covers the range
        assert X1_mesh.min() >= x1.min()
        assert X1_mesh.max() <= x1.max()
        assert X2_mesh.min() >= x2.min()
        assert X2_mesh.max() <= x2.max()
    
    @pytest.mark.unit
    def test_create_regression_mesh_shape(self):
        """Test mesh with different n_points."""
        x1 = np.linspace(0, 10, 50)
        x2 = np.linspace(0, 5, 50)
        params = [2.0, 1.5, -0.5]
        
        X1_mesh, X2_mesh, Y_mesh = create_regression_mesh(x1, x2, params, n_points=25)
        
        assert X1_mesh.shape == (25, 25)
        assert X2_mesh.shape == (25, 25)
        assert Y_mesh.shape == (25, 25)
    
    @pytest.mark.unit
    def test_create_regression_mesh_calculation(self):
        """Test that Y values are calculated correctly."""
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        params = [1.0, 2.0, 3.0]  # Y = 1 + 2*X1 + 3*X2
        
        X1_mesh, X2_mesh, Y_mesh = create_regression_mesh(x1, x2, params, n_points=5)
        
        # Check a few points manually
        expected_Y = params[0] + params[1]*X1_mesh + params[2]*X2_mesh
        np.testing.assert_array_almost_equal(Y_mesh, expected_Y)


class Test3DLayoutConfig:
    """Test 3D layout configuration."""
    
    @pytest.mark.unit
    def test_get_3d_layout_config_basic(self):
        """Test basic 3D layout configuration."""
        config = get_3d_layout_config("X", "Y", "Z")
        
        assert isinstance(config, dict)
        assert "template" in config
        assert "height" in config
        assert "scene" in config
        assert config["template"] == "plotly_white"
        assert config["height"] == 600
    
    @pytest.mark.unit
    def test_get_3d_layout_config_custom_height(self):
        """Test 3D layout with custom height."""
        config = get_3d_layout_config("X", "Y", "Z", height=800)
        assert config["height"] == 800
    
    @pytest.mark.unit
    def test_get_3d_layout_config_scene_structure(self):
        """Test scene structure in layout."""
        config = get_3d_layout_config("X Axis", "Y Axis", "Z Axis")
        
        scene = config["scene"]
        assert "xaxis_title" in scene
        assert "yaxis_title" in scene
        assert "zaxis_title" in scene
        assert "camera" in scene
        assert scene["xaxis_title"] == "X Axis"


class TestZeroPlane:
    """Test zero plane creation."""
    
    @pytest.mark.unit
    def test_create_zero_plane_basic(self):
        """Test basic zero plane creation."""
        x_range = [0, 10]
        y_range = [-5, 5]
        
        xx, yy, zz = create_zero_plane(x_range, y_range)
        
        assert xx.shape == (2, 2)
        assert yy.shape == (2, 2)
        assert zz.shape == (2, 2)
        assert np.all(zz == 0)


class TestPlotlyScatter:
    """Test plotly scatter plot creation."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_scatter_basic(self):
        """Test basic scatter plot creation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 5, 4, 5])
        
        fig = create_plotly_scatter(x, y)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'scatter'
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_scatter_with_labels(self):
        """Test scatter plot with custom labels."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        
        fig = create_plotly_scatter(
            x, y,
            x_label="Custom X",
            y_label="Custom Y",
            title="Test Plot"
        )
        
        assert fig.layout.xaxis.title.text == "Custom X"
        assert fig.layout.yaxis.title.text == "Custom Y"
        assert fig.layout.title.text == "Test Plot"
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_scatter_empty_arrays(self):
        """Test scatter plot with empty arrays."""
        x = np.array([])
        y = np.array([])
        
        fig = create_plotly_scatter(x, y)
        assert isinstance(fig, go.Figure)


class TestPlotlyScatterWithLine:
    """Test plotly scatter plot with regression line."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_scatter_with_line_basic(self):
        """Test scatter plot with regression line."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 5, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        
        fig = create_plotly_scatter_with_line(x, y, y_pred)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Points and line
        assert fig.data[0].type == 'scatter'  # Points
        assert fig.data[1].type == 'scatter'  # Line
        assert fig.data[1].mode == 'lines'


class TestPlotly3DScatter:
    """Test 3D scatter plot creation."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_3d_scatter_basic(self):
        """Test basic 3D scatter plot."""
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([2, 3, 4, 5, 6])
        y = np.array([3, 5, 7, 9, 11])
        
        fig = create_plotly_3d_scatter(x1, x2, y)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'scatter3d'
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_3d_scatter_with_color(self):
        """Test 3D scatter with color array."""
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        y = np.array([7, 8, 9])
        
        fig = create_plotly_3d_scatter(x1, x2, y, marker_color=y)
        
        assert isinstance(fig, go.Figure)
        # Check that colorscale is set (plotly may expand it to tuple format)
        assert fig.data[0].marker.colorscale is not None


class TestPlotly3DSurface:
    """Test 3D surface plot creation."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_3d_surface_basic(self):
        """Test basic 3D surface plot."""
        x1 = np.linspace(0, 5, 10)
        x2 = np.linspace(0, 3, 10)
        X1_mesh, X2_mesh = np.meshgrid(x1, x2)
        Y_mesh = X1_mesh + 2*X2_mesh
        
        x1_data = np.random.uniform(0, 5, 20)
        x2_data = np.random.uniform(0, 3, 20)
        y_data = x1_data + 2*x2_data
        
        fig = create_plotly_3d_surface(
            X1_mesh, X2_mesh, Y_mesh,
            x1_data, x2_data, y_data
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Surface and points
        assert fig.data[0].type == 'surface'
        assert fig.data[1].type == 'scatter3d'


class TestPlotlyResidualPlot:
    """Test residual plot creation."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_residual_plot_basic(self):
        """Test basic residual plot."""
        y_pred = np.array([1, 2, 3, 4, 5])
        residuals = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
        
        fig = create_plotly_residual_plot(y_pred, residuals)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].type == 'scatter'
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_residual_plot_with_title(self):
        """Test residual plot with custom title."""
        y_pred = np.array([1, 2, 3])
        residuals = np.array([0.1, -0.1, 0.05])
        
        fig = create_plotly_residual_plot(
            y_pred, residuals,
            title="Custom Residual Plot"
        )
        
        assert fig.layout.title.text == "Custom Residual Plot"


class TestPlotlyBar:
    """Test bar chart creation."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_bar_basic(self):
        """Test basic bar chart."""
        categories = ['A', 'B', 'C']
        values = [10, 20, 15]
        
        fig = create_plotly_bar(categories, values)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'bar'
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_bar_with_colors(self):
        """Test bar chart with custom colors."""
        categories = ['SST', 'SSR', 'SSE']
        values = [100, 70, 30]
        colors = ['gray', 'green', 'red']
        
        fig = create_plotly_bar(categories, values, colors=colors)
        
        assert isinstance(fig, go.Figure)
        # Plotly may convert list to tuple, so check values match
        assert list(fig.data[0].marker.color) == colors


class TestPlotlyDistribution:
    """Test distribution plot creation."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_distribution_basic(self):
        """Test basic distribution plot."""
        x_vals = np.linspace(-3, 3, 100)
        y_vals = np.exp(-x_vals**2/2) / np.sqrt(2*np.pi)
        
        fig = create_plotly_distribution(x_vals, y_vals)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].type == 'scatter'
        assert fig.data[0].mode == 'lines'
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_plotly_distribution_with_fill(self):
        """Test distribution plot with filled area."""
        x_vals = np.linspace(-3, 3, 100)
        y_vals = np.exp(-x_vals**2/2) / np.sqrt(2*np.pi)
        
        def fill_condition(x):
            return x > 1.96
        
        fig = create_plotly_distribution(x_vals, y_vals, fill_area=fill_condition)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Line and filled area


class TestROutputDisplay:
    """Test R-style output display functions."""
    
    @pytest.mark.unit
    def test_create_r_output_display_basic(self):
        """Test basic R output display creation."""
        # Create a mock model object
        model = Mock()
        model.resid = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
        model.params = np.array([1.5, 0.5])
        model.bse = np.array([0.1, 0.05])
        model.tvalues = np.array([15.0, 10.0])
        model.pvalues = np.array([0.001, 0.01])
        model.mse_resid = 0.04
        model.df_resid = 10
        model.df_model = 1
        model.rsquared = 0.85
        model.rsquared_adj = 0.83
        model.fvalue = 100.0
        model.f_pvalue = 0.001
        
        output = create_r_output_display(model, "TestVar")
        
        assert isinstance(output, str)
        assert "Residuals:" in output
        assert "Coefficients:" in output
        assert "R-squared:" in output
        assert "F-statistic:" in output
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_create_r_output_figure_basic(self):
        """Test R output figure creation."""
        # Create a mock model object
        model = Mock()
        model.resid = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
        model.params = np.array([1.5, 0.5])
        model.bse = np.array([0.1, 0.05])
        model.tvalues = np.array([15.0, 10.0])
        model.pvalues = np.array([0.001, 0.01])
        model.mse_resid = 0.04
        model.df_resid = 10
        model.df_model = 1
        model.rsquared = 0.85
        model.rsquared_adj = 0.83
        model.fvalue = 100.0
        model.f_pvalue = 0.001
        
        fig = create_r_output_figure(model, "TestVar")
        
        assert isinstance(fig, go.Figure)


class TestEdgeCases:
    """Test edge cases in plotting functions."""
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_empty_data_scatter(self):
        """Test scatter plot with empty data."""
        fig = create_plotly_scatter(np.array([]), np.array([]))
        assert isinstance(fig, go.Figure)
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_single_point_scatter(self):
        """Test scatter plot with single point."""
        fig = create_plotly_scatter(np.array([1]), np.array([2]))
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 1
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_nan_values_in_data(self):
        """Test handling of NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 5, np.nan, 6])
        
        fig = create_plotly_scatter(x, y)
        assert isinstance(fig, go.Figure)
    
    @pytest.mark.unit
    @pytest.mark.visual
    def test_inf_values_in_data(self):
        """Test handling of infinite values."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, np.inf, -np.inf, 6])
        
        fig = create_plotly_scatter(x, y)
        assert isinstance(fig, go.Figure)
    
    @pytest.mark.unit
    def test_zero_p_value(self):
        """Test significance for p-value of exactly 0."""
        stars = get_signif_stars(0.0)
        assert stars == '***'
    
    @pytest.mark.unit
    def test_boundary_p_values(self):
        """Test significance at exact boundary values."""
        assert get_signif_stars(0.001) == '**'  # Boundary
        assert get_signif_stars(0.01) == '*'     # Boundary
        assert get_signif_stars(0.05) == '.'     # Boundary
        assert get_signif_stars(0.1) == ' '      # Boundary


class TestSwissDatasetPlotCompatibility:
    """Test plot functions compatibility with Swiss dataset structures."""

    @pytest.mark.unit
    def test_swiss_canton_3d_plot_compatibility(self):
        """Test that Swiss canton data works with 3D plotting functions."""
        from data import generate_swiss_canton_regression_data

        # Generate Swiss canton data
        canton_data = generate_swiss_canton_regression_data()

        # Test regression mesh creation
        X1_mesh, X2_mesh, Y_mesh = create_regression_mesh(
            canton_data['x_population_density'],
            canton_data['x_foreign_pct'],
            [50000, 1000, -2000, 1000],  # Mock coefficients
            n_points=8
        )

        assert X1_mesh.shape == (8, 8)
        assert X2_mesh.shape == (8, 8)
        assert Y_mesh.shape == (8, 8)

        # Test 3D surface plot creation
        fig = create_plotly_3d_surface(
            X1_mesh, X2_mesh, Y_mesh,
            canton_data['x_population_density'],
            canton_data['x_foreign_pct'],
            canton_data['y_gdp_per_capita'],
            canton_data['x1_name'],
            canton_data['x2_name'],
            canton_data['y_name'],
            'Swiss Cantons: GDP ~ Population Density + Foreign Population'
        )

        assert isinstance(fig, go.Figure)
        # Check that the figure has the expected traces
        assert len(fig.data) >= 2  # Surface + scatter points

    @pytest.mark.unit
    def test_swiss_weather_3d_plot_compatibility(self):
        """Test that Swiss weather data works with 3D plotting functions."""
        from data import generate_swiss_weather_regression_data

        # Generate Swiss weather data
        weather_data = generate_swiss_weather_regression_data()

        # Test regression mesh creation
        X1_mesh, X2_mesh, Y_mesh = create_regression_mesh(
            weather_data['x_altitude'],
            weather_data['x_sunshine'],
            [10, -0.005, 0.1, 5],  # Mock coefficients for temperature model
            n_points=8
        )

        assert X1_mesh.shape == (8, 8)
        assert X2_mesh.shape == (8, 8)
        assert Y_mesh.shape == (8, 8)

        # Test 3D surface plot creation
        fig = create_plotly_3d_surface(
            X1_mesh, X2_mesh, Y_mesh,
            weather_data['x_altitude'],
            weather_data['x_sunshine'],
            weather_data['y_temperature'],
            weather_data['x1_name'],
            weather_data['x2_name'],
            weather_data['y_name'],
            'Swiss Weather: Temperature ~ Altitude + Sunshine + Humidity'
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    @pytest.mark.unit
    def test_swiss_simple_regression_plots(self):
        """Test that Swiss datasets work with simple regression plotting."""
        from data import generate_simple_regression_data

        # Test canton simple regression
        canton_data = generate_simple_regression_data(
            '🇨🇭 Schweizer Kantone (sozioökonomisch)',
            'Population Density',
            5, 42
        )

        # Mock OLS model for testing
        class MockOLS:
            def __init__(self):
                self.params = np.array([50000, 1000])

        mock_model = MockOLS()

        # Test scatter plot with regression line
        fig_scatter = create_plotly_scatter(
            canton_data['x'],
            canton_data['y'],
            canton_data['x_label'],
            canton_data['y_label'],
            mock_model
        )

        assert isinstance(fig_scatter, go.Figure)

        # Test residual plot
        fig_residual = create_plotly_residual_plot(
            mock_model.params[0] + mock_model.params[1] * canton_data['x'],
            np.random.normal(0, 5000, len(canton_data['x'])),
            'Residual Plot: Swiss Cantons'
        )

        assert isinstance(fig_residual, go.Figure)

    @pytest.mark.unit
    def test_swiss_dataset_label_consistency(self):
        """Test that Swiss dataset labels are consistent and meaningful."""
        from data import generate_swiss_canton_regression_data, generate_swiss_weather_regression_data

        # Test canton labels
        canton_data = generate_swiss_canton_regression_data()
        assert 'x1_name' in canton_data
        assert 'x2_name' in canton_data
        assert 'x3_name' in canton_data
        assert 'y_name' in canton_data
        assert 'Population Density' in canton_data['x1_name']
        assert 'GDP' in canton_data['y_name']

        # Test weather labels
        weather_data = generate_swiss_weather_regression_data()
        assert 'x1_name' in weather_data
        assert 'x2_name' in weather_data
        assert 'x3_name' in weather_data
        assert 'y_name' in weather_data
        assert 'Altitude' in weather_data['x1_name']
        assert 'Temperature' in weather_data['y_name']
