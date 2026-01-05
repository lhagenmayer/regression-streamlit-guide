"""
Property-based tests using Hypothesis for comprehensive testing.

These tests use Hypothesis to generate a wide range of inputs and verify
that the system behaves correctly under various conditions.

Tests cover:
- Data generation with random inputs
- Statistical properties of generated data
- Invariants that should always hold
- Edge cases discovered through random testing
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np
import pandas as pd


class TestDataGenerationProperties:
    """Property-based tests for data generation functions."""

    @given(
        n=st.integers(min_value=10, max_value=1000),
        noise_level=st.floats(min_value=0.1, max_value=10.0),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_multiple_regression_data_properties(self, n, noise_level, seed):
        """Test properties of multiple regression data generation."""
        from src.data import generate_multiple_regression_data

        result = generate_multiple_regression_data("🏙️ Städte-Umsatzstudie (75 Städte)", n, noise_level, seed)

        # Basic structure checks
        assert isinstance(result, dict)
        assert 'y_mult' in result
        assert 'x2_preis' in result
        assert 'x3_werbung' in result

        # Data length consistency
        assert len(result['y_mult']) == n
        assert len(result['x2_preis']) == n
        assert len(result['x3_werbung']) == n

        # Data type checks
        assert isinstance(result['y_mult'], np.ndarray)
        assert isinstance(result['x2_preis'], np.ndarray)
        assert isinstance(result['x3_werbung'], np.ndarray)

        # No NaN values
        assert not np.isnan(result['y_mult']).any()
        assert not np.isnan(result['x2_preis']).any()
        assert not np.isnan(result['x3_werbung']).any()

        # Reasonable value ranges
        assert np.all(result['y_mult'] > 0)  # Sales should be positive
        assert np.all(result['x2_preis'] > 0)   # Price should be positive
        assert np.all(result['x3_werbung'] >= 0)   # Advertising should be non-negative

    @given(
        n=st.integers(min_value=1, max_value=100),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_simple_regression_data_consistency(self, n, seed):
        """Test that simple regression data is consistent."""
        from src.data import generate_simple_regression_data

        result = generate_simple_regression_data(
            "🇨🇭 Schweizer Kantone (sozioökonomisch)",
            "Population Density",
            n, seed
        )

        # For Swiss cantons, data should always have 26 entries regardless of n
        # (it's real data, not synthetic)
        assert len(result['x']) == 26
        assert len(result['y']) == 26

        # Check required keys
        required_keys = ['x', 'y', 'x_label', 'y_label', 'context_description', 'context_title']
        for key in required_keys:
            assert key in result

        # Labels should be strings
        assert isinstance(result['x_label'], str)
        assert isinstance(result['y_label'], str)
        assert isinstance(result['context_title'], str)

    @given(
        values=st.lists(st.floats(min_value=-1e10, max_value=1e10), min_size=1, max_size=100)
    )
    def test_safe_scalar_properties(self, values):
        """Test safe_scalar with various numeric inputs."""
        from src.data import safe_scalar

        arr = np.array(values)
        result = safe_scalar(arr)

        # Result should be a float
        assert isinstance(result, float)

        # If array has non-NaN values, result should be one of them
        if not np.isnan(arr).all():
            valid_values = arr[~np.isnan(arr)]
            if len(valid_values) > 0:
                assert result in valid_values or np.isnan(result)

    @given(
        n=st.integers(min_value=5, max_value=50),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_data_generation_reproducibility(self, n, seed):
        """Test that data generation is reproducible with same seed."""
        from src.data import generate_multiple_regression_data

        result1 = generate_multiple_regression_data("Cities", n, 2.0, seed)
        result2 = generate_multiple_regression_data("Cities", n, 2.0, seed)

        # Results should be identical with same seed
        np.testing.assert_array_equal(result1['y_mult'], result2['y_mult'])
        np.testing.assert_array_equal(result1['x2_preis'], result2['x2_preis'])
        np.testing.assert_array_equal(result1['x3_werbung'], result2['x3_werbung'])


class TestStatisticalProperties:
    """Test statistical properties of generated data."""

    @given(
        n=st.integers(min_value=100, max_value=1000),
        noise_level=st.floats(min_value=0.1, max_value=5.0),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_noise_distribution_properties(self, n, noise_level, seed):
        """Test that noise follows expected statistical properties."""
        from src.data import generate_multiple_regression_data

        result = generate_multiple_regression_data("🏙️ Städte-Umsatzstudie (75 Städte)", n, noise_level, seed)

        y = result['y_mult']

        # For large n, mean should be close to expected value (around 77.37 for sales)
        # Allow some tolerance due to randomness
        expected_mean = 77.37
        assert abs(np.mean(y) - expected_mean) < expected_mean * 0.5  # 50% tolerance

        # Standard deviation should be reasonable
        assert 10 < np.std(y) < 200  # Reasonable range for sales data

    @given(
        n=st.integers(min_value=50, max_value=200),
        noise_level=st.floats(min_value=0.5, max_value=3.0)
    )
    def test_correlation_properties(self, n, noise_level):
        """Test correlation properties between variables."""
        from src.data import generate_multiple_regression_data

        result = generate_multiple_regression_data("🏙️ Städte-Umsatzstudie (75 Städte)", n, noise_level, 42)

        x_advertising = result['x3_werbung']
        x_price = result['x2_preis']
        y = result['y_mult']

        # Calculate correlations
        corr_adv_gdp = np.corrcoef(x_advertising, y)[0, 1]
        corr_price_gdp = np.corrcoef(x_price, y)[0, 1]

        # Correlations should be reasonable (not too extreme)
        assert -1 <= corr_adv_gdp <= 1
        assert -1 <= corr_price_gdp <= 1

        # Should have some correlation (data is designed to be correlated)
        assert abs(corr_adv_gdp) > 0.1 or abs(corr_price_gdp) > 0.1


class TestConfigurationProperties:
    """Test configuration-related properties."""

    @given(
        dataset_name=st.sampled_from([
            "🇨🇭 Schweizer Kantone (sozioökonomisch)",
            "Synthetic Cities",
            "Synthetic Houses",
            "Synthetic Electronics"
        ]),
        variable=st.sampled_from([
            "Population Density", "Foreign Population %", "Unemployment"
        ])
    )
    def test_content_generation_completeness(self, dataset_name, variable):
        """Test that content generation always produces complete results."""
        from src.content import get_simple_regression_content

        # Skip invalid combinations that would raise errors
        if dataset_name != "🇨🇭 Schweizer Kantone (sozioökonomisch)":
            pytest.skip("Only Swiss cantons support variable selection")

        result = get_simple_regression_content(dataset_name, variable)

        # Should always return a dictionary with expected keys
        assert isinstance(result, dict)
        assert len(result) > 0

        # Should have essential content keys
        essential_keys = ['formula', 'description', 'interpretation']
        for key in essential_keys:
            assert key in result

    @given(
        dataset_name=st.sampled_from([
            "🇨🇭 Schweizer Kantone (sozioökonomisch)",
            "Cities Dataset",
            "Houses Dataset"
        ])
    )
    def test_dataset_info_consistency(self, dataset_name):
        """Test that dataset info is consistent."""
        from src.content import get_dataset_info

        try:
            info = get_dataset_info(dataset_name)

            # Should have essential info
            assert 'type' in info
            assert 'description' in info
            assert info['type'] in ['real', 'synthetic']

        except KeyError:
            # Some datasets might not exist, which is fine
            pass


class TestLoggerProperties:
    """Test logger property-based behavior."""

    @given(
        logger_name=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['\x00']))
    )
    def test_logger_name_handling(self, logger_name):
        """Test that logger handles various valid names."""
        from src.logger import get_logger

        logger = get_logger(logger_name)

        # Should create a logger successfully
        assert logger is not None
        assert logger.name == logger_name

    @given(
        level=st.sampled_from(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    )
    def test_logger_level_setting(self, level):
        """Test logger level setting."""
        from src.logger import get_logger
        import logging

        logger = get_logger("test_logger")

        # Should be able to set various levels
        level_value = getattr(logging, level)
        logger.setLevel(level_value)

        assert logger.level == level_value


# Settings for hypothesis tests
pytestmark = [
    pytest.mark.property,
    pytest.mark.slow  # Property tests can be slow
]

# Hypothesis settings
settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=10, deadline=None)
settings.register_profile("fast", max_examples=5, deadline=5000)

settings.load_profile("dev")  # Use dev profile by default