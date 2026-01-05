"""
Comprehensive error handling and edge case tests.

Tests cover:
- Input validation and error handling
- Edge cases in data generation
- Invalid parameters and boundary conditions
- Exception handling in all modules
- Robustness against malformed inputs
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestDataValidationErrors:
    """Test error handling in data generation functions."""

    def test_generate_simple_regression_invalid_dataset(self):
        """Test error handling for invalid dataset choice."""
        from src.data import generate_simple_regression_data

        with pytest.raises(ValueError, match="Unknown dataset"):
            generate_simple_regression_data("invalid_dataset", "Population Density", 10, 42)

    # def test_generate_simple_regression_invalid_variable(self):
    #     """Test error handling for invalid x_variable."""
    #     from src.data import generate_simple_regression_data
    #
    #     with pytest.raises(KeyError):
    #         generate_simple_regression_data("🇨🇭 Schweizer Kantone (sozioökonomisch)", "invalid_variable", 10, 42)

    def test_generate_simple_regression_zero_n(self):
        """Test error handling for zero sample size."""
        from src.data import generate_simple_regression_data

        with pytest.raises(ValueError, match="Sample size n must be a positive integer"):
            generate_simple_regression_data("🇨🇭 Schweizer Kantone (sozioökonomisch)", "Population Density", 0, 42)

    def test_generate_simple_regression_negative_n(self):
        """Test error handling for negative sample size."""
        from src.data import generate_simple_regression_data

        with pytest.raises(ValueError, match="Sample size n must be a positive integer"):
            generate_simple_regression_data("🇨🇭 Schweizer Kantone (sozioökonomisch)", "Population Density", -5, 42)

    # def test_generate_multiple_regression_invalid_dataset(self):
    #     """Test error handling for invalid multiple regression dataset."""
    #     from src.data import generate_multiple_regression_data
    #
    #     with pytest.raises(ValueError, match="Unsupported multiple regression dataset"):
    #         generate_multiple_regression_data("invalid_dataset", 50, 2.0, 42)

    def test_generate_multiple_regression_zero_n(self):
        """Test error handling for zero n in multiple regression."""
        from src.data import generate_multiple_regression_data

        with pytest.raises(ValueError, match="Sample size n_mult must be a positive integer"):
            generate_multiple_regression_data("Cities", 0, 2.0, 42)

    def test_safe_scalar_with_none(self):
        """Test safe_scalar with None input."""
        from src.data import safe_scalar

        with pytest.raises((TypeError, AttributeError)):
            safe_scalar(None)

    def test_safe_scalar_with_empty_array(self):
        """Test safe_scalar with empty array."""
        from src.data import safe_scalar

        with pytest.raises(IndexError):
            safe_scalar(np.array([]))

    def test_safe_scalar_with_nan(self):
        """Test safe_scalar with NaN values."""
        from src.data import safe_scalar

        result = safe_scalar(np.array([np.nan, 2.0, 3.0]))
        assert np.isnan(result) or result == 2.0  # Should handle NaN gracefully


class TestConfigValidation:
    """Test configuration validation and edge cases."""

    def test_missing_config_keys(self):
        """Test behavior with missing configuration keys."""
        from src.config import CITIES_DATASET

        # Ensure all required keys are present
        required_keys = ['n_default', 'n_min', 'n_max', 'noise_std']
        for key in required_keys:
            assert key in CITIES_DATASET, f"Missing required key: {key}"

    def test_config_value_ranges(self):
        """Test that configuration values are within reasonable ranges."""
        from src.config import CITIES_DATASET, SIMPLE_REGRESSION

        # Test that min < max for ranges
        assert CITIES_DATASET['n_min'] < CITIES_DATASET['n_max']
        assert CITIES_DATASET['noise_min'] < CITIES_DATASET['noise_max']
        assert SIMPLE_REGRESSION['n_min'] < SIMPLE_REGRESSION['n_max']


class TestContentValidation:
    """Test content generation validation."""

    def test_get_simple_regression_content_invalid_dataset(self):
        """Test error handling for invalid dataset in content."""
        from src.content import get_simple_regression_content

        with pytest.raises(ValueError):
            get_simple_regression_content("invalid_dataset", "some_variable")

    def test_get_dataset_info_invalid_dataset(self):
        """Test error handling for invalid dataset info."""
        from src.content import get_dataset_info

        with pytest.raises(ValueError):
            get_dataset_info("invalid_dataset")


class TestLoggerValidation:
    """Test logger functionality and error handling."""

    def test_logger_creation(self):
        """Test logger creation with valid names."""
        from src.logger import get_logger

        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_logger_with_empty_name(self):
        """Test logger creation with empty name."""
        from src.logger import get_logger

        logger = get_logger("")
        assert logger is not None

    @patch('logging.getLogger')
    def test_logger_error_handling(self, mock_get_logger):
        """Test logger error handling."""
        from src.logger import get_logger

        # Manually restore to prevent mock from breaking other tests
        import logging as orig_logging
        original_get_logger = orig_logging.getLogger
        
        try:
            mock_get_logger.side_effect = Exception("Logger error")
            # get_logger should still work and return a logger
            logger = get_logger("test")
            assert logger is not None
        finally:
            # Restore original
            orig_logging.getLogger = original_get_logger


class TestAccessibilityValidation:
    """Test accessibility function validation."""

    def test_add_aria_label_valid_inputs(self):
        """Test add_aria_label with valid inputs."""
        from src.accessibility import add_aria_label

        result = add_aria_label("button", "Submit Form")
        assert 'aria-label="Submit Form"' in result


    # def test_add_aria_label_empty_element(self):
    #     \"\"\"Test add_aria_label with empty element type.\"\"\"
    #     from src.accessibility import add_aria_label
    #
    #     with pytest.raises(ValueError):
    #         add_aria_label(\"\", \"Label\")

    def test_add_aria_label_empty_label(self):
        """Test add_aria_label with empty label."""
        from src.accessibility import add_aria_label

        result = add_aria_label("button", "")
        assert 'aria-label=""' in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_dataset_generation(self):
        """Test data generation with large dataset parameter."""
        from src.data import generate_simple_regression_data

        # Even with large n parameter, Swiss canton data is limited
        result = generate_simple_regression_data("🇨🇭 Schweizer Kantone (sozioökonomisch)", "Population Density", 100, 42)
        assert len(result) > 0
        assert len(result['x']) > 0  # Should have some data

    def test_minimum_valid_inputs(self):
        """Test with minimum valid inputs."""
        from src.data import generate_simple_regression_data

        result = generate_simple_regression_data("🇨🇭 Schweizer Kantone (sozioökonomisch)", "Population Density", 1, 42)
        assert len(result) > 0

    def test_extreme_noise_values(self):
        """Test with extreme noise values."""
        from src.data import generate_multiple_regression_data

        # Very low noise
        result_low = generate_multiple_regression_data("🏙️ Städte-Umsatzstudie (75 Städte)", 50, 0.01, 42)
        assert len(result_low['y_mult']) == 50

        # Very high noise
        result_high = generate_multiple_regression_data("🏙️ Städte-Umsatzstudie (75 Städte)", 50, 10.0, 42)
        assert len(result_high['y_mult']) == 50

    def test_special_characters_in_dataset_names(self):
        """Test handling of special characters in dataset names."""
        from src.content import get_dataset_info

        # Test with emoji in name
        info = get_dataset_info("🇨🇭 Schweizer Kantone (sozioökonomisch)")
        assert info['type'] == 'real'


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_floating_point_precision(self):
        """Test floating point precision in calculations."""
        from src.data import safe_scalar

        # Test with very small numbers
        small_num = 1e-15
        result = safe_scalar(small_num)
        assert isinstance(result, float)

        # Test with very large numbers
        large_num = 1e15
        result = safe_scalar(large_num)
        assert isinstance(result, float)

    def test_numpy_array_precision(self):
        """Test numpy array precision handling."""
        from src.data import safe_scalar

        # Test with high precision array
        arr = np.array([1.23456789012345, 2.34567890123456])
        result = safe_scalar(arr)
        assert isinstance(result, float)


class TestMemoryAndPerformance:
    """Test memory usage and performance edge cases."""

    @pytest.mark.slow
    def test_memory_usage_large_dataset(self):
        """Test memory usage with large datasets."""
        from src.data import generate_multiple_regression_data

        # Generate large dataset
        result = generate_multiple_regression_data("🏙️ Städte-Umsatzstudie (75 Städte)", 1000, 2.0, 42)

        # Ensure it completes without memory errors
        assert len(result['y_mult']) == 1000
        assert len(result['x2_preis']) == 1000

    def test_cleanup_after_errors(self):
        """Test that resources are cleaned up after errors."""
        from src.data import generate_simple_regression_data

        # This should not leave any global state corrupted
        try:
            generate_simple_regression_data("invalid", "variable", 10, 42)
        except:
            pass

        # This should still work normally
        result = generate_simple_regression_data("🇨🇭 Schweizer Kantone (sozioökonomisch)", "Population Density", 10, 42)
        assert len(result) > 0