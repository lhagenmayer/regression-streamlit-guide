"""
Regression data generators for the Linear Regression Guide.

This module provides a unified interface to all data generation functions,
organized into specialized sub-modules for better maintainability.
"""

# Re-export all data generation functions from specialized modules
from .multiple_regression_generator import (
    generate_multiple_regression_data,
    generate_custom_multiple_regression_data
)

from .simple_regression_generator import (
    generate_simple_regression_data,
    generate_custom_simple_regression_data
)

from .dummy_encoding_generator import (
    create_dummy_encoded_dataset,
    generate_categorical_regression_data
)

from .market_data_generator import (
    generate_electronics_market_data,
    generate_real_estate_market_data,
    generate_stock_market_data
)

# Keep backward compatibility by importing all functions
__all__ = [
    # Multiple regression
    'generate_multiple_regression_data',
    'generate_custom_multiple_regression_data',

    # Simple regression
    'generate_simple_regression_data',
    'generate_custom_simple_regression_data',

    # Dummy encoding
    'create_dummy_encoded_dataset',
    'generate_categorical_regression_data',

    # Market data
    'generate_electronics_market_data',
    'generate_real_estate_market_data',
    'generate_stock_market_data',
]