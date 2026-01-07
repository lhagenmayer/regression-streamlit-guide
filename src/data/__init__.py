"""
Data handling package for the Linear Regression Guide.

This package contains data generation, API clients, and content management.

Modules:
- data_generators: Synthetic data generation for regression demos
- api_clients: External API integrations (World Bank, BFS, etc.)
- content: Static content and text templates
- data_loading: Data loading with native OLS calculations
- data_preparation: Data preparation and transformation
"""

# Data generators
from .data_generators import (
    generate_multiple_regression_data,
    generate_simple_regression_data,
    generate_electronics_market_data,
    generate_real_estate_market_data,
    create_dummy_encoded_dataset,
    safe_scalar,
)

# API clients
from .api_clients import (
    fetch_bfs_data,
    fetch_world_bank_data,
    fetch_fred_data,
    fetch_who_health_data,
    fetch_eurostat_data,
)

# Content
from .content import (
    get_simple_regression_content,
    get_multiple_regression_formulas,
    get_multiple_regression_descriptions,
)

# Data loading
from .data_loading import (
    load_multiple_regression_data,
    load_simple_regression_data,
    compute_simple_regression_model,
)

__all__ = [
    # Generators
    'generate_multiple_regression_data',
    'generate_simple_regression_data',
    'generate_electronics_market_data',
    'generate_real_estate_market_data',
    'create_dummy_encoded_dataset',
    'safe_scalar',
    # API clients
    'fetch_bfs_data',
    'fetch_world_bank_data',
    'fetch_fred_data',
    'fetch_who_health_data',
    'fetch_eurostat_data',
    # Content
    'get_simple_regression_content',
    'get_multiple_regression_formulas',
    'get_multiple_regression_descriptions',
    # Data loading
    'load_multiple_regression_data',
    'load_simple_regression_data',
    'compute_simple_regression_model',
]