"""
Data Generators Package.

This package provides various data generators for the Linear Regression Guide.
All generators follow a common interface defined by the DataGenerator Protocol.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol
import time


class DataGeneratorProtocol(Protocol):
    """Protocol interface for all data generators."""

    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate synthetic data based on parameters."""
        ...


class BaseDataGenerator(ABC):
    """
    Abstract base class for data generators.

    Provides common functionality like caching, validation, and logging.
    """

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache = {}

    def _get_cache_key(self, **kwargs) -> tuple:
        """Generate a cache key from the provided arguments."""
        return tuple(sorted(kwargs.items()))

    def _is_cached(self, cache_key: tuple) -> bool:
        """Check if result is cached and still valid."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return True
            del self._cache[cache_key]
        return False

    def _get_cached_result(self, cache_key: tuple) -> Any:
        """Get cached result."""
        return self._cache[cache_key][0]

    def _cache_result(self, cache_key: tuple, result: Any) -> None:
        """Cache a result with current timestamp."""
        self._cache[cache_key] = (result, time.time())

    def _validate_common_params(self, n: int, noise_level: float, seed: int) -> None:
        """Validate common parameters used by most generators."""
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"Sample size n must be a positive integer, got {n}")
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be an integer, got {seed}")
        if not isinstance(noise_level, (int, float)) or noise_level < 0:
            raise ValueError(f"Noise level must be a non-negative number, got {noise_level}")

    @abstractmethod
    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate synthetic data."""
        pass


# ============================================================================
# Re-export all data generation functions from specialized modules
# ============================================================================
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
from .mock_data_generator import safe_scalar

__all__ = [
    # Base classes
    'DataGeneratorProtocol',
    'BaseDataGenerator',
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
    # Utilities
    'safe_scalar',
]