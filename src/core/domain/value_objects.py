"""
Value Objects - immutable domain objects that represent concepts.

Value Objects have no identity, are compared by value, and are immutable.
They represent domain concepts like DatasetConfig, RegressionParameters, etc.
"""

from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass, field


class ValueObject(Protocol):
    """Protocol for value objects - immutable domain objects."""
    pass


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a dataset - immutable value object."""
    name: str
    dataset_type: str  # 'synthetic', 'real', 'api'
    source: str
    description: str
    variables: Dict[str, str]
    n_observations: int
    api_available: bool = False
    python_package: Optional[str] = None
    api_docs: Optional[str] = None

    def __post_init__(self):
        """Validate the dataset configuration."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        if self.n_observations <= 0:
            raise ValueError("Number of observations must be positive")


@dataclass(frozen=True)
class RegressionParameters:
    """Parameters for regression analysis - immutable value object."""
    intercept: float
    coefficients: Dict[str, float]
    noise_level: float
    seed: int
    confidence_level: float = 0.95

    def __post_init__(self):
        """Validate regression parameters."""
        if not isinstance(self.coefficients, dict):
            raise ValueError("Coefficients must be a dictionary")
        if self.noise_level < 0:
            raise ValueError("Noise level cannot be negative")
        if not (0 < self.confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")


@dataclass(frozen=True)
class StatisticalSummary:
    """Statistical summary of data - immutable value object."""
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    q25: float
    q75: float

    @classmethod
    def from_list(cls, data: List[float]) -> 'StatisticalSummary':
        """Create summary from list of floats."""
        if not data:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        sorted_data = sorted(data)
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std = variance ** 0.5

        return cls(
            count=n,
            mean=mean,
            std=std,
            min_val=min(data),
            max_val=max(data),
            median=sorted_data[n // 2],
            q25=sorted_data[n // 4],
            q75=sorted_data[3 * n // 4]
        )


@dataclass(frozen=True)
class ModelMetrics:
    """Regression model metrics - immutable value object."""
    r_squared: float
    adj_r_squared: float
    mse: float
    rmse: float
    mae: float
    f_statistic: float
    f_p_value: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if model is statistically significant."""
        return self.f_p_value < alpha

    def get_goodness_of_fit(self) -> str:
        """Get qualitative assessment of model fit."""
        if self.adj_r_squared > 0.8:
            return "excellent"
        elif self.adj_r_squared > 0.6:
            return "good"
        elif self.adj_r_squared > 0.3:
            return "fair"
        else:
            return "poor"