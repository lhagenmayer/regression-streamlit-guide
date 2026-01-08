"""
Domain Value Objects.
Immutable, validated data structures with business logic.
Pure Python - NO external dependencies.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto


class RegressionType(Enum):
    """Type-safe regression type enumeration."""
    SIMPLE = auto()
    MULTIPLE = auto()


class ModelQuality(Enum):
    """Model quality classification based on R²."""
    POOR = auto()      # R² < 0.3
    FAIR = auto()      # 0.3 <= R² < 0.5
    GOOD = auto()      # 0.5 <= R² < 0.7
    EXCELLENT = auto() # R² >= 0.7


@dataclass(frozen=True)
class RegressionParameters:
    """Immutable parameters of a regression model."""
    intercept: float
    coefficients: Dict[str, float]
    
    def __post_init__(self):
        """Validate coefficients are not empty."""
        if not self.coefficients:
            raise ValueError("coefficients cannot be empty")
    
    @property
    def slope(self) -> Optional[float]:
        """Helper for simple regression (single slope)."""
        if len(self.coefficients) == 1:
            return next(iter(self.coefficients.values()))
        return None
    
    @property
    def variable_names(self) -> List[str]:
        """Get ordered list of variable names."""
        return list(self.coefficients.keys())


@dataclass(frozen=True)
class RegressionMetrics:
    """Immutable quality metrics of a regression model."""
    r_squared: float
    r_squared_adj: float
    mse: float
    rmse: float
    f_statistic: Optional[float] = None
    p_value: Optional[float] = None
    
    def __post_init__(self):
        """Validate metric bounds."""
        if not (0 <= self.r_squared <= 1):
            raise ValueError(f"r_squared must be between 0 and 1, got {self.r_squared}")
        if self.mse < 0:
            raise ValueError(f"mse must be non-negative, got {self.mse}")
        if self.rmse < 0:
            raise ValueError(f"rmse must be non-negative, got {self.rmse}")
    
    @property
    def quality(self) -> ModelQuality:
        """Classify model quality based on R²."""
        if self.r_squared < 0.3:
            return ModelQuality.POOR
        elif self.r_squared < 0.5:
            return ModelQuality.FAIR
        elif self.r_squared < 0.7:
            return ModelQuality.GOOD
        else:
            return ModelQuality.EXCELLENT
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if model is statistically significant."""
        if self.p_value is None:
            return False
        return self.p_value < alpha


@dataclass(frozen=True)
class DataPoint:
    """Single data point (observation)."""
    x: Dict[str, float]
    y: float
    
    def __post_init__(self):
        """Validate data point."""
        if not self.x:
            raise ValueError("x cannot be empty")


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata about a dataset."""
    id: str
    name: str
    description: str
    source: str
    variables: tuple  # Use tuple for immutability
    n_observations: int
    is_time_series: bool = False
    
    def __post_init__(self):
        """Validate metadata."""
        if not self.id:
            raise ValueError("id cannot be empty")
        if self.n_observations < 0:
            raise ValueError(f"n_observations must be non-negative, got {self.n_observations}")


# Result type for error handling (Either pattern)
@dataclass(frozen=True)
class Success:
    """Success result wrapper."""
    value: Any


@dataclass(frozen=True)  
class Failure:
    """Failure result wrapper."""
    error: str
    code: str = "UNKNOWN"


# Type alias for Result
Result = Success | Failure
