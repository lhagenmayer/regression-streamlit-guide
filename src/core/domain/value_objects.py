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


# Result type for error handling (Result Pattern base class)
@dataclass(frozen=True)
class Result:
    """Base class for results."""
    pass


@dataclass(frozen=True)
class Success(Result):
    """Success result wrapper."""
    value: Any


@dataclass(frozen=True)  
class Failure(Result):
    """Failure result wrapper."""
    error: str
    code: str = "UNKNOWN"
    
    
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
class RegressionResult(Result):
    """Container for regression calculation results."""
    parameters: RegressionParameters
    metrics: RegressionMetrics
    predictions: np.ndarray
    residuals: np.ndarray
    model_equation: str


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics for classification model performance."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]
    auc: Optional[float] = None
    
    def __post_init__(self):
        """Validate metrics are in [0, 1] range."""
        for name, value in [
            ("accuracy", self.accuracy),
            ("precision", self.precision), 
            ("recall", self.recall),
            ("f1_score", self.f1_score)
        ]:
            if not (0 <= value <= 1):
                # Small tolerance for floating point errors
                if not (-0.0001 <= value <= 1.0001):
                    raise ValueError(f"{name} must be between 0 and 1, got {value}")

@dataclass(frozen=True)
class ClassificationResult(Result):
    """Container for classification results."""
    classes: List[Any]
    predictions: np.ndarray
    probabilities: np.ndarray
    metrics: ClassificationMetrics      # Training metrics
    model_params: Dict[str, Any]  # e.g., coefficients, k-neighbors
    test_metrics: Optional[ClassificationMetrics] = None # Test metrics
    is_success: bool = True


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


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for data splitting."""
    train_size: float
    stratify: bool
    seed: int
    
    def __post_init__(self):
        if not (0 < self.train_size < 1):
            raise ValueError(f"train_size must be between 0 and 1, got {self.train_size}")


@dataclass(frozen=True)
class SplitStats:
    """Statistics about a data split."""
    train_count: int
    test_count: int
    train_distribution: Dict[Any, int]
    test_distribution: Dict[Any, int]
    
    @property
    def total_count(self) -> int:
        return self.train_count + self.test_count
