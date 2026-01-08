"""
Data Transfer Objects (DTOs) for Application Layer.
Type-safe data transfer between API/CLI and Use Cases.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from ..domain.value_objects import RegressionType, ModelQuality


@dataclass(frozen=True)
class RegressionRequestDTO:
    """
    Immutable Request DTO for running a regression.
    Uses Enum for type safety.
    """
    dataset_id: str
    n_observations: int
    noise_level: float
    seed: int
    regression_type: RegressionType = RegressionType.SIMPLE
    
    # Optional overrides for synthetic data
    true_intercept: Optional[float] = None
    true_slope: Optional[float] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if self.n_observations < 2:
            raise ValueError(f"n_observations must be >= 2, got {self.n_observations}")
        if self.noise_level < 0:
            raise ValueError(f"noise_level must be non-negative, got {self.noise_level}")


@dataclass(frozen=True)
class RegressionResponseDTO:
    """
    Immutable Response DTO containing results and data.
    
    Note: frozen=True ensures immutability, but lists are still mutable internally.
    For true immutability, consider using tuples for x_data, y_data, etc.
    """
    model_id: str
    success: bool
    
    # Result Data (using tuple for true immutability)
    coefficients: Dict[str, float]
    metrics: Dict[str, float]
    
    # Raw Data (for plotting by frontend)
    x_data: tuple  # Tuple for immutability
    y_data: tuple
    residuals: tuple
    predictions: tuple
    
    # Metadata
    x_label: str
    y_label: str
    title: str
    description: str
    
    # Quality classification
    quality: Optional[ModelQuality] = None
    is_significant: bool = False
    
    # Extensibility
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def r_squared(self) -> Optional[float]:
        """Convenience accessor for R²."""
        return self.metrics.get("r_squared")
    
    @property
    def slope(self) -> Optional[float]:
        """Convenience accessor for slope (simple regression)."""
        if "x" in self.coefficients:
            return self.coefficients["x"]
        return None


@dataclass(frozen=True)
class ErrorDTO:
    """Standardized error response."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


# Type alias for responses that can fail
ResponseResult = Union[RegressionResponseDTO, ErrorDTO]


@dataclass(frozen=True)
class ClassificationRequestDTO:
    """DTO for classification analysis request."""
    dataset_id: str
    n_observations: int
    noise_level: float
    seed: int
    method: str  # "logistic" or "knn"
    k_neighbors: int = 3
    stratify: bool = False
    train_size: float = 0.8
    
    # Optional parameters
    # test_size removed in favor of train_size for consistency


@dataclass(frozen=True)
class ClassificationResponseDTO:
    """Response DTO for classification results."""
    success: bool
    method: str
    classes: tuple
    
    # Metrics (Train & Test)
    metrics: Dict[str, Any]      # Train metrics
    test_metrics: Dict[str, Any] # Test metrics
    
    # Model parameters
    parameters: Dict[str, Any]
    
    # Data for Plotting
    X_data: tuple # Tuple of tuples? or flattened?
    y_data: tuple
    predictions: tuple
    probabilities: tuple
    
    # Metadata
    feature_names: tuple
    target_names: tuple
    dataset_name: str
    dataset_description: str
    
    extra: Dict[str, Any] = field(default_factory=dict)
