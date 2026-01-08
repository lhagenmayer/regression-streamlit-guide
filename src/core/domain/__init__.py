"""
Domain Package - Pure Python, No External Dependencies.
Core business logic, entities, value objects, and interfaces.
"""
from .value_objects import (
    RegressionType,
    ModelQuality,
    RegressionParameters,
    RegressionMetrics,
    DataPoint,
    DatasetMetadata,
    Success,
    Failure,
    Result,
)

from .entities import (
    RegressionModel,
)

from .interfaces import (
    # Granular interfaces (SRP)
    IDatasetFetcher,
    IDatasetLister,
    ISimpleRegressionTrainer,
    IMultipleRegressionTrainer,
    IModelRepository,
    IPredictor,
    # Combined interfaces (backward compatible)
    IDataProvider,
    IRegressionService,
)

__all__ = [
    # Enums
    "RegressionType",
    "ModelQuality",
    # Value Objects
    "RegressionParameters",
    "RegressionMetrics",
    "DataPoint",
    "DatasetMetadata",
    # Result Types
    "Success",
    "Failure",
    "Result",
    # Entities
    "RegressionModel",
    # Interfaces
    "IDatasetFetcher",
    "IDatasetLister",
    "ISimpleRegressionTrainer",
    "IMultipleRegressionTrainer",
    "IModelRepository",
    "IPredictor",
    "IDataProvider",
    "IRegressionService",
]
