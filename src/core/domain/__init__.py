"""
Domain layer - contains business rules and domain logic.

This package implements Domain-Driven Design principles with:
- Value Objects for immutable data structures
- Entities for domain objects with identity
- Domain Services for business logic
- Repository interfaces for data access
"""

from .value_objects import (
    DatasetConfig,
    RegressionParameters,
    StatisticalSummary,
    ModelMetrics
)
from .entities import RegressionModel, Dataset
from .services import RegressionAnalysisService
from .repositories import DatasetRepository, RegressionModelRepository
from .events import (
    DomainEvent,
    DatasetCreated,
    DatasetUpdated,
    DatasetDeleted,
    RegressionModelCreated,
    RegressionModelValidated,
    ModelsCompared
)

__all__ = [
    # Value Objects
    'DatasetConfig',
    'RegressionParameters',
    'StatisticalSummary',
    'ModelMetrics',
    # Entities
    'RegressionModel',
    'Dataset',
    # Services
    'RegressionAnalysisService',
    # Repositories
    'DatasetRepository',
    'RegressionModelRepository',
    # Events
    'DomainEvent',
    'DatasetCreated',
    'DatasetUpdated',
    'DatasetDeleted',
    'RegressionModelCreated',
    'RegressionModelValidated',
    'ModelsCompared',
]