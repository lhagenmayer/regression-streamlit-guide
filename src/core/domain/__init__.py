"""
Domain Layer - Core Business Logic und DDD Patterns.

Diese Package implementiert Domain-Driven Design Prinzipien mit:
- Value Objects: Immutable Datenstrukturen
- Entities: Domain Objects mit Identität
- Aggregates: Konsistenzgrenzen
- Domain Services: Komplexe Geschäftslogik
- Repository Interfaces: Datenzugriffsabstraktion
- Domain Events: Event Sourcing
- Specifications: Geschäftsregeln
- Result Pattern: Fehlerbehandlung
- Unit of Work: Transaktionsmanagement
- Factories: Objekt-Erstellung
"""

# Value Objects
from .value_objects import (
    DatasetConfig,
    RegressionParameters,
    StatisticalSummary,
    ModelMetrics
)

# Entities
from .entities import RegressionModel, Dataset

# Aggregates
from .aggregates import (
    AggregateRoot,
    DatasetAggregate,
    RegressionModelAggregate
)

# Services
from .services import RegressionAnalysisService

# Repositories
from .repositories import DatasetRepository, RegressionModelRepository

# Events
from .events import (
    DomainEvent,
    EventMetadata,
    DatasetCreated,
    DatasetUpdated,
    DatasetDeleted,
    DatasetValidated,
    RegressionModelCreated,
    RegressionModelValidated,
    ModelsCompared,
    ModelPredictionMade,
    EventStore,
    InMemoryEventStore,
    EventDispatcher
)

# Result Pattern
from .result import (
    Result,
    Error,
    DomainError,
    ValidationError,
    NotFoundError,
    BusinessRuleError,
    ValidationResult
)

# Specifications
from .specifications import (
    Specification,
    AndSpecification,
    OrSpecification,
    NotSpecification,
    # Dataset Specifications
    DatasetSpecification,
    HasMinimumSampleSize,
    HasRequiredVariables,
    HasNoMissingValues,
    HasSufficientVariation,
    IsReadyForAnalysis,
    # Model Specifications
    ModelSpecification,
    HasMinimumRSquared,
    IsStatisticallySignificant,
    HasReasonableComplexity,
    IsProductionReady,
    # Factory
    SpecificationFactory
)

# Unit of Work
from .unit_of_work import (
    UnitOfWork,
    InMemoryUnitOfWork,
    unit_of_work_scope,
    UnitOfWorkFactory,
    DefaultUnitOfWorkFactory,
    TransactionScript
)

# Factories
from .factories import (
    DatasetConfigFactory,
    RegressionParametersFactory,
    ModelMetricsFactory,
    DatasetFactory,
    RegressionModelFactory,
    DatasetAggregateFactory,
    RegressionModelAggregateFactory
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

    # Aggregates
    'AggregateRoot',
    'DatasetAggregate',
    'RegressionModelAggregate',

    # Services
    'RegressionAnalysisService',

    # Repositories
    'DatasetRepository',
    'RegressionModelRepository',

    # Events
    'DomainEvent',
    'EventMetadata',
    'DatasetCreated',
    'DatasetUpdated',
    'DatasetDeleted',
    'DatasetValidated',
    'RegressionModelCreated',
    'RegressionModelValidated',
    'ModelsCompared',
    'ModelPredictionMade',
    'EventStore',
    'InMemoryEventStore',
    'EventDispatcher',

    # Result Pattern
    'Result',
    'Error',
    'DomainError',
    'ValidationError',
    'NotFoundError',
    'BusinessRuleError',
    'ValidationResult',

    # Specifications
    'Specification',
    'AndSpecification',
    'OrSpecification',
    'NotSpecification',
    'DatasetSpecification',
    'HasMinimumSampleSize',
    'HasRequiredVariables',
    'HasNoMissingValues',
    'HasSufficientVariation',
    'IsReadyForAnalysis',
    'ModelSpecification',
    'HasMinimumRSquared',
    'IsStatisticallySignificant',
    'HasReasonableComplexity',
    'IsProductionReady',
    'SpecificationFactory',

    # Unit of Work
    'UnitOfWork',
    'InMemoryUnitOfWork',
    'unit_of_work_scope',
    'UnitOfWorkFactory',
    'DefaultUnitOfWorkFactory',
    'TransactionScript',

    # Factories
    'DatasetConfigFactory',
    'RegressionParametersFactory',
    'ModelMetricsFactory',
    'DatasetFactory',
    'RegressionModelFactory',
    'DatasetAggregateFactory',
    'RegressionModelAggregateFactory',
]