"""
Core Package - Clean Architecture Implementation.

Dieses Package implementiert eine saubere Architektur basierend auf
Domain-Driven Design (DDD), CQRS und Clean Architecture Prinzipien.

Struktur:
- domain/: Reine Geschäftslogik ohne externe Abhängigkeiten
  - Value Objects, Entities, Aggregates
  - Domain Services, Repository Interfaces
  - Domain Events, Specifications
  - Result Pattern, Unit of Work, Factories

- application/: Anwendungslogik und Use Cases
  - Use Cases orchestrieren Domain Objects
  - CQRS Commands und Queries
  - Handler verarbeiten Commands/Queries
  - Application Services

Verwendung:
    from src.core.domain import (
        Result, DatasetFactory, RegressionModelFactory,
        SpecificationFactory, unit_of_work_scope
    )
    from src.core.application import (
        CreateRegressionModelCommand,
        Mediator, create_mediator
    )
"""

# Domain Layer - Kern-Exports (Lazy imports vermeiden zirkuläre Abhängigkeiten)
from .domain.services import RegressionAnalysisService
from .domain.result import Result, Error, ValidationResult
from .domain.aggregates import DatasetAggregate, RegressionModelAggregate
from .domain.factories import (
    DatasetFactory,
    RegressionModelFactory,
    DatasetAggregateFactory,
    RegressionModelAggregateFactory
)
from .domain.specifications import SpecificationFactory
from .domain.events import EventDispatcher, InMemoryEventStore

# Application Layer Commands (ohne Handler um zirkuläre Imports zu vermeiden)
from .application.commands import CreateRegressionModelCommand, CreateDatasetCommand


def get_mediator():
    """
    Factory function für Mediator (Lazy Loading).

    Vermeidet zirkuläre Imports durch späten Import.
    """
    from .application.handlers import Mediator, create_mediator
    return Mediator, create_mediator


__all__ = [
    # Domain Services
    'RegressionAnalysisService',

    # Result Pattern
    'Result',
    'Error',
    'ValidationResult',

    # Aggregates
    'DatasetAggregate',
    'RegressionModelAggregate',

    # Factories
    'DatasetFactory',
    'RegressionModelFactory',
    'DatasetAggregateFactory',
    'RegressionModelAggregateFactory',

    # Specifications
    'SpecificationFactory',

    # Events
    'EventDispatcher',
    'InMemoryEventStore',

    # CQRS Commands
    'CreateRegressionModelCommand',
    'CreateDatasetCommand',

    # Factory function
    'get_mediator',
]