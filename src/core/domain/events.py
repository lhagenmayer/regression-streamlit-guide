"""
Domain Events - Domain-Driven Design domain events.

Domain events represent important business events that have occurred
in the domain. They are used to communicate state changes and trigger
side effects in a decoupled way.
"""

from dataclasses import dataclass
from typing import List, Protocol
from .entities import Dataset, RegressionModel
from .value_objects import ModelMetrics


class DomainEventProtocol(Protocol):
    """Protocol for domain events."""
    pass


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events."""
    pass


@dataclass(frozen=True)
class DatasetCreated(DomainEvent):
    """Event fired when a dataset is created."""
    dataset: Dataset


# Future enhancement: Dataset update/delete operations
# Uncomment when implementing dataset modification features

# @dataclass(frozen=True)
# class DatasetUpdated(DomainEvent):
#     """Event fired when a dataset is updated."""
#     dataset: Dataset
#
#
# @dataclass(frozen=True)
# class DatasetDeleted(DomainEvent):
#     """Event fired when a dataset is deleted."""
#     dataset_id: str


@dataclass(frozen=True)
class RegressionModelCreated(DomainEvent):
    """Event fired when a regression model is created."""
    model: RegressionModel


@dataclass(frozen=True)
class RegressionModelValidated(DomainEvent):
    """Event fired when a regression model is validated."""
    model_id: str
    quality_score: float
    recommendations: List[str]


@dataclass(frozen=True)
class ModelsCompared(DomainEvent):
    """Event fired when two models are compared."""
    model1_id: str
    model2_id: str
    winner: str
    reason: str