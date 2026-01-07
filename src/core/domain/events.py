"""
Domain Events - Domain-Driven Design domain events.

Domain events represent important business events that have occurred
in the domain. They are used to communicate state changes and trigger
side effects in a decoupled way.

Verbesserungen:
- Event Metadata (timestamp, correlation_id, causation_id)
- Event versioning für Schema Evolution
- Immutable Events mit frozen dataclasses
- Event Store Interface
"""

from dataclasses import dataclass, field
from typing import List, Protocol, Optional, Dict, Any, Callable, Type
from abc import ABC, abstractmethod
import uuid


def generate_event_id() -> str:
    """Generate unique event ID."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class EventMetadata:
    """Metadata für Domain Events."""
    event_id: str
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    version: int = 1
    aggregate_id: Optional[str] = None
    aggregate_type: Optional[str] = None

    @classmethod
    def create(
        cls,
        aggregate_id: Optional[str] = None,
        aggregate_type: Optional[str] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None
    ) -> 'EventMetadata':
        """Factory method für Event Metadata."""
        return cls(
            event_id=generate_event_id(),
            correlation_id=correlation_id or generate_event_id(),
            causation_id=causation_id,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type
        )


class DomainEventProtocol(Protocol):
    """Protocol for domain events."""
    metadata: EventMetadata


class DomainEvent(ABC):
    """
    Base class for all domain events.

    Alle Domain Events sind immutable und enthalten Metadata
    für Event Sourcing und Tracing.

    Hinweis: Wir verwenden keine frozen dataclass als Basisklasse,
    um Vererbungsprobleme mit Default-Werten zu vermeiden.
    """

    @property
    @abstractmethod
    def metadata(self) -> EventMetadata:
        """Get event metadata."""
        pass

    @property
    def event_id(self) -> str:
        """Get event ID."""
        return self.metadata.event_id

    @property
    def event_type(self) -> str:
        """Get event type name."""
        return self.__class__.__name__

    @property
    def aggregate_id(self) -> Optional[str]:
        """Get aggregate ID this event belongs to."""
        return self.metadata.aggregate_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "metadata": {
                "event_id": self.metadata.event_id,
                "correlation_id": self.metadata.correlation_id,
                "causation_id": self.metadata.causation_id,
                "version": self.metadata.version,
                "aggregate_id": self.metadata.aggregate_id,
                "aggregate_type": self.metadata.aggregate_type,
            }
        }


# ============================================================================
# Dataset Events
# ============================================================================

@dataclass(frozen=True)
class DatasetCreated(DomainEvent):
    """Event fired when a dataset is created."""
    dataset_id: str
    dataset_name: str
    n_variables: int
    n_observations: int
    source: str
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def from_dataset(cls, dataset: 'Dataset') -> 'DatasetCreated':
        """Factory method to create event from Dataset entity."""
        return cls(
            dataset_id=dataset.id,
            dataset_name=dataset.config.name,
            n_variables=len(dataset.data),
            n_observations=dataset.get_sample_size(),
            source=dataset.config.source,
            _metadata=EventMetadata.create(
                aggregate_id=dataset.id,
                aggregate_type="Dataset"
            )
        )


@dataclass(frozen=True)
class DatasetUpdated(DomainEvent):
    """Event fired when a dataset is updated."""
    dataset_id: str
    changes: Dict[str, Any]
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def create(cls, dataset_id: str, changes: Dict[str, Any]) -> 'DatasetUpdated':
        """Factory method."""
        return cls(
            dataset_id=dataset_id,
            changes=changes,
            _metadata=EventMetadata.create(
                aggregate_id=dataset_id,
                aggregate_type="Dataset"
            )
        )


@dataclass(frozen=True)
class DatasetDeleted(DomainEvent):
    """Event fired when a dataset is deleted."""
    dataset_id: str
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def create(cls, dataset_id: str) -> 'DatasetDeleted':
        """Factory method."""
        return cls(
            dataset_id=dataset_id,
            _metadata=EventMetadata.create(
                aggregate_id=dataset_id,
                aggregate_type="Dataset"
            )
        )


@dataclass(frozen=True)
class DatasetValidated(DomainEvent):
    """Event fired when dataset validation is completed."""
    dataset_id: str
    is_valid: bool
    issues: tuple  # Using tuple instead of List for immutability
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def create(cls, dataset_id: str, is_valid: bool, issues: List[str]) -> 'DatasetValidated':
        """Factory method."""
        return cls(
            dataset_id=dataset_id,
            is_valid=is_valid,
            issues=tuple(issues) if issues else (),
            _metadata=EventMetadata.create(
                aggregate_id=dataset_id,
                aggregate_type="Dataset"
            )
        )


# ============================================================================
# Model Events
# ============================================================================

@dataclass(frozen=True)
class RegressionModelCreated(DomainEvent):
    """Event fired when a regression model is created."""
    model_id: str
    dataset_id: str
    model_type: str
    feature_names: tuple
    r_squared: float
    adj_r_squared: float
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def from_model(cls, model: 'RegressionModel') -> 'RegressionModelCreated':
        """Factory method to create event from RegressionModel entity."""
        return cls(
            model_id=model.id,
            dataset_id=model.dataset_id,
            model_type=model.model_type,
            feature_names=tuple(model.feature_names),
            r_squared=model.metrics.r_squared,
            adj_r_squared=model.metrics.adj_r_squared,
            _metadata=EventMetadata.create(
                aggregate_id=model.id,
                aggregate_type="RegressionModel"
            )
        )


@dataclass(frozen=True)
class RegressionModelValidated(DomainEvent):
    """Event fired when a regression model is validated."""
    model_id: str
    quality_score: float
    is_production_ready: bool
    recommendations: tuple
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def create(
        cls,
        model_id: str,
        quality_score: float,
        is_production_ready: bool,
        recommendations: List[str]
    ) -> 'RegressionModelValidated':
        """Factory method."""
        return cls(
            model_id=model_id,
            quality_score=quality_score,
            is_production_ready=is_production_ready,
            recommendations=tuple(recommendations),
            _metadata=EventMetadata.create(
                aggregate_id=model_id,
                aggregate_type="RegressionModel"
            )
        )


@dataclass(frozen=True)
class ModelsCompared(DomainEvent):
    """Event fired when two models are compared."""
    model1_id: str
    model2_id: str
    winner: str
    reason: str
    r2_difference: float
    complexity_difference: int
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def create(
        cls,
        model1_id: str,
        model2_id: str,
        winner: str,
        reason: str,
        r2_difference: float,
        complexity_difference: int
    ) -> 'ModelsCompared':
        """Factory method."""
        return cls(
            model1_id=model1_id,
            model2_id=model2_id,
            winner=winner,
            reason=reason,
            r2_difference=r2_difference,
            complexity_difference=complexity_difference,
            _metadata=EventMetadata.create()
        )


@dataclass(frozen=True)
class ModelPredictionMade(DomainEvent):
    """Event fired when a model makes predictions."""
    model_id: str
    n_predictions: int
    input_variables: tuple
    _metadata: EventMetadata = field(default_factory=EventMetadata.create)

    @property
    def metadata(self) -> EventMetadata:
        return self._metadata

    @classmethod
    def create(
        cls,
        model_id: str,
        n_predictions: int,
        input_variables: List[str]
    ) -> 'ModelPredictionMade':
        """Factory method."""
        return cls(
            model_id=model_id,
            n_predictions=n_predictions,
            input_variables=tuple(input_variables),
            _metadata=EventMetadata.create(
                aggregate_id=model_id,
                aggregate_type="RegressionModel"
            )
        )


# ============================================================================
# Event Store Interface
# ============================================================================

class EventStore(Protocol):
    """
    Protocol for Event Store implementations.

    Ein Event Store speichert Domain Events persistent und ermöglicht
    Event Sourcing und Event Replay.
    """

    def append(self, event: DomainEvent) -> None:
        """Append an event to the store."""
        ...

    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[DomainEvent]:
        """Get events with optional filtering."""
        ...

    def get_events_by_correlation(self, correlation_id: str) -> List[DomainEvent]:
        """Get all events with same correlation ID."""
        ...


class InMemoryEventStore:
    """
    In-memory implementation of EventStore for development/testing.

    Production implementations would use databases like EventStoreDB,
    PostgreSQL with event sourcing tables, etc.
    """

    def __init__(self):
        self._events: List[DomainEvent] = []

    def append(self, event: DomainEvent) -> None:
        """Append an event to the store."""
        self._events.append(event)

    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[DomainEvent]:
        """Get events with optional filtering."""
        events = self._events

        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[:limit]

    def get_events_by_correlation(self, correlation_id: str) -> List[DomainEvent]:
        """Get all events with same correlation ID."""
        return [
            e for e in self._events
            if e.metadata.correlation_id == correlation_id
        ]

    def get_all_events(self) -> List[DomainEvent]:
        """Get all stored events."""
        return self._events.copy()

    def clear(self) -> None:
        """Clear all events (useful for testing)."""
        self._events.clear()

    def count(self) -> int:
        """Get total number of stored events."""
        return len(self._events)


# ============================================================================
# Event Dispatcher
# ============================================================================

EventHandler = Callable[[DomainEvent], None]


class EventDispatcher:
    """
    Dispatcher für Domain Events.

    Verteilt Events an registrierte Handler.
    Unterstützt sowohl synchrone als auch optionale persistente Speicherung.
    """

    def __init__(self, event_store: Optional[EventStore] = None):
        self._handlers: Dict[Type[DomainEvent], List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._event_store = event_store

    def register(
        self,
        event_type: Type[DomainEvent],
        handler: EventHandler
    ) -> None:
        """Register a handler for specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def register_global(self, handler: EventHandler) -> None:
        """Register a handler that receives all events."""
        self._global_handlers.append(handler)

    def unregister(
        self,
        event_type: Type[DomainEvent],
        handler: EventHandler
    ) -> None:
        """Unregister a handler."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def dispatch(self, event: DomainEvent) -> None:
        """
        Dispatch an event to all registered handlers.

        If event_store is configured, the event is also persisted.
        """
        # Persist event if store is available
        if self._event_store:
            self._event_store.append(event)

        # Dispatch to type-specific handlers
        event_type = type(event)
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event)
                except Exception:
                    # Log error but don't stop dispatch
                    pass

        # Dispatch to global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                pass

    def dispatch_all(self, events: List[DomainEvent]) -> None:
        """Dispatch multiple events in order."""
        for event in events:
            self.dispatch(event)
