"""
Event Handlers - Domain Event Processing in Application Layer.

Event Handlers verarbeiten Domain Events und lösen Nebeneffekte aus
wie Logging, Benachrichtigungen oder externe System-Updates.

Verbesserungen:
- Verwendung des neuen Event Systems
- Bessere Fehlerbehandlung
- Logging für alle Events
"""

from typing import Protocol, Callable, Dict, Any, List
from abc import ABC, abstractmethod

from ..domain.events import (
    DomainEvent,
    DatasetCreated,
    DatasetUpdated,
    DatasetDeleted,
    DatasetValidated,
    RegressionModelCreated,
    RegressionModelValidated,
    ModelsCompared,
    ModelPredictionMade
)

from ...config import get_logger

logger = get_logger(__name__)


class EventHandlerProtocol(Protocol):
    """Protocol for event handlers."""

    def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        ...


class EventBus:
    """
    Simple event bus for publishing and handling domain events.

    This implements the Publisher-Subscriber pattern for domain events.
    """

    def __init__(self):
        self._handlers: Dict[type, List[Callable[[DomainEvent], None]]] = {}

    def subscribe(self, event_type: type, handler: Callable[[DomainEvent], None]) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.__name__}")

    def unsubscribe(self, event_type: type, handler: Callable[[DomainEvent], None]) -> None:
        """Unsubscribe from a specific event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.__name__}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type.__name__}")

    def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribed handlers."""
        event_type = type(event)
        logger.info(f"Publishing event: {event_type.__name__}")

        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error handling event {event_type.__name__}: {e}")
        else:
            logger.debug(f"No handlers registered for {event_type.__name__}")


class DatasetEventHandler:
    """Handles dataset-related domain events."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._register_handlers()

    def _register_handlers(self):
        """Register event handlers."""
        self.event_bus.subscribe(DatasetCreated, self._handle_dataset_created)
        # self.event_bus.subscribe(DatasetUpdated, self._handle_dataset_updated)  # Not yet implemented
        # self.event_bus.subscribe(DatasetDeleted, self._handle_dataset_deleted)  # Not yet implemented

    def _handle_dataset_created(self, event: DatasetCreated) -> None:
        """Handle dataset creation events."""
        logger.info(f"Dataset created: {event.dataset.id} - {event.dataset.config.name}")
        # Could trigger:
        # - Data quality validation
        # - Cache invalidation
        # - Analytics tracking

    # def _handle_dataset_updated(self, event: DatasetUpdated) -> None:
    #     """Handle dataset update events."""
    #     logger.info(f"Dataset updated: {event.dataset.id}")
    #     # Could trigger:
    #     # - Dependent model retraining
    #     # - Cache invalidation

    # def _handle_dataset_deleted(self, event: DatasetDeleted) -> None:
    #     """Handle dataset deletion events."""
    #     logger.info(f"Dataset deleted: {event.dataset_id}")
    #     # Could trigger:
    #     # - Cleanup of dependent models
    #     # - Cache invalidation
    #     # - Audit logging


class RegressionModelEventHandler:
    """Handles regression model-related domain events."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._register_handlers()

    def _register_handlers(self):
        """Register event handlers."""
        self.event_bus.subscribe(RegressionModelCreated, self._handle_model_created)
        self.event_bus.subscribe(RegressionModelValidated, self._handle_model_validated)
        self.event_bus.subscribe(ModelsCompared, self._handle_models_compared)

    def _handle_model_created(self, event: RegressionModelCreated) -> None:
        """Handle model creation events."""
        model = event.model
        logger.info(f"Regression model created: {model.id} for dataset {model.dataset_id}")
        # Could trigger:
        # - Model validation
        # - Performance monitoring
        # - Notification to stakeholders

    def _handle_model_validated(self, event: RegressionModelValidated) -> None:
        """Handle model validation events."""
        logger.info(f"Model validated: {event.model_id} - Quality score: {event.quality_score}")
        # Could trigger:
        # - Quality alerts if score is low
        # - Model deployment decisions
        # - Stakeholder notifications

        if event.quality_score < 0.7:
            logger.warning(f"Low quality model detected: {event.model_id} (score: {event.quality_score})")
            # Could send alerts, trigger retraining, etc.

    def _handle_models_compared(self, event: ModelsCompared) -> None:
        """Handle model comparison events."""
        logger.info(f"Models compared: {event.model1_id} vs {event.model2_id} - Winner: {event.winner}")
        # Could trigger:
        # - Model selection logic
        # - Performance tracking
        # - Automated deployment


# Global event bus instance
_event_bus = EventBus()

# Initialize default handlers
_dataset_handler = DatasetEventHandler(_event_bus)
_model_handler = RegressionModelEventHandler(_event_bus)


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return _event_bus


def publish_event(event: DomainEvent) -> None:
    """Convenience function to publish events."""
    _event_bus.publish(event)