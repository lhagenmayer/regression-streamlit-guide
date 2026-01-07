"""
Unit of Work Pattern - Transaktionsmanagement.

Das Unit of Work Pattern koordiniert das Speichern von Änderungen
an mehreren Aggregaten als eine atomare Transaktion.

Vorteile:
- Atomare Operationen über mehrere Aggregate
- Konsistente Event-Veröffentlichung
- Saubere Transaktion-Boundaries
- Testbarkeit durch Abstraktion
"""

from typing import Protocol, List, Optional, Dict, Any, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from contextlib import contextmanager

from .aggregates import AggregateRoot, DatasetAggregate, RegressionModelAggregate
from .events import DomainEvent, EventDispatcher
from .repositories import DatasetRepository, RegressionModelRepository
from .result import Result, Error


T = TypeVar('T', bound=AggregateRoot)


class UnitOfWork(Protocol):
    """
    Protocol für Unit of Work.

    Definiert den Vertrag für transaktionales Arbeiten mit Aggregaten.
    """

    def begin(self) -> None:
        """Begin a new unit of work."""
        ...

    def commit(self) -> Result[None]:
        """Commit all changes."""
        ...

    def rollback(self) -> None:
        """Rollback all changes."""
        ...

    def register_new(self, aggregate: AggregateRoot) -> None:
        """Register a new aggregate for insertion."""
        ...

    def register_dirty(self, aggregate: AggregateRoot) -> None:
        """Register a modified aggregate for update."""
        ...

    def register_deleted(self, aggregate: AggregateRoot) -> None:
        """Register an aggregate for deletion."""
        ...


@dataclass
class InMemoryUnitOfWork:
    """
    In-Memory Implementation des Unit of Work Patterns.

    Koordiniert Änderungen an Datasets und Models als eine Transaktion.
    Verwendet Listen statt Sets, da Aggregates mutable und nicht hashable sind.
    """
    dataset_repository: DatasetRepository
    model_repository: RegressionModelRepository
    event_dispatcher: Optional[EventDispatcher] = None

    _new_aggregates: List[AggregateRoot] = field(default_factory=list, repr=False)
    _dirty_aggregates: List[AggregateRoot] = field(default_factory=list, repr=False)
    _deleted_aggregates: List[AggregateRoot] = field(default_factory=list, repr=False)
    _collected_events: List[DomainEvent] = field(default_factory=list, repr=False)
    _is_active: bool = field(default=False, repr=False)

    def _aggregate_in_list(self, aggregate: AggregateRoot, lst: List[AggregateRoot]) -> bool:
        """Check if aggregate is in list by ID."""
        return any(a.id == aggregate.id for a in lst)

    def _remove_from_list(self, aggregate: AggregateRoot, lst: List[AggregateRoot]) -> None:
        """Remove aggregate from list by ID."""
        for i, a in enumerate(lst):
            if a.id == aggregate.id:
                lst.pop(i)
                break

    def begin(self) -> None:
        """Begin a new unit of work."""
        if self._is_active:
            raise RuntimeError("Unit of Work is already active")

        self._new_aggregates.clear()
        self._dirty_aggregates.clear()
        self._deleted_aggregates.clear()
        self._collected_events.clear()
        self._is_active = True

    def commit(self) -> Result[None]:
        """
        Commit all registered changes.

        Steps:
        1. Validate all aggregates
        2. Persist new aggregates
        3. Persist dirty aggregates
        4. Delete removed aggregates
        5. Collect and dispatch events
        """
        if not self._is_active:
            return Result.failure(Error("NOT_ACTIVE", "Unit of Work is not active"))

        try:
            # Validate all aggregates
            all_aggregates = self._new_aggregates + self._dirty_aggregates
            for aggregate in all_aggregates:
                validation = aggregate.validate()
                if validation.is_invalid:
                    return Result.failures(validation.errors)

            # Persist new aggregates
            for aggregate in self._new_aggregates:
                self._persist_aggregate(aggregate)
                self._collected_events.extend(aggregate.clear_events())
                aggregate.increment_version()

            # Persist dirty aggregates
            for aggregate in self._dirty_aggregates:
                self._persist_aggregate(aggregate)
                self._collected_events.extend(aggregate.clear_events())
                aggregate.increment_version()

            # Delete aggregates
            for aggregate in self._deleted_aggregates:
                self._delete_aggregate(aggregate)
                self._collected_events.extend(aggregate.clear_events())

            # Dispatch events
            if self.event_dispatcher:
                self.event_dispatcher.dispatch_all(self._collected_events)

            # Reset state
            self._reset()

            return Result.success(None)

        except Exception as e:
            self.rollback()
            return Result.failure(Error("COMMIT_ERROR", str(e)))

    def rollback(self) -> None:
        """Rollback all changes."""
        self._reset()

    def _reset(self) -> None:
        """Reset unit of work state."""
        self._new_aggregates.clear()
        self._dirty_aggregates.clear()
        self._deleted_aggregates.clear()
        self._collected_events.clear()
        self._is_active = False

    def register_new(self, aggregate: AggregateRoot) -> None:
        """Register a new aggregate for insertion."""
        if not self._is_active:
            raise RuntimeError("Unit of Work is not active")
        if not self._aggregate_in_list(aggregate, self._new_aggregates):
            self._new_aggregates.append(aggregate)

    def register_dirty(self, aggregate: AggregateRoot) -> None:
        """Register a modified aggregate for update."""
        if not self._is_active:
            raise RuntimeError("Unit of Work is not active")
        if not self._aggregate_in_list(aggregate, self._new_aggregates):
            if not self._aggregate_in_list(aggregate, self._dirty_aggregates):
                self._dirty_aggregates.append(aggregate)

    def register_deleted(self, aggregate: AggregateRoot) -> None:
        """Register an aggregate for deletion."""
        if not self._is_active:
            raise RuntimeError("Unit of Work is not active")
        if not self._aggregate_in_list(aggregate, self._deleted_aggregates):
            self._deleted_aggregates.append(aggregate)
        # Remove from other lists if present
        self._remove_from_list(aggregate, self._new_aggregates)
        self._remove_from_list(aggregate, self._dirty_aggregates)

    def _persist_aggregate(self, aggregate: AggregateRoot) -> None:
        """Persist an aggregate to the appropriate repository."""
        if isinstance(aggregate, DatasetAggregate):
            self.dataset_repository.save(aggregate)
        elif isinstance(aggregate, RegressionModelAggregate):
            self.model_repository.save(aggregate)
        else:
            raise ValueError(f"Unknown aggregate type: {type(aggregate)}")

    def _delete_aggregate(self, aggregate: AggregateRoot) -> None:
        """Delete an aggregate from the appropriate repository."""
        if isinstance(aggregate, DatasetAggregate):
            self.dataset_repository.delete(aggregate.id)
        elif isinstance(aggregate, RegressionModelAggregate):
            self.model_repository.delete(aggregate.id)
        else:
            raise ValueError(f"Unknown aggregate type: {type(aggregate)}")

    @property
    def is_active(self) -> bool:
        """Check if unit of work is active."""
        return self._is_active

    @property
    def pending_events(self) -> List[DomainEvent]:
        """Get pending events (before commit)."""
        events = []
        for aggregate in self._new_aggregates | self._dirty_aggregates:
            events.extend(aggregate.uncommitted_events)
        return events


@contextmanager
def unit_of_work_scope(uow: InMemoryUnitOfWork):
    """
    Context manager für Unit of Work.

    Usage:
        with unit_of_work_scope(uow) as uow:
            uow.register_new(dataset)
            uow.register_new(model)
            # commit happens automatically
    """
    uow.begin()
    try:
        yield uow
        result = uow.commit()
        if result.is_failure:
            raise RuntimeError(f"Commit failed: {result.error}")
    except Exception:
        uow.rollback()
        raise


class UnitOfWorkFactory(Protocol):
    """Factory Protocol für Unit of Work Erstellung."""

    def create(self) -> InMemoryUnitOfWork:
        """Create a new Unit of Work instance."""
        ...


@dataclass
class DefaultUnitOfWorkFactory:
    """Default Factory für Unit of Work."""
    dataset_repository: DatasetRepository
    model_repository: RegressionModelRepository
    event_dispatcher: Optional[EventDispatcher] = None

    def create(self) -> InMemoryUnitOfWork:
        """Create a new Unit of Work instance."""
        return InMemoryUnitOfWork(
            dataset_repository=self.dataset_repository,
            model_repository=self.model_repository,
            event_dispatcher=self.event_dispatcher
        )


# ============================================================================
# Transaction Script Pattern (Alternative für einfache Operationen)
# ============================================================================

class TransactionScript:
    """
    Transaction Script für einfache Operationen.

    Für Operationen, die nur ein Aggregate betreffen,
    ist ein vollständiges Unit of Work oft Overkill.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        model_repository: RegressionModelRepository,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        self.dataset_repository = dataset_repository
        self.model_repository = model_repository
        self.event_dispatcher = event_dispatcher

    def save_dataset(self, dataset: DatasetAggregate) -> Result[None]:
        """Save a single dataset."""
        validation = dataset.validate()
        if validation.is_invalid:
            return validation.to_result(None)

        try:
            self.dataset_repository.save(dataset)

            if self.event_dispatcher:
                self.event_dispatcher.dispatch_all(dataset.clear_events())

            return Result.success(None)
        except Exception as e:
            return Result.failure(Error("SAVE_ERROR", str(e)))

    def save_model(self, model: RegressionModelAggregate) -> Result[None]:
        """Save a single model."""
        validation = model.validate()
        if validation.is_invalid:
            return validation.to_result(None)

        try:
            self.model_repository.save(model)

            if self.event_dispatcher:
                self.event_dispatcher.dispatch_all(model.clear_events())

            return Result.success(None)
        except Exception as e:
            return Result.failure(Error("SAVE_ERROR", str(e)))

    def delete_dataset(self, dataset_id: str) -> Result[None]:
        """Delete a dataset."""
        try:
            self.dataset_repository.delete(dataset_id)
            return Result.success(None)
        except Exception as e:
            return Result.failure(Error("DELETE_ERROR", str(e)))

    def delete_model(self, model_id: str) -> Result[None]:
        """Delete a model."""
        try:
            self.model_repository.delete(model_id)
            return Result.success(None)
        except Exception as e:
            return Result.failure(Error("DELETE_ERROR", str(e)))
