"""
Repository Interfaces - Domain-Driven Design repository patterns.

Repositories provide access to aggregates and entities. In DDD, repositories
separate the domain layer from the infrastructure layer by defining interfaces
that infrastructure implements.
"""

from typing import Protocol, Optional, List
from .entities import Dataset, RegressionModel


class DatasetRepository(Protocol):
    """Repository interface for dataset persistence."""

    def save(self, dataset: Dataset) -> None:
        """Save a dataset."""
        ...

    def find_by_id(self, dataset_id: str) -> Optional[Dataset]:
        """Find dataset by ID."""
        ...

    def find_by_name(self, name: str) -> Optional[Dataset]:
        """Find dataset by name."""
        ...

    def list_all(self) -> List[Dataset]:
        """List all datasets."""
        ...

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset."""
        ...


class RegressionModelRepository(Protocol):
    """Repository interface for regression model persistence."""

    def save(self, model: RegressionModel) -> None:
        """Save a regression model."""
        ...

    def find_by_id(self, model_id: str) -> Optional[RegressionModel]:
        """Find model by ID."""
        ...

    def find_by_dataset_id(self, dataset_id: str) -> List[RegressionModel]:
        """Find models by dataset ID."""
        ...

    def list_all(self) -> List[RegressionModel]:
        """List all models."""
        ...

    def delete(self, model_id: str) -> None:
        """Delete a model."""
        ...