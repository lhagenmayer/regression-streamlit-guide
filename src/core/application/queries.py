"""
Queries - CQRS read operations.

Queries represent read operations that retrieve data from the system.
They are immutable data structures that express data retrieval intent.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class GetDatasetByIdQuery:
    """Query to get a dataset by ID."""
    dataset_id: str


@dataclass(frozen=True)
class ListDatasetsQuery:
    """Query to list all datasets."""
    limit: Optional[int] = None
    offset: Optional[int] = 0


@dataclass(frozen=True)
class GetModelByIdQuery:
    """Query to get a model by ID."""
    model_id: str


@dataclass(frozen=True)
class ListModelsQuery:
    """Query to list all models."""
    dataset_id: Optional[str] = None  # Filter by dataset
    limit: Optional[int] = None
    offset: Optional[int] = 0


@dataclass(frozen=True)
class GetDatasetStatisticsQuery:
    """Query to get dataset statistics."""
    dataset_id: str


@dataclass(frozen=True)
class GetModelDiagnosticsQuery:
    """Query to get model diagnostics."""
    model_id: str


@dataclass(frozen=True)
class CompareModelsQuery:
    """Query to compare two models."""
    model1_id: str
    model2_id: str


@dataclass(frozen=True)
class GetAvailableDataSourcesQuery:
    """Query to get available data sources."""
    pass