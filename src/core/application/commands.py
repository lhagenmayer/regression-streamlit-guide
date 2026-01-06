"""
Commands - CQRS write operations.

Commands represent write operations that change the state of the system.
They are immutable data structures that express user intent.
"""

from dataclasses import dataclass
from typing import List, Optional

from ...domain.value_objects import DatasetConfig, RegressionParameters


@dataclass(frozen=True)
class CreateDatasetCommand:
    """Command to create a new dataset."""
    config: DatasetConfig


@dataclass(frozen=True)
class UpdateDatasetCommand:
    """Command to update an existing dataset."""
    dataset_id: str
    name: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class DeleteDatasetCommand:
    """Command to delete a dataset."""
    dataset_id: str


@dataclass(frozen=True)
class CreateRegressionModelCommand:
    """Command to create a regression model."""
    dataset_id: str
    target_variable: str
    feature_variables: List[str]
    parameters: RegressionParameters


@dataclass(frozen=True)
class UpdateModelParametersCommand:
    """Command to update model parameters."""
    model_id: str
    parameters: RegressionParameters


@dataclass(frozen=True)
class DeleteModelCommand:
    """Command to delete a model."""
    model_id: str


@dataclass(frozen=True)
class GenerateSyntheticDataCommand:
    """Command to generate synthetic data."""
    config: DatasetConfig
    n_observations: int