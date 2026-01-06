"""
Application Layer - Use Cases and Application Services.

This layer orchestrates domain objects to fulfill business use cases.
It defines the application boundary and coordinates between domain
and infrastructure layers.

Key principles:
- Use Cases encapsulate application logic
- CQRS: Commands (write operations) and Queries (read operations)
- Application Services coordinate domain objects
"""

from .use_cases import (
    CreateRegressionModelUseCase,
    AnalyzeModelQualityUseCase,
    CompareModelsUseCase,
    LoadDatasetUseCase,
    GenerateSyntheticDataUseCase
)
from .commands import (
    CreateRegressionModelCommand,
    LoadDatasetCommand,
    GenerateDataCommand
)
from .queries import (
    GetModelQuery,
    GetDatasetQuery,
    ListModelsQuery,
    GetModelComparisonQuery
)

__all__ = [
    # Use Cases
    'CreateRegressionModelUseCase',
    'AnalyzeModelQualityUseCase',
    'CompareModelsUseCase',
    'LoadDatasetUseCase',
    'GenerateSyntheticDataUseCase',

    # Commands & Queries
    'CreateRegressionModelCommand',
    'LoadDatasetCommand',
    'GenerateDataCommand',
    'GetModelQuery',
    'GetDatasetQuery',
    'ListModelsQuery',
    'GetModelComparisonQuery'
]