"""
Application Layer - Use Cases, CQRS und Application Services.

Diese Schicht orchestriert Domain Objects für Business Use Cases.
Sie definiert die Anwendungsgrenze und koordiniert zwischen
Domain und Infrastructure Layer.

Prinzipien:
- Use Cases kapseln Anwendungslogik
- CQRS: Commands (Schreiboperationen) und Queries (Leseoperationen)
- Handlers verarbeiten Commands und Queries
- Mediator routet Anfragen an Handler
- Application Services koordinieren Domain Objects
"""

# Use Cases
from .use_cases import (
    CreateDatasetUseCase,
    CreateRegressionModelUseCase,
    AnalyzeModelQualityUseCase,
    CompareModelsUseCase,
    LoadDatasetUseCase,
    GenerateSyntheticDataUseCase
)

# Commands
from .commands import (
    Command,
    CreateDatasetCommand,
    UpdateDatasetCommand,
    DeleteDatasetCommand,
    CreateRegressionModelCommand,
    UpdateModelParametersCommand,
    DeleteModelCommand,
    GenerateSyntheticDataCommand
)

# Queries
from .queries import (
    Query,
    GetModelByIdQuery,
    GetDatasetByIdQuery,
    ListModelsQuery,
    ListDatasetsQuery,
    GetDatasetStatisticsQuery,
    GetModelDiagnosticsQuery,
    CompareModelsQuery,
    GetAvailableDataSourcesQuery
)

# Handlers (Lazy import function um zirkuläre Imports zu vermeiden)
def get_handlers():
    """Lazy import für Handlers."""
    from .handlers import (
        CreateDatasetHandler,
        CreateRegressionModelHandler,
        DeleteDatasetHandler,
        DeleteModelHandler,
        GetDatasetByIdHandler,
        ListDatasetsHandler,
        GetModelByIdHandler,
        ListModelsHandler,
        GetDatasetStatisticsHandler,
        GetModelDiagnosticsHandler,
        CompareModelsHandler,
        Mediator,
        create_mediator
    )
    return {
        'CreateDatasetHandler': CreateDatasetHandler,
        'CreateRegressionModelHandler': CreateRegressionModelHandler,
        'DeleteDatasetHandler': DeleteDatasetHandler,
        'DeleteModelHandler': DeleteModelHandler,
        'GetDatasetByIdHandler': GetDatasetByIdHandler,
        'ListDatasetsHandler': ListDatasetsHandler,
        'GetModelByIdHandler': GetModelByIdHandler,
        'ListModelsHandler': ListModelsHandler,
        'GetDatasetStatisticsHandler': GetDatasetStatisticsHandler,
        'GetModelDiagnosticsHandler': GetModelDiagnosticsHandler,
        'CompareModelsHandler': CompareModelsHandler,
        'Mediator': Mediator,
        'create_mediator': create_mediator
    }


def get_application_service_container():
    """Lazy import für ApplicationServiceContainer."""
    from .application_services import ApplicationServiceContainer
    return ApplicationServiceContainer


# Event Handlers
from .event_handlers import (
    EventBus,
    DatasetEventHandler,
    RegressionModelEventHandler,
    get_event_bus,
    publish_event
)

__all__ = [
    # Use Cases
    'CreateDatasetUseCase',
    'CreateRegressionModelUseCase',
    'AnalyzeModelQualityUseCase',
    'CompareModelsUseCase',
    'LoadDatasetUseCase',
    'GenerateSyntheticDataUseCase',

    # Commands
    'Command',
    'CreateDatasetCommand',
    'UpdateDatasetCommand',
    'DeleteDatasetCommand',
    'CreateRegressionModelCommand',
    'UpdateModelParametersCommand',
    'DeleteModelCommand',
    'GenerateSyntheticDataCommand',

    # Queries
    'Query',
    'GetModelByIdQuery',
    'GetDatasetByIdQuery',
    'ListModelsQuery',
    'ListDatasetsQuery',
    'GetDatasetStatisticsQuery',
    'GetModelDiagnosticsQuery',
    'CompareModelsQuery',
    'GetAvailableDataSourcesQuery',

    # Lazy Load Functions
    'get_handlers',
    'get_application_service_container',

    # Event Handlers
    'EventBus',
    'DatasetEventHandler',
    'RegressionModelEventHandler',
    'get_event_bus',
    'publish_event'
]