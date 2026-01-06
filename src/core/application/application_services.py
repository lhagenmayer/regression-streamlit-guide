"""
Application Services Layer.

This layer handles the wiring of application-level dependencies (use cases).
It coordinates between domain services and infrastructure repositories.

According to Clean Architecture:
- Infrastructure layer provides basic services (repositories, adapters)
- Application layer wires them into use cases
- Domain layer contains pure business logic
"""

from typing import Dict, Any

# Infrastructure layer
from ...infrastructure.dependency_container import InfrastructureContainer

# Domain layer
from ..domain.services import RegressionAnalysisService
from ..domain.repositories import DatasetRepository, RegressionModelRepository

# Application layer
from .use_cases import (
    CreateDatasetUseCase,
    CreateRegressionModelUseCase,
    AnalyzeModelQualityUseCase,
    CompareModelsUseCase,
    LoadDatasetUseCase,
    GenerateSyntheticDataUseCase
)
from .event_handlers import get_event_bus


class ApplicationServiceContainer:
    """
    Application Service Container.

    Wires application layer dependencies (use cases) using infrastructure services.
    This maintains proper dependency direction: Application → Domain → Infrastructure.
    """

    def __init__(self, infrastructure_container: InfrastructureContainer):
        """
        Initialize application services using dependency inversion.

        Args:
            infrastructure_container: Infrastructure dependency container (abstract)
        """
        if not isinstance(infrastructure_container, InfrastructureContainer):
            raise TypeError("infrastructure_container must be an InfrastructureContainer instance")

        self.infrastructure = infrastructure_container
        self._services: Dict[str, Any] = {}
        self._initialize_application_services()

    def _initialize_application_services(self):
        """Initialize application layer services (use cases)."""

        # Get infrastructure services
        dataset_repo = self.infrastructure.dataset_repository
        model_repo = self.infrastructure.model_repository

        # Initialize domain services with infrastructure
        regression_service = RegressionAnalysisService(dataset_repo)

        # Initialize use cases with domain services
        dataset_use_case = CreateDatasetUseCase(dataset_repo)
        create_model_use_case = CreateRegressionModelUseCase(
            dataset_use_case, regression_service
        )
        analyze_quality_use_case = AnalyzeModelQualityUseCase(regression_service)
        compare_models_use_case = CompareModelsUseCase(regression_service)
        load_dataset_use_case = LoadDatasetUseCase(dataset_use_case)
        generate_data_use_case = GenerateSyntheticDataUseCase(dataset_use_case)

        # Register application services
        self._services = {
            # Domain services (wired with infrastructure)
            'regression_service': regression_service,

            # Use cases (application layer)
            'dataset_use_case': dataset_use_case,
            'create_model_use_case': create_model_use_case,
            'analyze_quality_use_case': analyze_quality_use_case,
            'compare_models_use_case': compare_models_use_case,
            'load_dataset_use_case': load_dataset_use_case,
            'generate_data_use_case': generate_data_use_case,
        }

    def get(self, service_name: str) -> Any:
        """Get an application service by name."""
        if service_name not in self._services:
            raise ValueError(f"Application service '{service_name}' not found")
        return self._services[service_name]

    # Convenience properties for commonly used use cases
    @property
    def dataset_use_case(self) -> CreateDatasetUseCase:
        """Get dataset use case."""
        return self.get('dataset_use_case')

    @property
    def create_model_use_case(self) -> CreateRegressionModelUseCase:
        """Get create model use case."""
        return self.get('create_model_use_case')

    @property
    def analyze_quality_use_case(self) -> AnalyzeModelQualityUseCase:
        """Get analyze quality use case."""
        return self.get('analyze_quality_use_case')

    @property
    def regression_service(self) -> RegressionAnalysisService:
        """Get regression service."""
        return self.get('regression_service')

    @property
    def event_bus(self):
        """Get the event bus for publishing domain events."""
        return get_event_bus()