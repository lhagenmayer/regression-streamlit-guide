"""
Infrastructure Dependency Injection Container.

This provides access to infrastructure-level dependencies only.
Application layer dependencies (use cases) are wired separately.

Benefits:
- Loose coupling between infrastructure and application layers
- Clear separation of concerns
- Testable infrastructure components
- Proper dependency inversion using Protocol interfaces
"""

from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod

# Domain layer interfaces only
from ..core.domain.repositories import DatasetRepository, RegressionModelRepository

# Infrastructure layer only
from .repositories import InMemoryDatasetRepository, InMemoryModelRepository


class RepositoryFactory(Protocol):
    """Protocol for repository factory methods."""

    def create_dataset_repository(self) -> DatasetRepository: ...

    def create_model_repository(self) -> RegressionModelRepository: ...


class DefaultRepositoryFactory:
    """Default implementation of repository factory using in-memory repositories."""

    def create_dataset_repository(self) -> DatasetRepository:
        """Create dataset repository implementation."""
        return InMemoryDatasetRepository()

    def create_model_repository(self) -> RegressionModelRepository:
        """Create model repository implementation."""
        return InMemoryModelRepository()


class InfrastructureContainer(ABC):
    """
    Abstract Infrastructure Dependency Container using Dependency Inversion Principle.

    Provides access to infrastructure services only (repositories, adapters, external services).
    Uses Protocol interfaces for proper dependency inversion.
    Application layer services are wired separately to maintain proper dependency direction.
    """

    def __init__(self, repository_factory: Optional[RepositoryFactory] = None):
        """Initialize the infrastructure dependency container."""
        self._services: Dict[str, Any] = {}
        self._repository_factory = repository_factory or DefaultRepositoryFactory()
        self._initialize_infrastructure_dependencies()

    def _initialize_infrastructure_dependencies(self):
        """Initialize infrastructure dependencies using factory pattern."""

        # Use factory pattern for dependency inversion
        dataset_repo = self._repository_factory.create_dataset_repository()
        model_repo = self._repository_factory.create_model_repository()

        # Additional infrastructure services with proper typing
        self._services = {
            # Repositories (infrastructure implementations via factory)
            'dataset_repository': dataset_repo,
            'model_repository': model_repo,

            # Factory for testing/mocking
            'repository_factory': self._repository_factory,
        }

    def get_service(self, service_name: str) -> Any:
        """Get a service by name with proper error handling."""
        if service_name not in self._services:
            available_services = list(self._services.keys())
            raise ValueError(f"Service '{service_name}' not found. Available services: {available_services}")
        return self._services[service_name]

    def register_service(self, service_name: str, service: Any) -> None:
        """Register a new service with validation."""
        if not service_name or not service_name.strip():
            raise ValueError("Service name cannot be empty")
        self._services[service_name] = service

    def override_service(self, service_name: str, service: Any) -> None:
        """Override an existing service (useful for testing)."""
        if service_name not in self._services:
            raise ValueError(f"Cannot override non-existent service '{service_name}'")
        self._services[service_name] = service

    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self._services

    def list_services(self) -> list:
        """List all registered service names."""
        return list(self._services.keys())

    # Abstract methods for extensibility
    @abstractmethod
    def create_test_container(self) -> 'InfrastructureContainer':
        """Create a test container with mock services."""
        pass

    # Convenience properties for infrastructure services
    @property
    def dataset_repository(self) -> DatasetRepository:
        """Get dataset repository with proper typing."""
        return self.get_service('dataset_repository')

    @property
    def model_repository(self) -> RegressionModelRepository:
        """Get model repository with proper typing."""
        return self.get_service('model_repository')


class DependencyContainer(InfrastructureContainer):
    """
    Concrete implementation of Infrastructure Container.

    Provides the default infrastructure container implementation.
    """

    def create_test_container(self) -> 'InfrastructureContainer':
        """
        Create a test container with empty repositories for testing.

        This is useful for testing without side effects.
        """
        test_container = DependencyContainer()

        # Override with empty repositories for testing
        test_container.override_service('dataset_repository', InMemoryDatasetRepository())
        test_container.override_service('model_repository', InMemoryModelRepository())

        return test_container

    def get(self, service_name: str) -> Any:
        """Get a service by name."""
        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not found")
        return self._services[service_name]

    def register(self, service_name: str, service: Any) -> None:
        """Register a new service."""
        self._services[service_name] = service

    def override(self, service_name: str, service: Any) -> None:
        """Override an existing service (useful for testing)."""
        if service_name not in self._services:
            raise ValueError(f"Cannot override non-existent service '{service_name}'")
        self._services[service_name] = service

    # Convenience methods for infrastructure services
    @property
    def dataset_repository(self) -> DatasetRepository:
        """Get dataset repository."""
        return self.get('dataset_repository')

    @property
    def model_repository(self) -> InMemoryModelRepository:
        """Get model repository."""
        return self.get('model_repository')

    def create_test_container(self) -> 'DependencyContainer':
        """
        Create a test container with empty repositories for testing.

        This is useful for testing without side effects.
        """
        test_container = DependencyContainer()

        # Override with empty repositories for testing
        test_container.override('dataset_repository', InMemoryDatasetRepository())
        test_container.override('model_repository', InMemoryModelRepository())

        return test_container