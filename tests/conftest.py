"""
Pytest configuration and shared fixtures.
"""

import pytest
from unittest.mock import Mock

from src.infrastructure.dependency_container import DependencyContainer, InfrastructureContainer
from src.core.application.application_services import ApplicationServiceContainer


@pytest.fixture
def infrastructure_container():
    """Provide a test infrastructure dependency container."""
    return DependencyContainer().create_test_container()


@pytest.fixture
def application_container(infrastructure_container: InfrastructureContainer):
    """Provide a test application service container."""
    return ApplicationServiceContainer(infrastructure_container)


@pytest.fixture
def container(application_container):
    """Provide the full application container for backward compatibility."""
    return application_container


@pytest.fixture
def mock_dataset_repository():
    """Provide a mock dataset repository."""
    return Mock()


@pytest.fixture
def mock_model_repository():
    """Provide a mock model repository."""
    return Mock()