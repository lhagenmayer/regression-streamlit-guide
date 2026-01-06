"""
Infrastructure Layer - External concerns and implementations.

This layer contains implementations of interfaces defined in the domain
and application layers. It handles external concerns like:

- Data persistence (repositories)
- External APIs (adapters)
- Framework integrations (Streamlit, web APIs)
- Configuration management
- Dependency injection container

The infrastructure layer depends on the domain and application layers,
but domain and application layers do not depend on infrastructure.
"""

from .dependency_container import DependencyContainer
from .repositories import InMemoryDatasetRepository, InMemoryModelRepository
from .adapters import StreamlitAdapter, APIClientAdapter

__all__ = [
    'DependencyContainer',
    'InMemoryDatasetRepository',
    'InMemoryModelRepository',
    'StreamlitAdapter',
    'APIClientAdapter'
]