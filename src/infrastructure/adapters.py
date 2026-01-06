"""
Interface Adapters - Ports and Adapters pattern.

Adapters connect the application to external systems like:
- UI frameworks (Streamlit)
- External APIs
- File systems
- Databases
- Message queues

Adapters implement interfaces defined in the domain/application layers.
"""

from typing import Dict, Any, List, Optional, Protocol
from abc import ABC, abstractmethod

import numpy as np

from ..core.domain.entities import Dataset, RegressionModel
from ..core.domain.value_objects import DatasetConfig, RegressionParameters
from ..core.application.commands import Command
from ..core.application.queries import Query


# Interface definitions (Ports)
class UIAdapter(Protocol):
    """Interface for UI interactions."""
    def display_model_results(self, model: RegressionModel, dataset: Dataset) -> None: ...
    def get_user_parameters(self) -> Dict[str, Any]: ...
    def show_progress(self, message: str) -> None: ...
    def show_error(self, message: str) -> None: ...


class ExternalAPIAdapter(Protocol):
    """Interface for external API calls."""
    def fetch_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]: ...
    def is_available(self) -> bool: ...


# Adapter implementations
# NOTE: UI adapters should be moved to the UI layer to avoid infrastructure dependencies on UI frameworks
# TODO: Move StreamlitAdapter implementation to src/ui/adapters.py
# The infrastructure layer should only define interfaces, not UI framework implementations


class APIClientAdapter(ExternalAPIAdapter):
    """
    HTTP API client adapter.

    This adapter handles external API calls, abstracting away
    the details of HTTP requests, authentication, error handling, etc.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._session = None

    def fetch_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from external API.

        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters

        Returns:
            JSON response data
        """
        try:
            # This would implement actual HTTP calls
            # For now, return mock data
            return {
                "status": "success",
                "data": f"Mock data from {endpoint}",
                "params": params,
                "timestamp": "2024-01-06T12:00:00Z"
            }
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")

    def is_available(self) -> bool:
        """Check if the API is available."""
        try:
            # This would make a health check call
            return True
        except:
            return False


# Command/Query Bus implementations
class CommandBus:
    """Command bus for handling write operations."""

    def __init__(self):
        self._handlers: Dict[str, Any] = {}

    def register_handler(self, command_type: str, handler: Any) -> None:
        """Register a command handler."""
        self._handlers[command_type] = handler

    def send(self, command: Command) -> Any:
        """Send a command to its handler."""
        command_type = command.__class__.__name__
        if command_type not in self._handlers:
            raise ValueError(f"No handler registered for command: {command_type}")

        handler = self._handlers[command_type]
        return handler.handle(command)


class QueryBus:
    """Query bus for handling read operations."""

    def __init__(self):
        self._handlers: Dict[str, Any] = {}

    def register_handler(self, query_type: str, handler: Any) -> None:
        """Register a query handler."""
        self._handlers[query_type] = handler

    def ask(self, query: Query) -> Any:
        """Send a query to its handler."""
        query_type = query.__class__.__name__
        if query_type not in self._handlers:
            raise ValueError(f"No handler registered for query: {query_type}")

        handler = self._handlers[query_type]
        return handler.handle(query)