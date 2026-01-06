"""
API Clients module for external data sources.

This module provides unified interfaces to various external APIs
for retrieving economic, social, and environmental data.
"""

from .bfs_client import fetch_bfs_data
from .world_bank_client import fetch_world_bank_data
from .fred_client import fetch_fred_data
from .who_client import fetch_who_health_data
from .eurostat_client import fetch_eurostat_data

__all__ = [
    'fetch_bfs_data',
    'fetch_world_bank_data',
    'fetch_fred_data',
    'fetch_who_health_data',
    'fetch_eurostat_data',
]