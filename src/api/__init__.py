"""
API Module - 100% Platform Agnostic REST API.

This module provides a pure REST/JSON API that can be consumed by ANY frontend:
- Next.js
- Vite/React
- Vue.js
- Angular
- Svelte
- Mobile Apps
- Or any HTTP client

NO framework-specific code in this module.
All data is JSON serializable.
"""

from .serializers import (
    DataSerializer,
    StatsSerializer,
    PlotSerializer,
    ContentSerializer,
    PipelineSerializer,
)

from .endpoints import (
    RegressionAPI,
    ContentAPI,
    AIInterpretationAPI,
    UnifiedAPI,
)

from .server import create_api_server

__all__ = [
    # Serializers
    "DataSerializer",
    "StatsSerializer", 
    "PlotSerializer",
    "ContentSerializer",
    "PipelineSerializer",
    # API Endpoints
    "RegressionAPI",
    "ContentAPI",
    "AIInterpretationAPI",
    "UnifiedAPI",
    # Server
    "create_api_server",
]
