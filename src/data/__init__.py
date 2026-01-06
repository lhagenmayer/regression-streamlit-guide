"""
Data handling package for the Linear Regression Guide.

This package contains data generation, API clients, and content management.
"""

from .data_generators import *
from .data_generators.mock_data_generator import safe_scalar
from .api_clients import *
from .content import *
from .data_loading import *
from .data_preparation import *