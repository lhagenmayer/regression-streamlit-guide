"""
AI Module - Perplexity AI Integration.

Hauptfeature: Gesamtheitliche Interpretation des R-Outputs.

Usage:
    from src.ai import PerplexityClient
    
    client = PerplexityClient()
    
    if client.is_configured:
        response = client.interpret_r_output(stats_dict)
        print(response.content)
"""

from .perplexity_client import PerplexityClient, PerplexityConfig, PerplexityResponse

__all__ = [
    "PerplexityClient",
    "PerplexityConfig",
    "PerplexityResponse",
]
