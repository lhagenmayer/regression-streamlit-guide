"""
API Endpoints - Platform Agnostic Business Logic.

These classes provide the core API logic WITHOUT any framework dependency.
They can be wrapped by any web framework:
- Flask
- FastAPI
- Django REST Framework
- Or served directly

All methods return JSON-serializable dictionaries.
"""

from typing import Dict, Any, Optional, List
import logging

from .serializers import (
    DataSerializer,
    StatsSerializer,
    PlotSerializer,
    ContentSerializer,
    PipelineSerializer,
)

logger = logging.getLogger(__name__)


class RegressionAPI:
    """
    Regression Analysis API.
    
    Provides endpoints for running regression analysis.
    100% framework agnostic - returns pure dictionaries.
    
    Usage (direct):
        api = RegressionAPI()
        result = api.run_simple(dataset="electronics", n=50)
        
    Usage (via Flask):
        @app.route('/api/regression/simple')
        def simple():
            return jsonify(api.run_simple(**request.json))
            
    Usage (via FastAPI):
        @app.post('/api/regression/simple')
        def simple(params: SimpleParams):
            return api.run_simple(**params.dict())
    """
    
    def __init__(self):
        """Initialize with lazy-loaded pipeline."""
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load pipeline to avoid import issues."""
        if self._pipeline is None:
            from ..pipeline import RegressionPipeline
            self._pipeline = RegressionPipeline()
        return self._pipeline
    
    def run_simple(
        self,
        dataset: str = "electronics",
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
        true_intercept: float = 0.6,
        true_slope: float = 0.52,
        include_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Run simple regression analysis.
        
        Args:
            dataset: Dataset name ("electronics", "advertising", "temperature")
            n: Sample size
            noise: Noise level
            seed: Random seed
            true_intercept: True β₀
            true_slope: True β₁
            include_predictions: Include y_pred and residuals arrays
            
        Returns:
            JSON-serializable result dictionary
        """
        logger.info(f"API: run_simple({dataset}, n={n})")
        
        try:
            result = self.pipeline.run_simple(
                dataset=dataset,
                n=n,
                noise=noise,
                seed=seed,
                true_intercept=true_intercept,
                true_slope=true_slope,
            )
            return {
                "success": True,
                "data": PipelineSerializer.serialize(result, include_predictions),
            }
        except Exception as e:
            logger.error(f"API error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def run_multiple(
        self,
        dataset: str = "cities",
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
        include_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Run multiple regression analysis.
        
        Args:
            dataset: Dataset name ("cities", "houses")
            n: Sample size
            noise: Noise level
            seed: Random seed
            include_predictions: Include predictions arrays
            
        Returns:
            JSON-serializable result dictionary
        """
        logger.info(f"API: run_multiple({dataset}, n={n})")
        
        try:
            result = self.pipeline.run_multiple(
                dataset=dataset,
                n=n,
                noise=noise,
                seed=seed,
            )
            return {
                "success": True,
                "data": PipelineSerializer.serialize(result, include_predictions),
            }
        except Exception as e:
            logger.error(f"API error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_datasets(self) -> Dict[str, Any]:
        """
        List available datasets.
        
        Returns:
            Dictionary of available datasets
        """
        return {
            "success": True,
            "data": {
                "simple": [
                    {
                        "id": "electronics",
                        "name": "Elektronikmarkt",
                        "description": "Verkaufsfläche vs Umsatz",
                        "icon": "🏪",
                    },
                    {
                        "id": "advertising",
                        "name": "Werbestudie",
                        "description": "Werbeausgaben vs Umsatz",
                        "icon": "📢",
                    },
                    {
                        "id": "temperature",
                        "name": "Eisverkauf",
                        "description": "Temperatur vs Verkauf",
                        "icon": "🍦",
                    },
                ],
                "multiple": [
                    {
                        "id": "cities",
                        "name": "Städtestudie",
                        "description": "Preis & Werbung → Umsatz",
                        "icon": "🏙️",
                    },
                    {
                        "id": "houses",
                        "name": "Hauspreise",
                        "description": "Fläche & Pool → Preis",
                        "icon": "🏠",
                    },
                ],
            },
        }


class ContentAPI:
    """
    Educational Content API.
    
    Returns framework-agnostic content structures.
    Any frontend can render these using their own components.
    """
    
    def __init__(self):
        self._pipeline = None
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..pipeline import RegressionPipeline
            self._pipeline = RegressionPipeline()
        return self._pipeline
    
    def get_simple_content(
        self,
        dataset: str = "electronics",
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Get educational content for simple regression.
        
        Returns complete content structure that can be rendered
        by any frontend framework.
        
        Args:
            dataset: Dataset name
            n: Sample size
            noise: Noise level
            seed: Random seed
            
        Returns:
            {
                "success": True,
                "content": { ... educational content ... },
                "plots": { ... plotly figures ... },
                "stats": { ... statistics ... },
            }
        """
        logger.info(f"API: get_simple_content({dataset}, n={n})")
        
        try:
            # Run pipeline
            result = self.pipeline.run_simple(
                dataset=dataset, n=n, noise=noise, seed=seed
            )
            
            # Build content
            from ..content import SimpleRegressionContent
            
            # Get flat stats for content builder
            stats_dict = StatsSerializer.to_flat_dict(result.stats, result.data)
            
            # Build content
            plot_keys = {
                "scatter": "scatter",
                "residuals": "residuals",
                "diagnostics": "diagnostics",
            }
            if result.plots.extra:
                plot_keys.update({k: k for k in result.plots.extra.keys()})
            
            builder = SimpleRegressionContent(stats_dict, plot_keys)
            content = builder.build()
            
            return {
                "success": True,
                "content": ContentSerializer.serialize(content),
                "plots": PlotSerializer.serialize_collection(result.plots),
                "stats": StatsSerializer.serialize_simple(result.stats),
                "data": DataSerializer.serialize_simple(result.data),
            }
            
        except Exception as e:
            logger.error(f"Content API error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_multiple_content(
        self,
        dataset: str = "cities",
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Get educational content for multiple regression.
        
        Returns:
            Complete content structure for any frontend
        """
        logger.info(f"API: get_multiple_content({dataset}, n={n})")
        
        try:
            # Run pipeline
            result = self.pipeline.run_multiple(
                dataset=dataset, n=n, noise=noise, seed=seed
            )
            
            # Build content
            from ..content import MultipleRegressionContent
            
            stats_dict = StatsSerializer.to_flat_dict(result.stats, result.data)
            
            plot_keys = {
                "scatter": "scatter",
                "residuals": "residuals",
                "diagnostics": "diagnostics",
            }
            
            builder = MultipleRegressionContent(stats_dict, plot_keys)
            content = builder.build()
            
            return {
                "success": True,
                "content": ContentSerializer.serialize(content),
                "plots": PlotSerializer.serialize_collection(result.plots),
                "stats": StatsSerializer.serialize_multiple(result.stats),
                "data": DataSerializer.serialize_multiple(result.data),
            }
            
        except Exception as e:
            logger.error(f"Content API error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_content_schema(self) -> Dict[str, Any]:
        """
        Get schema for content elements.
        
        Useful for frontend developers to understand the structure.
        """
        return {
            "success": True,
            "schema": {
                "element_types": [
                    "markdown", "metric", "metric_row", "formula", "plot",
                    "table", "columns", "expander", "info_box", "warning_box",
                    "success_box", "code_block", "divider", "chapter", "section"
                ],
                "structure": {
                    "EducationalContent": {
                        "title": "string",
                        "subtitle": "string",
                        "chapters": "Chapter[]",
                    },
                    "Chapter": {
                        "number": "string",
                        "title": "string",
                        "icon": "string (emoji)",
                        "sections": "Section[] | ContentElement[]",
                    },
                    "Section": {
                        "title": "string",
                        "icon": "string (emoji)",
                        "content": "ContentElement[]",
                    },
                    "Markdown": {"text": "string (markdown)"},
                    "Metric": {"label": "string", "value": "string", "help_text": "string", "delta": "string"},
                    "MetricRow": {"metrics": "Metric[]"},
                    "Formula": {"latex": "string", "inline": "boolean"},
                    "Plot": {"plot_key": "string", "title": "string", "description": "string", "height": "number"},
                    "Table": {"headers": "string[]", "rows": "string[][]", "caption": "string"},
                    "Columns": {"columns": "ContentElement[][]", "widths": "number[]"},
                    "Expander": {"title": "string", "content": "ContentElement[]", "expanded": "boolean"},
                    "InfoBox": {"content": "string"},
                    "WarningBox": {"content": "string"},
                    "SuccessBox": {"content": "string"},
                    "CodeBlock": {"code": "string", "language": "string"},
                    "Divider": {},
                },
            },
        }


class AIInterpretationAPI:
    """
    AI Interpretation API.
    
    Provides AI-powered interpretation of regression results.
    Completely framework agnostic.
    """
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy load Perplexity client."""
        if self._client is None:
            from ..ai import PerplexityClient
            self._client = PerplexityClient()
        return self._client
    
    def interpret(
        self,
        stats: Dict[str, Any],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Get AI interpretation of regression statistics.
        
        Args:
            stats: Statistics dictionary (from StatsSerializer.to_flat_dict)
            use_cache: Whether to use cached responses
            
        Returns:
            AI interpretation result
        """
        logger.info("API: interpret")
        
        response = self.client.interpret_r_output(stats, use_cache)
        
        return {
            "success": not response.error,
            "interpretation": {
                "content": response.content,
                "model": response.model,
                "cached": response.cached,
                "latency_ms": response.latency_ms,
            },
            "usage": response.usage,
            "citations": response.citations,
        }
    
    def get_r_output(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate R-style output for display.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            R-output string
        """
        r_output = self.client.generate_r_output(stats)
        
        return {
            "success": True,
            "r_output": r_output,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get AI service status.
        
        Returns:
            Status information
        """
        return {
            "success": True,
            "status": self.client.get_status(),
        }
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear interpretation cache.
        
        Returns:
            Cache clear result
        """
        result = self.client.clear_cache()
        return {
            "success": True,
            **result,
        }


# =========================================================================
# Unified API - Combines all endpoints
# =========================================================================

class UnifiedAPI:
    """
    Unified API combining all endpoints.
    
    Single entry point for all API functionality.
    """
    
    def __init__(self):
        self.regression = RegressionAPI()
        self.content = ContentAPI()
        self.ai = AIInterpretationAPI()
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """
        Generate OpenAPI specification.
        
        Returns:
            OpenAPI 3.0 specification
        """
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Regression Analysis API",
                "version": "1.0.0",
                "description": "Platform-agnostic API for regression analysis with educational content",
            },
            "paths": {
                "/api/regression/simple": {
                    "post": {
                        "summary": "Run simple regression",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "dataset": {"type": "string", "default": "electronics"},
                                            "n": {"type": "integer", "default": 50},
                                            "noise": {"type": "number", "default": 0.4},
                                            "seed": {"type": "integer", "default": 42},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {"200": {"description": "Regression result"}},
                    },
                },
                "/api/regression/multiple": {
                    "post": {
                        "summary": "Run multiple regression",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "dataset": {"type": "string", "default": "cities"},
                                            "n": {"type": "integer", "default": 75},
                                            "noise": {"type": "number", "default": 3.5},
                                            "seed": {"type": "integer", "default": 42},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {"200": {"description": "Regression result"}},
                    },
                },
                "/api/content/simple": {
                    "post": {
                        "summary": "Get educational content for simple regression",
                        "responses": {"200": {"description": "Educational content"}},
                    },
                },
                "/api/content/multiple": {
                    "post": {
                        "summary": "Get educational content for multiple regression",
                        "responses": {"200": {"description": "Educational content"}},
                    },
                },
                "/api/content/schema": {
                    "get": {
                        "summary": "Get content schema",
                        "responses": {"200": {"description": "Content schema"}},
                    },
                },
                "/api/ai/interpret": {
                    "post": {
                        "summary": "Get AI interpretation",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "stats": {"type": "object"},
                                            "use_cache": {"type": "boolean", "default": True},
                                        },
                                        "required": ["stats"],
                                    },
                                },
                            },
                        },
                        "responses": {"200": {"description": "AI interpretation"}},
                    },
                },
                "/api/ai/r-output": {
                    "post": {
                        "summary": "Generate R-style output",
                        "responses": {"200": {"description": "R output"}},
                    },
                },
                "/api/ai/status": {
                    "get": {
                        "summary": "Get AI service status",
                        "responses": {"200": {"description": "Status"}},
                    },
                },
                "/api/datasets": {
                    "get": {
                        "summary": "List available datasets",
                        "responses": {"200": {"description": "Dataset list"}},
                    },
                },
            },
        }
