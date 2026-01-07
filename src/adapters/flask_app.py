"""
Flask Adapter - Renders regression analysis as a Flask web app.
"""

import json
import os
from typing import Any, Dict, Optional
from flask import Flask, render_template, request, jsonify, Response

from .base import BaseRenderer, RenderContext
from ..pipeline import RegressionPipeline
from ..pipeline.plot import PlotCollection
from ..data import get_multiple_regression_formulas, get_multiple_regression_descriptions
from ..config import get_logger

logger = get_logger(__name__)


class FlaskRenderer(BaseRenderer):
    """
    Flask implementation of the regression renderer.
    
    Serves a traditional web application with server-side rendering
    and interactive Plotly charts.
    """
    
    def __init__(self, template_folder: Optional[str] = None):
        self.pipeline = RegressionPipeline()
        
        # Setup template folder
        if template_folder is None:
            template_folder = os.path.join(os.path.dirname(__file__), "templates")
        
        self.app = Flask(
            __name__,
            template_folder=template_folder,
            static_folder=os.path.join(os.path.dirname(__file__), "static"),
        )
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup Flask routes."""
        
        @self.app.route("/")
        def index():
            """Main page."""
            return render_template("index.html")
        
        @self.app.route("/api/analyze", methods=["POST"])
        def analyze():
            """API endpoint for running analysis."""
            data = request.get_json()
            
            analysis_type = data.get("analysis_type", "simple")
            dataset = data.get("dataset", "electronics")
            n = data.get("n", 100)
            seed = data.get("seed", 42)
            show_formulas = data.get("show_formulas", True)
            
            try:
                if analysis_type == "simple":
                    result = self.pipeline.run_simple(dataset=dataset, n=n, seed=seed)
                    plots = self.pipeline.plot_builder.create_simple_plots(result.data, result.stats)
                    
                    context = RenderContext(
                        analysis_type="simple",
                        data=result.data,
                        stats=result.stats,
                        plots_json=self.serialize_plots(plots),
                        show_formulas=show_formulas,
                        dataset_name=dataset,
                    )
                else:
                    result = self.pipeline.run_multiple(dataset=dataset, n=n, seed=seed)
                    plots = self.pipeline.plot_builder.create_multiple_plots(result.data, result.stats)
                    
                    content = get_multiple_regression_descriptions(dataset)
                    formulas = get_multiple_regression_formulas(dataset)
                    
                    context = RenderContext(
                        analysis_type="multiple",
                        data=result.data,
                        stats=result.stats,
                        plots_json=self.serialize_plots(plots),
                        show_formulas=show_formulas,
                        content=content,
                        formulas=formulas,
                        dataset_name=dataset,
                    )
                
                return jsonify({
                    "success": True,
                    "data": context.to_dict()
                })
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route("/api/datasets")
        def get_datasets():
            """Get available datasets."""
            return jsonify({
                "simple": {
                    "electronics": "🏪 Elektronikmarkt",
                    "advertising": "📢 Werbestudie",
                },
                "multiple": {
                    "cities": "🌆 Städte-Studie",
                    "houses": "🏠 Immobilien",
                }
            })
        
        @self.app.route("/simple")
        def simple_regression_page():
            """Simple regression page."""
            dataset = request.args.get("dataset", "electronics")
            n = int(request.args.get("n", 100))
            seed = int(request.args.get("seed", 42))
            
            result = self.pipeline.run_simple(dataset=dataset, n=n, seed=seed)
            plots = self.pipeline.plot_builder.create_simple_plots(result.data, result.stats)
            
            context = RenderContext(
                analysis_type="simple",
                data=result.data,
                stats=result.stats,
                plots_json=self.serialize_plots(plots),
                show_formulas=True,
                dataset_name=dataset,
            )
            
            return render_template("simple_regression.html", **context.to_dict())
        
        @self.app.route("/multiple")
        def multiple_regression_page():
            """Multiple regression page."""
            dataset = request.args.get("dataset", "cities")
            n = int(request.args.get("n", 100))
            seed = int(request.args.get("seed", 42))
            
            result = self.pipeline.run_multiple(dataset=dataset, n=n, seed=seed)
            plots = self.pipeline.plot_builder.create_multiple_plots(result.data, result.stats)
            
            content = get_multiple_regression_descriptions(dataset)
            formulas = get_multiple_regression_formulas(dataset)
            
            context = RenderContext(
                analysis_type="multiple",
                data=result.data,
                stats=result.stats,
                plots_json=self.serialize_plots(plots),
                show_formulas=True,
                content=content,
                formulas=formulas,
                dataset_name=dataset,
            )
            
            return render_template("multiple_regression.html", **context.to_dict())
    
    def render(self, context: RenderContext) -> Response:
        """Render based on analysis type."""
        if context.analysis_type == "simple":
            return self.render_simple_regression(context)
        else:
            return self.render_multiple_regression(context)
    
    def render_simple_regression(self, context: RenderContext) -> str:
        """Render simple regression template."""
        return render_template("simple_regression.html", **context.to_dict())
    
    def render_multiple_regression(self, context: RenderContext) -> str:
        """Render multiple regression template."""
        return render_template("multiple_regression.html", **context.to_dict())
    
    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        """Run the Flask server."""
        logger.info(f"Starting Flask server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_flask_app() -> Flask:
    """Factory function to create Flask app (for WSGI servers)."""
    renderer = FlaskRenderer()
    return renderer.app


# Direct execution entry point
if __name__ == "__main__":
    renderer = FlaskRenderer()
    renderer.run(debug=True)
