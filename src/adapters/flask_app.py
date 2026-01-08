"""
Flask Application - 100% Platform Agnostic.

Uses the same API layer as external frontends (Next.js, Vite, etc.)
This ensures consistency across all frontends.

Architecture:
    Flask App → API Layer → Core Pipeline
    
    Same as:
    Next.js App → HTTP → API Layer → Core Pipeline
"""

import json
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any

from ..config import get_logger
from ..api import RegressionAPI, ContentAPI, AIInterpretationAPI
from .renderers import HTMLContentRenderer

logger = get_logger(__name__)


def create_flask_app() -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__, template_folder='templates')
    
    # Initialize APIs (same as external frontends would use)
    regression_api = RegressionAPI()
    content_api = ContentAPI()
    ai_api = AIInterpretationAPI()
    
    @app.route('/')
    def index():
        """Landing page."""
        # Get available datasets from API
        datasets = regression_api.get_datasets()
        return render_template(
            'index.html',
            datasets=datasets['data'],
            api_status=ai_api.get_status()['status']
        )
    
    @app.route('/simple')
    def simple_regression():
        """Simple regression analysis page."""
        # Get parameters from query string
        dataset = request.args.get('dataset', 'electronics')
        n_points = int(request.args.get('n', 50))
        noise = float(request.args.get('noise', 0.4))
        seed = int(request.args.get('seed', 42))
        
        # Call Content API (same as external frontend would)
        response = content_api.get_simple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
        
        if not response['success']:
            return render_template('error.html', error=response.get('error', 'Unknown error'))
        
        # Extract data from API response
        content = response['content']
        plots = response['plots']
        stats = response['stats']
        data = response['data']
        
        # Build flat stats dict for template
        stats_dict = _flatten_stats_for_template(stats, data)
        
        # Render content to HTML using HTMLContentRenderer
        from ..content import SimpleRegressionContent
        content_builder = SimpleRegressionContent(stats_dict, {})
        content_obj = content_builder.build()
        
        renderer = HTMLContentRenderer(
            plots=plots,  # Pass serialized plots
            data=data,
            stats=stats_dict
        )
        content_dict = renderer.render_to_dict(content_obj)
        
        # Get available datasets for dropdown
        datasets = regression_api.get_datasets()
        
        return render_template(
            'educational_content.html',
            title=content['title'],
            subtitle=content['subtitle'],
            content_html=content_dict['full_html'],
            chapters=content_dict['chapters'],
            analysis_type='simple',
            dataset=dataset,
            datasets=datasets['data']['simple'],
            n_points=n_points,
            noise=noise,
            seed=seed,
            stats=stats_dict,
            plots_json=json.dumps(plots),
            ai_configured=ai_api.get_status()['status']['configured']
        )
    
    @app.route('/multiple')
    def multiple_regression():
        """Multiple regression analysis page."""
        # Get parameters
        dataset = request.args.get('dataset', 'cities')
        n_points = int(request.args.get('n', 75))
        noise = float(request.args.get('noise', 3.5))
        seed = int(request.args.get('seed', 42))
        
        # Call Content API
        response = content_api.get_multiple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
        
        if not response['success']:
            return render_template('error.html', error=response.get('error', 'Unknown error'))
        
        # Extract data
        content = response['content']
        plots = response['plots']
        stats = response['stats']
        data = response['data']
        
        # Build flat stats dict
        stats_dict = _flatten_multiple_stats_for_template(stats, data)
        
        # Render content
        from ..content import MultipleRegressionContent
        content_builder = MultipleRegressionContent(stats_dict, {})
        content_obj = content_builder.build()
        
        renderer = HTMLContentRenderer(
            plots=plots,
            data=data,
            stats=stats_dict
        )
        content_dict = renderer.render_to_dict(content_obj)
        
        # Get available datasets
        datasets = regression_api.get_datasets()
        
        return render_template(
            'educational_content.html',
            title=content['title'],
            subtitle=content['subtitle'],
            content_html=content_dict['full_html'],
            chapters=content_dict['chapters'],
            analysis_type='multiple',
            dataset=dataset,
            datasets=datasets['data']['multiple'],
            n_points=n_points,
            noise=noise,
            seed=seed,
            stats=stats_dict,
            plots_json=json.dumps(plots),
            ai_configured=ai_api.get_status()['status']['configured']
        )
    
    # =========================================================================
    # API ENDPOINTS - Proxy to API Layer
    # =========================================================================
    
    @app.route('/api/datasets', methods=['GET'])
    def api_datasets():
        """Get available datasets via API."""
        return jsonify(regression_api.get_datasets())
    
    @app.route('/api/regression/simple', methods=['POST'])
    def api_simple_regression():
        """Run simple regression via API."""
        data = request.get_json() or {}
        return jsonify(regression_api.run_simple(**data))
    
    @app.route('/api/regression/multiple', methods=['POST'])
    def api_multiple_regression():
        """Run multiple regression via API."""
        data = request.get_json() or {}
        return jsonify(regression_api.run_multiple(**data))
    
    @app.route('/api/content/simple', methods=['POST'])
    def api_content_simple():
        """Get simple regression content via API."""
        data = request.get_json() or {}
        return jsonify(content_api.get_simple_content(**data))
    
    @app.route('/api/content/multiple', methods=['POST'])
    def api_content_multiple():
        """Get multiple regression content via API."""
        data = request.get_json() or {}
        return jsonify(content_api.get_multiple_content(**data))
    
    @app.route('/api/content/schema', methods=['GET'])
    def api_content_schema():
        """Get content schema via API."""
        return jsonify(content_api.get_content_schema())
    
    @app.route('/api/ai/interpret', methods=['POST'])
    def api_ai_interpret():
        """
        AI interpretation via API.
        
        Accepts JSON with 'stats' field.
        Returns JSON with interpretation.
        """
        data = request.get_json() or {}
        stats = data.get('stats', {})
        use_cache = data.get('use_cache', True)
        
        result = ai_api.interpret(stats=stats, use_cache=use_cache)
        return jsonify(result)
    
    @app.route('/api/ai/interpret-html', methods=['POST'])
    def api_ai_interpret_html():
        """
        AI interpretation via API - returns HTML.
        
        For HTMX integration - returns rendered HTML.
        """
        from ..ai.ui_components import AIInterpretationHTML
        from ..ai import PerplexityClient
        
        data = request.get_json() or {}
        stats = data.get('stats', {})
        
        # Get interpretation via API
        result = ai_api.interpret(stats=stats, use_cache=True)
        
        # Render as HTML
        client = PerplexityClient()
        ui = AIInterpretationHTML(stats, client)
        
        # Build response object
        class ResponseObj:
            def __init__(self, result):
                interp = result.get('interpretation', {})
                self.content = interp.get('content', '')
                self.model = interp.get('model', 'unknown')
                self.cached = interp.get('cached', False)
                self.latency_ms = interp.get('latency_ms', 0)
                self.error = not result.get('success', False)
                self.usage = result.get('usage', {})
                self.citations = result.get('citations', [])
        
        response_obj = ResponseObj(result)
        html = ui.render_response(response_obj)
        
        return html
    
    @app.route('/api/ai/r-output', methods=['POST'])
    def api_ai_r_output():
        """Generate R-style output via API."""
        data = request.get_json() or {}
        stats = data.get('stats', {})
        return jsonify(ai_api.get_r_output(stats))
    
    @app.route('/api/ai/status', methods=['GET'])
    def api_ai_status():
        """Get AI service status via API."""
        return jsonify(ai_api.get_status())
    
    @app.route('/api/openapi.json', methods=['GET'])
    def api_openapi():
        """Get OpenAPI specification."""
        from ..api.endpoints import UnifiedAPI
        unified = UnifiedAPI()
        return jsonify(unified.get_openapi_spec())
    
    @app.route('/api/health', methods=['GET'])
    def api_health():
        """Health check endpoint."""
        return jsonify({
            'status': 'ok',
            'framework': 'flask',
            'api_powered': True
        })
    
    # =========================================================================
    # AI INTERPRETATION PAGES
    # =========================================================================
    
    @app.route('/interpret/<analysis_type>')
    def interpret_page(analysis_type: str):
        """
        Dedicated page for AI interpretation.
        """
        dataset = request.args.get('dataset', 'electronics' if analysis_type == 'simple' else 'cities')
        n_points = int(request.args.get('n', 50 if analysis_type == 'simple' else 75))
        
        # Get content via API
        if analysis_type == 'simple':
            response = content_api.get_simple_content(dataset=dataset, n=n_points)
            stats_dict = _flatten_stats_for_template(response['stats'], response['data'])
        else:
            response = content_api.get_multiple_content(dataset=dataset, n=n_points)
            stats_dict = _flatten_multiple_stats_for_template(response['stats'], response['data'])
        
        # Get AI interpretation via API
        interp_result = ai_api.interpret(stats=stats_dict)
        r_output_result = ai_api.get_r_output(stats_dict)
        
        return render_template(
            'interpret.html',
            analysis_type=analysis_type,
            dataset=dataset,
            n_points=n_points,
            stats=stats_dict,
            interpretation=interp_result.get('interpretation', {}),
            r_output=r_output_result.get('r_output', ''),
            ai_configured=ai_api.get_status()['status']['configured'],
            usage=interp_result.get('usage', {}),
            citations=interp_result.get('citations', [])
        )
    
    return app


def _flatten_stats_for_template(stats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten API stats response for Jinja2 templates."""
    coefficients = stats.get('coefficients', {})
    model_fit = stats.get('model_fit', {})
    t_tests = stats.get('t_tests', {})
    sum_of_squares = stats.get('sum_of_squares', {})
    sample = stats.get('sample', {})
    standard_errors = stats.get('standard_errors', {})
    extra = stats.get('extra', {})
    
    return {
        # Context
        'context_title': data.get('context', {}).get('title', 'Regressionsanalyse'),
        'context_description': data.get('context', {}).get('description', ''),
        'x_label': data.get('x_label', 'X'),
        'y_label': data.get('y_label', 'Y'),
        'y_unit': data.get('y_unit', ''),
        
        # Sample
        'n': sample.get('n', 0),
        'df': sample.get('df', 0),
        
        # Coefficients
        'intercept': coefficients.get('intercept', 0),
        'slope': coefficients.get('slope', 0),
        
        # Standard errors
        'se_intercept': standard_errors.get('intercept', 0),
        'se_slope': standard_errors.get('slope', 0),
        
        # t-tests
        't_intercept': t_tests.get('intercept', {}).get('t_value', 0),
        't_slope': t_tests.get('slope', {}).get('t_value', 0),
        'p_intercept': t_tests.get('intercept', {}).get('p_value', 1),
        'p_slope': t_tests.get('slope', {}).get('p_value', 1),
        
        # Model fit
        'r_squared': model_fit.get('r_squared', 0),
        'r_squared_adj': model_fit.get('r_squared_adj', 0),
        
        # Sum of squares
        'sse': sum_of_squares.get('sse', 0),
        'sst': sum_of_squares.get('sst', 0),
        'ssr': sum_of_squares.get('ssr', 0),
        'mse': sum_of_squares.get('mse', 0),
        
        # Extra
        'correlation': extra.get('correlation', 0),
        'x_mean': extra.get('x_mean', 0),
        'y_mean': extra.get('y_mean', 0),
        
        # Computed
        'f_statistic': (sum_of_squares.get('ssr', 0) / 1) / sum_of_squares.get('mse', 1) if sum_of_squares.get('mse', 0) > 0 else 0,
        'p_f': t_tests.get('slope', {}).get('p_value', 1),
    }


def _flatten_multiple_stats_for_template(stats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten API stats response for multiple regression templates."""
    coefficients = stats.get('coefficients', {})
    model_fit = stats.get('model_fit', {})
    t_tests = stats.get('t_tests', {})
    sample = stats.get('sample', {})
    standard_errors = stats.get('standard_errors', [0, 0, 0])
    
    slopes = coefficients.get('slopes', [0, 0])
    t_values = t_tests.get('t_values', [0, 0, 0])
    p_values = t_tests.get('p_values', [1, 1, 1])
    
    return {
        # Context
        'context_title': 'Multiple Regression',
        'context_description': f"Analyse von {data.get('y_label', 'Y')}",
        'x1_label': data.get('x1_label', 'X₁'),
        'x2_label': data.get('x2_label', 'X₂'),
        'y_label': data.get('y_label', 'Y'),
        
        # Sample
        'n': sample.get('n', 0),
        'k': sample.get('k', 2),
        'df': sample.get('n', 0) - 3,
        
        # Coefficients
        'intercept': coefficients.get('intercept', 0),
        'beta1': slopes[0] if len(slopes) > 0 else 0,
        'beta2': slopes[1] if len(slopes) > 1 else 0,
        
        # Standard errors
        'se_intercept': standard_errors[0] if len(standard_errors) > 0 else 0,
        'se_beta1': standard_errors[1] if len(standard_errors) > 1 else 0,
        'se_beta2': standard_errors[2] if len(standard_errors) > 2 else 0,
        
        # t-tests
        't_intercept': t_values[0] if len(t_values) > 0 else 0,
        't_beta1': t_values[1] if len(t_values) > 1 else 0,
        't_beta2': t_values[2] if len(t_values) > 2 else 0,
        'p_intercept': p_values[0] if len(p_values) > 0 else 1,
        'p_beta1': p_values[1] if len(p_values) > 1 else 1,
        'p_beta2': p_values[2] if len(p_values) > 2 else 1,
        
        # Model fit
        'r_squared': model_fit.get('r_squared', 0),
        'r_squared_adj': model_fit.get('r_squared_adj', 0),
        'f_statistic': model_fit.get('f_statistic', 0),
        'p_f': model_fit.get('f_p_value', 1),
    }


def run_flask(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """Run Flask application."""
    app = create_flask_app()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🌐 Flask Web Application                               ║
║                    100% API-Powered Architecture                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Uses the same API layer as external frontends (Next.js, Vite, etc.)     ║
║                                                                           ║
║  Pages:                                                                   ║
║    /              - Landing Page                                          ║
║    /simple        - Simple Regression Analysis                            ║
║    /multiple      - Multiple Regression Analysis                          ║
║    /interpret/*   - AI Interpretation                                     ║
║                                                                           ║
║  API Endpoints (same as REST API server):                                 ║
║    /api/datasets                  - List datasets                         ║
║    /api/regression/simple         - Run simple regression                 ║
║    /api/content/simple            - Get educational content               ║
║    /api/ai/interpret              - AI interpretation                     ║
║    /api/openapi.json              - OpenAPI specification                 ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📚 API: http://{host}:{port}/api/openapi.json")
    print()
    
    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run_flask()
