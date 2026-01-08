#!/usr/bin/env python3
"""
Unified Entry Point - 100% Platform Agnostic Application.

This module provides THREE ways to run the application:

1. **API Server** (for ANY frontend: Next.js, Vite, Vue, Angular, etc.)
   python run.py --api
   python run.py --api --port 8000

2. **Flask Web App** (traditional server-rendered HTML)
   python run.py --flask
   python run.py --flask --port 5000

3. **Streamlit App** (interactive Python UI)
   streamlit run run.py
   python run.py --streamlit

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                              run.py                                  │
    │                         (Auto-Detection)                             │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ↓                       ↓                       ↓
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
    │   REST API      │   │   Flask App     │   │   Streamlit App         │
    │   (JSON only)   │   │   (HTML)        │   │   (Interactive)         │
    └────────┬────────┘   └────────┬────────┘   └────────────┬────────────┘
             │                     │                         │
             │                     │                         │
             └─────────────────────┴─────────────────────────┘
                                   │
                                   ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Core Pipeline (Platform-Agnostic)                 │
    │         DataFetcher → StatisticsCalculator → PlotBuilder             │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                 Content Layer (Pure Data Structures)                 │
    │       ContentBuilder → EducationalContent (JSON-serializable)        │
    └─────────────────────────────────────────────────────────────────────┘

Frontend Integration Examples:

    # Next.js / Vite / React:
    fetch('http://localhost:8000/api/content/simple', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: 'electronics', n: 50 })
    })
    .then(res => res.json())
    .then(data => {
        // data.content - Educational content structure
        // data.plots   - Plotly figures (JSON)
        // data.stats   - Statistics
    });

    # Vue.js:
    const { data } = await axios.post('/api/content/simple', { n: 50 });
    
    # Any HTTP client:
    curl -X POST http://localhost:8000/api/content/simple \\
         -H "Content-Type: application/json" \\
         -d '{"dataset": "electronics", "n": 50}'
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_framework() -> str:
    """
    Detect which framework to use.
    
    Priority:
    1. Command line arguments (--api, --flask, --streamlit)
    2. Environment variables
    3. Runtime detection (Streamlit context)
    4. Default to showing help
    
    Returns:
        'api', 'streamlit', 'flask', or 'help'
    """
    # Check command line arguments
    if '--api' in sys.argv:
        return 'api'
    if '--flask' in sys.argv:
        return 'flask'
    if '--streamlit' in sys.argv:
        return 'streamlit'
    
    # Check environment variables
    if os.environ.get('RUN_API'):
        return 'api'
    if os.environ.get('FLASK_APP'):
        return 'flask'
    
    # Check if running in Streamlit
    try:
        import streamlit.runtime.scriptrunner as sr
        ctx = sr.get_script_run_ctx()
        if ctx is not None:
            return 'streamlit'
    except (ImportError, Exception):
        pass
    
    # Check if imported by Streamlit CLI
    if any('streamlit' in arg.lower() for arg in sys.argv):
        return 'streamlit'
    
    # Default: show help if run directly without arguments
    if __name__ == '__main__' and len(sys.argv) == 1:
        return 'help'
    
    return 'streamlit'


def run_api_server(host: str = "0.0.0.0", port: int = 8000, cors_origins: list = None):
    """
    Run pure REST API server.
    
    This server can be consumed by ANY frontend:
    - Next.js
    - Vite/React
    - Vue.js
    - Angular
    - Mobile Apps
    - Any HTTP client
    """
    from src.api import create_api_server
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🚀 Regression Analysis REST API                        ║
║                      100% Platform Agnostic                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Use this API with ANY frontend framework:                                ║
║  • Next.js / React                                                        ║
║  • Vite / Vue.js                                                          ║
║  • Angular / Svelte                                                       ║
║  • Mobile Apps (iOS/Android)                                              ║
║  • Any HTTP client                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    app = create_api_server(cors_origins=cors_origins or ["*"])
    
    print(f"📡 API Server: http://{host}:{port}")
    print(f"📚 OpenAPI Spec: http://{host}:{port}/api/openapi.json")
    print(f"❤️  Health Check: http://{host}:{port}/api/health")
    print()
    print("Endpoints:")
    print("  POST /api/regression/simple     - Run simple regression")
    print("  POST /api/regression/multiple   - Run multiple regression")
    print("  POST /api/content/simple        - Get educational content (simple)")
    print("  POST /api/content/multiple      - Get educational content (multiple)")
    print("  GET  /api/content/schema        - Get content schema")
    print("  POST /api/ai/interpret          - AI interpretation")
    print("  GET  /api/datasets              - List available datasets")
    print()
    
    app.run(host=host, port=port, debug=False, threaded=True)


def run_streamlit():
    """Run Streamlit application."""
    from src.adapters.streamlit.app import run_streamlit_app
    run_streamlit_app()


def run_flask(host: str = "0.0.0.0", port: int = 5000):
    """Run Flask web application (HTML rendering)."""
    from src.adapters.flask_app import run_flask as flask_run
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🌐 Flask Web Application                               ║
║                    Server-Side HTML Rendering                             ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    flask_run(host=host, port=port)


def create_app():
    """Create Flask app for WSGI servers (gunicorn, uwsgi, etc.)."""
    from src.adapters.flask_app import create_flask_app
    return create_flask_app()


def create_api_app():
    """Create API app for WSGI servers."""
    from src.api import create_api_server
    return create_api_server()


def show_help():
    """Show usage help."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║              📊 Regression Analysis - Platform Agnostic                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  Usage:                                                                   ║
║                                                                           ║
║    1. REST API (for Next.js, Vite, Vue, etc.):                           ║
║       python run.py --api [--port 8000]                                  ║
║                                                                           ║
║    2. Flask Web App (server-rendered HTML):                              ║
║       python run.py --flask [--port 5000]                                ║
║                                                                           ║
║    3. Streamlit App (interactive Python UI):                             ║
║       streamlit run run.py                                               ║
║       python run.py --streamlit                                          ║
║                                                                           ║
║  Options:                                                                 ║
║    --api        Run pure REST API server                                 ║
║    --flask      Run Flask web application                                ║
║    --streamlit  Run Streamlit application                                ║
║    --port PORT  Specify port number                                      ║
║    --host HOST  Specify host (default: 0.0.0.0)                         ║
║                                                                           ║
║  Examples:                                                                ║
║    python run.py --api --port 8000                                       ║
║    python run.py --flask --port 5000                                     ║
║    streamlit run run.py -- --port 8501                                   ║
║                                                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Regression Analysis - Platform Agnostic Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--api', action='store_true', help='Run REST API server')
    parser.add_argument('--flask', action='store_true', help='Run Flask web app')
    parser.add_argument('--streamlit', action='store_true', help='Run Streamlit app')
    parser.add_argument('--port', type=int, default=None, help='Port number')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    
    # Parse known args (ignore Streamlit args)
    args, _ = parser.parse_known_args()
    
    framework = detect_framework()
    
    if framework == 'help':
        show_help()
        return
    
    if framework == 'api':
        port = args.port or 8000
        run_api_server(host=args.host, port=port)
    elif framework == 'flask':
        port = args.port or 5000
        run_flask(host=args.host, port=port)
    elif framework == 'streamlit':
        run_streamlit()
    else:
        show_help()


# =========================================================================
# Execution
# =========================================================================

framework = detect_framework()

if framework == 'streamlit':
    # Streamlit mode - execute immediately
    run_streamlit()
elif framework in ('api', 'flask', 'help'):
    # Other modes - run main()
    if __name__ == '__main__':
        main()
    else:
        # For WSGI servers
        if framework == 'api':
            app = create_api_app()
        else:
            app = create_app()
else:
    if __name__ == '__main__':
        main()
