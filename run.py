#!/usr/bin/env python3
"""
📊 Linear Regression Guide - Unified Entry Point

Auto-detects the framework and runs appropriately:
- Streamlit: `streamlit run run.py`
- Flask: `python run.py` or `flask run`

Environment variable override:
    REGRESSION_FRAMEWORK=streamlit|flask
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.adapters.detector import FrameworkDetector, Framework
from src.config import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point with auto-detection."""
    framework = FrameworkDetector.detect()
    
    logger.info(f"Detected framework: {framework.value}")
    
    if framework == Framework.STREAMLIT:
        run_streamlit()
    elif framework == Framework.FLASK:
        run_flask()
    else:
        # Default: try to determine from how we were called
        if _is_streamlit_invocation():
            run_streamlit()
        else:
            # Default to Flask for direct python execution
            print("🔍 Framework not detected. Defaulting to Flask.")
            print("   For Streamlit: streamlit run run.py")
            print("   For Flask: python run.py")
            print("")
            run_flask()


def _is_streamlit_invocation() -> bool:
    """Check if we're being run via streamlit command."""
    return any('streamlit' in arg.lower() for arg in sys.argv)


def run_streamlit():
    """Run as Streamlit app."""
    logger.info("Starting Streamlit app...")
    
    from src.adapters.streamlit_app import StreamlitRenderer
    
    app = StreamlitRenderer()
    app.run()


def run_flask(host: str = "0.0.0.0", port: int = 5000, debug: bool = True):
    """Run as Flask app."""
    logger.info(f"Starting Flask app on {host}:{port}...")
    
    from src.adapters.flask_app import FlaskRenderer
    
    app = FlaskRenderer()
    app.run(host=host, port=port, debug=debug)


# Flask WSGI entry point
def create_app():
    """Factory function for WSGI servers (gunicorn, waitress, etc.)."""
    from src.adapters.flask_app import create_flask_app
    return create_flask_app()


# Streamlit entry point (when run via `streamlit run`)
if FrameworkDetector.is_streamlit():
    run_streamlit()


if __name__ == "__main__":
    main()
