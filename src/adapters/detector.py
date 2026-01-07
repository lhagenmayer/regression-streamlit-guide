"""
Framework Detector - Automatically detects which framework is running.
"""

import sys
import os
from enum import Enum
from typing import Optional


class Framework(Enum):
    """Supported frameworks."""
    STREAMLIT = "streamlit"
    FLASK = "flask"
    UNKNOWN = "unknown"


class FrameworkDetector:
    """
    Detects the current runtime framework.
    
    Detection order:
    1. Environment variable REGRESSION_FRAMEWORK
    2. Streamlit runtime context
    3. Flask app context
    4. Command line arguments
    5. Default to unknown
    """
    
    _cached_framework: Optional[Framework] = None
    
    @classmethod
    def detect(cls) -> Framework:
        """Detect the current framework."""
        if cls._cached_framework is not None:
            return cls._cached_framework
        
        # 1. Check environment variable (explicit override)
        env_framework = os.environ.get("REGRESSION_FRAMEWORK", "").lower()
        if env_framework == "streamlit":
            cls._cached_framework = Framework.STREAMLIT
            return cls._cached_framework
        elif env_framework == "flask":
            cls._cached_framework = Framework.FLASK
            return cls._cached_framework
        
        # 2. Check if running in Streamlit
        if cls._is_streamlit():
            cls._cached_framework = Framework.STREAMLIT
            return cls._cached_framework
        
        # 3. Check if running in Flask
        if cls._is_flask():
            cls._cached_framework = Framework.FLASK
            return cls._cached_framework
        
        # 4. Check command line arguments
        if cls._check_cli_args():
            return cls._cached_framework
        
        # 5. Default
        cls._cached_framework = Framework.UNKNOWN
        return cls._cached_framework
    
    @classmethod
    def _is_streamlit(cls) -> bool:
        """Check if running in Streamlit context."""
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            return ctx is not None
        except ImportError:
            pass
        
        # Fallback: check if streamlit is in sys.modules and was invoked
        if "streamlit" in sys.modules:
            # Check if we're being run by streamlit
            for arg in sys.argv:
                if "streamlit" in arg.lower():
                    return True
        
        return False
    
    @classmethod
    def _is_flask(cls) -> bool:
        """Check if running in Flask context."""
        try:
            from flask import current_app
            # Check if we're in an application context
            return current_app is not None
        except (ImportError, RuntimeError):
            pass
        
        return False
    
    @classmethod
    def _check_cli_args(cls) -> bool:
        """Check command line arguments for framework hints."""
        args = " ".join(sys.argv).lower()
        
        if "streamlit" in args:
            cls._cached_framework = Framework.STREAMLIT
            return True
        elif "flask" in args or "gunicorn" in args or "waitress" in args:
            cls._cached_framework = Framework.FLASK
            return True
        
        return False
    
    @classmethod
    def reset(cls) -> None:
        """Reset cached detection (useful for testing)."""
        cls._cached_framework = None
    
    @classmethod
    def is_streamlit(cls) -> bool:
        """Convenience method to check if running Streamlit."""
        return cls.detect() == Framework.STREAMLIT
    
    @classmethod
    def is_flask(cls) -> bool:
        """Convenience method to check if running Flask."""
        return cls.detect() == Framework.FLASK
