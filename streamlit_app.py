"""
Streamlit Cloud Entry Point for Linear Regression Guide

This file serves as the entry point for Streamlit Cloud deployment.
It simply imports and runs the main app from run.py.
"""

import sys
import os

# Add src to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the app
from app import *
