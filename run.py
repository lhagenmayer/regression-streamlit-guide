#!/usr/bin/env python3
"""
Entry point for the Linear Regression Guide application.

Usage:
    python run.py
    streamlit run run.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the app
from app import *
