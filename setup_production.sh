#!/bin/bash
# Production Setup Script for Linear Regression Guide with AI Interpretation
# This script sets up the application for production use

set -e  # Exit on error

echo "🚀 Setting up Linear Regression Guide for Production"
echo "==================================================="

# Check if we're on macOS with Homebrew
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📍 macOS detected - using Homebrew approach"

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    echo "✅ Homebrew found"

    # Install Python if not available
    if ! command -v python3 &> /dev/null; then
        echo "📦 Installing Python via Homebrew..."
        brew install python@3.11
        echo "✅ Python installed"
    else
        echo "✅ Python already available"
    fi

    # Create virtual environment
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"

else
    echo "📍 Linux/other OS detected - using pip approach"
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Verify installations
echo "🔍 Verifying installations..."
python -c "
try:
    import streamlit
    print('✅ Streamlit:', streamlit.__version__)
except ImportError:
    print('❌ Streamlit not found')

try:
    import openai
    print('✅ OpenAI:', openai.__version__)
except ImportError:
    print('❌ OpenAI not found')

try:
    import numpy
    print('✅ NumPy:', numpy.__version__)
except ImportError:
    print('❌ NumPy not found')

try:
    import pandas
    print('✅ Pandas:', pandas.__version__)
except ImportError:
    print('❌ Pandas not found')

try:
    import plotly
    print('✅ Plotly:', plotly.__version__)
except ImportError:
    print('❌ Plotly not found')

try:
    import statsmodels
    print('✅ Statsmodels:', statsmodels.__version__)
except ImportError:
    print('❌ Statsmodels not found')
"

# Test API configuration
echo "🔑 Testing API configuration..."
python -c "
from src.perplexity_api import is_api_configured
if is_api_configured():
    print('✅ Perplexity API configured')
else:
    print('⚠️  Perplexity API not configured - set PERPLEXITY_API_KEY environment variable')
"

# Test basic syntax and file structure
echo "🔍 Testing application structure..."
python -c "
import os
import ast

def test_python_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f'❌ Syntax error in {filepath}: {e}')
        return False
    except Exception as e:
        print(f'❌ Error reading {filepath}: {e}')
        return False

# Test core modules
core_modules = ['src/app.py', 'src/session_state.py', 'src/sidebar.py', 'src/r_output.py']
syntax_ok = True
for module in core_modules:
    if os.path.exists(module):
        if test_python_file(module):
            print(f'✅ {os.path.basename(module)} syntax OK')
        else:
            syntax_ok = False
    else:
        print(f'❌ {module} not found')
        syntax_ok = False

if syntax_ok:
    print('✅ All core modules syntax check passed')
"

echo ""
echo "🎉 Production setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate  # Activate virtual environment"
echo "   export PERPLEXITY_API_KEY='your-api-key-here'  # If not set in secrets.toml"
echo "   streamlit run run.py"
echo ""
echo "📚 Documentation:"
echo "   docs/PRODUCTION_SETUP.md - Complete production guide"
echo "   docs/PERPLEXITY_INTEGRATION.md - AI feature documentation"
echo ""
echo "🔧 Configuration files:"
echo "   .streamlit/secrets.toml - API keys (create from secrets.toml.example)"
echo "   .streamlit/config.toml - Streamlit configuration"