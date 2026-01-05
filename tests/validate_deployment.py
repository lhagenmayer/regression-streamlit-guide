#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Validator

This script validates that the app is ready for Streamlit Cloud deployment.
"""

import sys
import os
from pathlib import Path

# Required packages that must be present in requirements.txt
REQUIRED_PACKAGES = ["streamlit", "numpy", "pandas", "plotly", "statsmodels", "scipy"]

def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def check_requirements_file():
    """Check if requirements.txt exists and has required packages."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    with open("requirements.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    missing = []
    
    for package in REQUIRED_PACKAGES:
        if package.lower() not in content.lower():
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages in requirements.txt: {', '.join(missing)}")
        return False
    else:
        print(f"✅ requirements.txt contains all required packages")
        return True

def check_imports():
    """Check if main modules can be imported."""
    modules_to_check = ["config", "data", "plots", "content"]
    all_ok = True
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ Module {module}.py imports successfully")
        except Exception as e:
            print(f"❌ Module {module}.py import failed: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Streamlit Cloud Deployment Validation")
    print("=" * 60)
    print()
    
    checks = []
    
    # Check required files
    print("Checking required files...")
    checks.append(check_file_exists("app.py", "Main app file"))
    checks.append(check_file_exists("requirements.txt", "Requirements file"))
    checks.append(check_file_exists(".streamlit/config.toml", "Streamlit config"))
    checks.append(check_file_exists("DEPLOYMENT.md", "Deployment guide"))
    print()
    
    # Check requirements.txt content
    print("Checking requirements.txt content...")
    checks.append(check_requirements_file())
    print()
    
    # Check module imports
    print("Checking module imports...")
    checks.append(check_imports())
    print()
    
    # Summary
    print("=" * 60)
    if all(checks):
        print("✅ All checks passed! App is ready for Streamlit Cloud deployment.")
        print()
        print("Next steps:")
        print("1. Push changes to GitHub")
        print("2. Visit https://share.streamlit.io")
        print("3. Connect your GitHub repository")
        print("4. Set app.py as the main file")
        print("5. Deploy!")
        print()
        print("See DEPLOYMENT.md for detailed instructions.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
