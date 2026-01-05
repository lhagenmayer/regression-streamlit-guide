#!/usr/bin/env bash
# Verification script for code quality checks
# Run this script to verify all linting and formatting tools work correctly

set -e  # Exit on error

echo "=================================="
echo "Code Quality Verification Script"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track overall status
ALL_PASSED=true

# Function to run check and report status
run_check() {
    local name=$1
    local command=$2
    local required=$3  # "required" or "optional"

    echo "Running $name..."
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name passed"
    else
        if [ "$required" = "required" ]; then
            echo -e "${RED}✗${NC} $name failed"
            ALL_PASSED=false
        else
            echo -e "${YELLOW}⚠${NC} $name has warnings (informational only)"
        fi
    fi
    echo ""
}

# Check if dependencies are installed
echo "Checking dependencies..."
if ! command -v black &> /dev/null; then
    echo -e "${RED}✗${NC} Black not installed. Run: pip install -r requirements-dev.txt"
    exit 1
fi
echo -e "${GREEN}✓${NC} All dependencies installed"
echo ""

# Run checks
run_check "Black formatting" "python -m black --check *.py tests/*.py" "required"
run_check "Flake8 linting" "python -m flake8 *.py tests/*.py" "required"
run_check "MyPy type checking" "python -m mypy app.py config.py data.py plots.py" "optional"

# Check if pre-commit is set up
if [ -f .git/hooks/pre-commit ]; then
    echo -e "${GREEN}✓${NC} Pre-commit hooks are installed"
    run_check "Pre-commit hooks" "pre-commit run --all-files" "required"
else
    echo -e "${RED}✗${NC} Pre-commit hooks not installed. Run: pre-commit install"
    ALL_PASSED=false
fi
echo ""

# Final summary
echo "=================================="
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}All required checks passed! ✓${NC}"
    echo "Your code meets all quality standards."
    exit 0
else
    echo -e "${RED}Some required checks failed! ✗${NC}"
    echo "Please fix the issues and try again."
    echo ""
    echo "Quick fixes:"
    echo "  - Format code: black *.py tests/*.py"
    echo "  - Fix linting: Review flake8 output and fix manually"
    echo "  - Install hooks: pre-commit install"
    exit 1
fi
