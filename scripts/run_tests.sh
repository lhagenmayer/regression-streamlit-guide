#!/usr/bin/env bash
# Minimal test runner for a personal project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RUN_UNIT=true
RUN_INTEGRATION=true
WITH_COVERAGE=false
VERBOSE=false

usage() {
    cat <<EOF
Minimal test runner

Usage:
  $0              # run unit + integration + error-handling tests
  $0 --unit       # run only unit tests
  $0 --integration# run only integration tests
  $0 --coverage   # include coverage report
  $0 --verbose    # verbose pytest output
  $0 --help       # show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_INTEGRATION=false
            ;;
        --integration)
            RUN_UNIT=false
            ;;
        --coverage)
            WITH_COVERAGE=true
            ;;
        --verbose|-v)
            VERBOSE=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

if [ ! -d "$PROJECT_ROOT/tests" ]; then
    echo "Run from project root" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

pytest_cmd() {
    local files="$1"
    local markers="$2"
    local cmd="python -m pytest"

    if [ -n "$markers" ]; then
        cmd="$cmd -m $markers"
    fi

    if [ "$WITH_COVERAGE" = true ]; then
        cmd="$cmd --cov=src --cov-report=term-missing"
    fi

    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi

    cmd="$cmd $files"
    echo "→ $cmd"
    eval "$cmd"
}

UNIT_FILES="tests/test_data.py tests/test_plots.py tests/test_logging.py tests/test_accessibility.py tests/test_error_handling.py"
INTEGRATION_FILES="tests/test_app_integration.py"

if [ "$RUN_UNIT" = true ]; then
    pytest_cmd "$UNIT_FILES" "unit"
fi

if [ "$RUN_INTEGRATION" = true ]; then
    pytest_cmd "$INTEGRATION_FILES" "integration"
fi

echo "Done."