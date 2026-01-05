# 🧪 Tests (minimal)

This is a slimmed-down test suite for a personal project. It keeps the core unit and integration coverage and drops heavy/CI-only suites.

## Structure

```
tests/
├── conftest.py             # shared fixtures
├── test_data.py            # unit: data generation
├── test_plots.py           # unit: plotting helpers
├── test_logging.py         # unit: logging
├── test_accessibility.py   # unit: accessibility helpers
├── test_error_handling.py  # unit: edge cases
├── test_app_integration.py # integration: streamlit workflows
└── README.md
```

## Run tests

```bash
# all core tests
pytest

# unit only
pytest -m unit

# integration only
pytest -m integration

# with coverage
pytest --cov=src --cov-report=term-missing
```

Quick helper script:

```bash
./scripts/run_tests.sh            # unit + integration
./scripts/run_tests.sh --unit     # unit only
./scripts/run_tests.sh --integration
```

## Notes

- Requires `pytest` (install via `pip install -r requirements-dev.txt`).
- Markers used: `unit`, `integration`, `streamlit` (for app tests).