# Testing Guide for Linear Regression Guide

This document provides comprehensive information about the testing infrastructure for the Linear Regression Guide Streamlit application.

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Writing New Tests](#writing-new-tests)
6. [CI/CD Integration](#cicd-integration)
7. [Coverage Reports](#coverage-reports)
8. [Troubleshooting](#troubleshooting)

## Overview

The testing suite includes:
- **Unit tests** for data generation and plotting functions
- **Integration tests** using Streamlit's AppTest framework
- **Performance regression tests** to ensure caching works correctly
- **Visual regression tests** for plot generation
- **Session state and caching tests**

## Test Structure

```
tests/
├── __init__.py                 # Test suite initialization
├── test_data.py               # Unit tests for data.py
├── test_plots.py              # Unit tests for plots.py
├── test_app_integration.py    # Streamlit AppTest integration tests
└── test_performance.py        # Performance and regression tests
```

## Running Tests

### Install Dependencies

First, install all dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
# Data generation tests
pytest tests/test_data.py -v

# Plotting function tests
pytest tests/test_plots.py -v

# Integration tests
pytest tests/test_app_integration.py -v

# Performance tests
pytest tests/test_performance.py -v
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run Streamlit-specific tests
pytest -m streamlit

# Exclude slow tests
pytest -m "not slow"
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Run a specific test class
pytest tests/test_data.py::TestSafeScalar -v

# Run a specific test method
pytest tests/test_data.py::TestSafeScalar::test_safe_scalar_with_float -v
```

## Test Categories

### Unit Tests (61 tests)

Located in `test_data.py` and `test_plots.py`, these tests cover:

#### Data Generation (`test_data.py` - 26 tests)
- `safe_scalar()` type conversion
- `generate_dataset()` for all dataset types
- `generate_multiple_regression_data()` with all variations
- `generate_simple_regression_data()` with all variations
- Edge cases (small/large n, zero/high noise)

#### Plotting Functions (`test_plots.py` - 35 tests)
- Significance helper functions (`get_signif_stars`, `get_signif_color`)
- 3D visualization helpers (`create_regression_mesh`, `create_zero_plane`)
- Plotly figure creation (scatter, 3D, surface, residual, bar, distribution)
- R-output display functions
- Edge cases (empty data, NaN/Inf values)

**Run unit tests:**
```bash
pytest tests/test_data.py tests/test_plots.py -v
```

### Integration Tests (16 tests)

Located in `test_app_integration.py`, these tests use Streamlit's AppTest framework to test:
- App initialization and structure
- Sidebar widget interactions (selectboxes, sliders, checkboxes)
- Session state management
- Tab navigation
- Complete user workflows
- Error handling and edge cases

**Run integration tests (excluding slow):**
```bash
pytest tests/test_app_integration.py -m "not slow" -v
```

### Performance Tests (13 tests)

Located in `test_performance.py`, these tests ensure:
- Data generation completes within expected time limits
- Caching provides expected speedup (2x minimum)
- No performance regressions
- Linear scaling with sample size
- Cache invalidation works correctly

**Run performance tests (excluding slow):**
```bash
pytest tests/test_performance.py -m "performance and not slow" -v
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests for individual functions
- `@pytest.mark.integration` - Integration tests for workflows
- `@pytest.mark.streamlit` - Tests using Streamlit AppTest
- `@pytest.mark.performance` - Performance and regression tests
- `@pytest.mark.slow` - Slow-running tests (skipped in quick runs)
- `@pytest.mark.visual` - Visual regression tests for plots

## Writing New Tests

### Structure

Follow this structure for new test files:

```python
"""
Brief description of what this test file covers.
"""

import pytest
from module_to_test import function_to_test


class TestFeatureName:
    """Test a specific feature or function."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test the basic use case."""
        result = function_to_test(input_data)
        assert result == expected_output

    @pytest.mark.unit
    def test_edge_case(self):
        """Test edge case behavior."""
        result = function_to_test(edge_case_input)
        assert result is not None
```

### Best Practices

1. **Descriptive Names**: Use clear, descriptive test names
2. **One Assertion**: Prefer one logical assertion per test
3. **Arrange-Act-Assert**: Follow the AAA pattern
4. **Use Markers**: Apply appropriate markers to categorize tests
5. **Document**: Add docstrings explaining what each test validates

### Example: Adding a New Data Generation Test

```python
@pytest.mark.unit
def test_new_dataset_generation(self):
    """Test generation of new dataset type."""
    result = generate_new_dataset(n=100, seed=42)

    assert "x" in result
    assert "y" in result
    assert len(result["x"]) == 100

    # Test reproducibility
    result2 = generate_new_dataset(n=100, seed=42)
    np.testing.assert_array_equal(result["x"], result2["x"])
```

### Example: Adding a Streamlit Integration Test

```python
@pytest.mark.streamlit
@pytest.mark.integration
def test_new_widget_interaction(self):
    """Test interaction with new widget."""
    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    # Interact with widget
    if len(at.slider) > 0:
        at.slider[0].set_value(50).run(timeout=30)
        assert not at.exception
```

## CI/CD Integration

### GitHub Actions Workflow

The `.github/workflows/tests.yml` workflow runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual trigger via workflow_dispatch

### Workflow Jobs

1. **test** - Runs on Python 3.9, 3.10, 3.11, 3.12
   - Unit tests with coverage
   - Integration tests (non-slow)
   - Performance tests (non-slow)
   - Uploads coverage to Codecov

2. **test-full** - Runs on Python 3.12 (PRs and main only)
   - All tests including slow tests
   - Full coverage report

### Triggering Manually

```bash
# Via GitHub CLI
gh workflow run tests.yml

# Via GitHub web interface
# Navigate to Actions → Run Tests → Run workflow
```

## Coverage Reports

### Generating Coverage Locally

```bash
# Generate coverage report
pytest --cov --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

### Coverage Goals

- **Overall**: Target 80%+ coverage
- **data.py**: Target 95%+ coverage (currently 99%)
- **plots.py**: Target 90%+ coverage (currently 96%)
- **app.py**: Integration tests provide indirect coverage

### Coverage Configuration

Coverage is configured in `.coveragerc`:
- Source: Current directory
- Omits: Tests, virtual environments, cache
- Excludes: Abstract methods, debug code, type checking

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Problem: "ModuleNotFoundError: No module named 'streamlit'"
# Solution: Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

#### 2. Timeout Errors

```bash
# Problem: Tests timeout with Streamlit AppTest
# Solution: Increase timeout or use simpler test
at.run(timeout=60)  # Increase from 30 to 60 seconds
```

#### 3. Cache Issues

```bash
# Problem: Cached data causing test failures
# Solution: Clear pytest cache
pytest --cache-clear
```

#### 4. Slow Tests

```bash
# Problem: Tests take too long
# Solution: Skip slow tests
pytest -m "not slow"
```

### Debug Mode

Run tests with more verbose output:

```bash
# Very verbose output
pytest -vv --tb=long

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb
```

### Performance Issues

If tests are slow:

1. Use markers to skip slow tests during development:
   ```bash
   pytest -m "not slow"
   ```

2. Run specific test files instead of the full suite:
   ```bash
   pytest tests/test_data.py  # Fast unit tests only
   ```

3. Use parallel execution (requires pytest-xdist):
   ```bash
   pytest -n auto  # Run tests in parallel
   ```

## Continuous Testing During Development

### Watch Mode

Use `pytest-watch` for continuous testing:

```bash
pip install pytest-watch
ptw tests/  # Reruns tests on file changes
```

### Pre-commit Hooks

Set up pre-commit hooks to run tests before committing:

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/test_data.py tests/test_plots.py -m "unit and not slow"
```

## Test Metrics

### Current Status

- ✅ **61 unit tests** - All passing
- ⚠️ **16 integration tests** - App has existing bug detected by tests
- ✅ **13 performance tests** - All passing (non-slow)

### Performance Benchmarks

- Data generation (n=75): < 1.0s (first call), < 0.01s (cached)
- Data generation (n=2000): < 5.0s
- Cache hit speedup: > 50x faster
- Unit test suite: ~4s
- Performance test suite: ~2.5s

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [Streamlit testing documentation](https://docs.streamlit.io/develop/api-reference/app-testing)
- [Coverage.py documentation](https://coverage.readthedocs.io/)

## Contributing

When adding new features:

1. Write tests first (TDD approach recommended)
2. Ensure all existing tests pass
3. Add appropriate markers
4. Update this documentation if needed
5. Verify coverage doesn't decrease

## Questions?

For questions about testing:
1. Check this guide
2. Review existing test examples
3. Check pytest documentation
4. Open an issue in the repository
