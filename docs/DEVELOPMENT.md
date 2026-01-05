# Development Guide

## Code Quality and Formatting Standards

This project follows professional Python development standards with automated code quality checks.

## Tools Used

### Black - Code Formatter

Black is an opinionated code formatter that ensures consistent code style across the project.

- **Line Length**: 100 characters
- **Configuration**: `pyproject.toml`
- **Target Python**: 3.9+

```bash
# Format all Python files
black *.py tests/*.py

# Check formatting without making changes
black --check *.py tests/*.py

# Show diff of what would change
black --diff *.py tests/*.py
```

### Flake8 - Linter

Flake8 checks for code style violations and potential errors.

- **Configuration**: `.flake8`
- **Max Line Length**: 100 characters
- **Complexity Limit**: 15

```bash
# Run flake8 on all files
flake8 *.py tests/*.py

# Show statistics
flake8 *.py tests/*.py --statistics

# Count violations
flake8 *.py tests/*.py --count
```

**Ignored Rules** (configured to work with Black):
- E203: Whitespace before ':'
- W503: Line break before binary operator
- E501: Line too long (handled by Black)
- F541: f-string without placeholders

### MyPy - Static Type Checker

MyPy provides optional static type checking for Python code.

- **Configuration**: `mypy.ini`
- **Mode**: Lenient (does not require all functions to be typed)

```bash
# Run mypy on main modules
mypy app.py config.py data.py plots.py

# Run mypy with more verbose output
mypy --show-error-codes app.py
```

**Note**: MyPy is configured to be lenient and runs with `continue-on-error` in CI/CD. It provides helpful warnings but doesn't fail the build.

## Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit.

### Installation

```bash
# Install pre-commit hooks
pip install pre-commit

# Set up git hooks
pre-commit install
```

### Usage

```bash
# Run manually on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files file1.py file2.py

# Update hooks to latest versions
pre-commit autoupdate

# Bypass hooks (not recommended)
git commit --no-verify -m "message"
```

### Hooks Configured

1. **Trailing Whitespace**: Remove trailing whitespace
2. **End of File Fixer**: Ensure files end with a newline
3. **YAML/JSON/TOML Checker**: Validate configuration files
4. **Large File Checker**: Prevent committing large files (>1MB)
5. **Merge Conflict Checker**: Detect unresolved merge conflicts
6. **Debug Statements**: Find accidentally committed debug statements
7. **Black**: Automatic code formatting
8. **Flake8**: Style and error checking

## Development Workflow

### 1. Before Starting Work

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify everything works
pre-commit run --all-files
pytest tests/
```

### 2. During Development

```bash
# Format code frequently
black *.py tests/*.py

# Check for issues
flake8 *.py tests/*.py

# Run tests
pytest tests/ -v
```

### 3. Before Committing

```bash
# Run all checks
pre-commit run --all-files

# Run full test suite
pytest tests/ --cov

# Verify no issues
flake8 *.py tests/*.py
```

### 4. Commit

```bash
# Pre-commit hooks run automatically
git add .
git commit -m "Your descriptive message"

# If hooks fail, fix issues and try again
git add .
git commit -m "Your descriptive message"
```

## GitHub Actions CI/CD

### Workflows

1. **tests.yml**: Runs full test suite on multiple Python versions
2. **lint.yml**: Runs code quality checks (Black, Flake8, MyPy)

### Lint Workflow

The lint workflow runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual trigger via workflow_dispatch

**Checks performed**:
- Black formatting check
- Flake8 linting
- MyPy type checking (non-blocking)

## Configuration Files

### pyproject.toml

Contains configuration for:
- Black formatter
- isort (import sorting)
- Pytest

### .flake8

Flake8 configuration including:
- Max line length
- Ignored rules
- Excluded directories
- Complexity limits

### mypy.ini

MyPy configuration with:
- Python version target
- Warning levels
- Module-specific settings
- Import handling

### .pre-commit-config.yaml

Pre-commit hooks configuration including:
- Hook repositories
- Hook IDs and versions
- Arguments and exclusions

## Code Style Guidelines

### General Principles

1. **Consistency**: Follow existing code patterns
2. **Readability**: Code is read more often than written
3. **Simplicity**: Prefer simple, clear solutions
4. **Documentation**: Document complex logic
5. **Testing**: Write tests for new features

### Python-Specific

1. **Line Length**: Maximum 100 characters
2. **Imports**: Organized (stdlib, third-party, local)
3. **Naming**:
   - Functions/variables: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
4. **Docstrings**: Use for public functions and classes
5. **Type Hints**: Optional but encouraged for function signatures

### Examples

```python
# Good
def calculate_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Calculate linear regression parameters.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        Dictionary with slope and intercept
    """
    slope = np.polyfit(x, y, 1)[0]
    intercept = np.polyfit(x, y, 1)[1]
    return {"slope": slope, "intercept": intercept}


# Bad
def calcReg(x,y):
    s=np.polyfit(x,y,1)[0]
    i=np.polyfit(x,y,1)[1]
    return {"slope":s,"intercept":i}
```

## Troubleshooting

### Pre-commit Hooks Fail

```bash
# View detailed error
pre-commit run --all-files --verbose

# Fix formatting issues
black *.py tests/*.py

# Fix flake8 issues manually or with autoflake
pip install autoflake
autoflake --in-place --remove-unused-variables --remove-all-unused-imports *.py
```

### MyPy Errors

MyPy errors are informational and don't block commits. To fix:

```bash
# Add type annotations
def my_function(x: int) -> str:
    return str(x)

# Use type: ignore for complex cases
result = complex_function()  # type: ignore
```

### Tests Fail

```bash
# Run specific test
pytest tests/test_data.py::test_function_name -v

# Run with more output
pytest tests/ -vv --tb=long

# Run with pdb debugger
pytest tests/ --pdb
```

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Python Style Guide (PEP 8)](https://pep8.org/)

## Deployment

### Streamlit Cloud Deployment

This application is configured for easy deployment to Streamlit Cloud.

#### Quick Deployment

1. Push your changes to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set `app.py` as the main file
5. Deploy!

#### Configuration Files

- **`.streamlit/config.toml`**: Streamlit configuration (theme, server settings)
- **`requirements.txt`**: Python dependencies for deployment
- **No secrets required**: All data is simulated/offline

#### Local Testing Before Deployment

```bash
# Test the app locally exactly as it will run in production
streamlit run app.py

# Open browser to http://localhost:8501
# Test all features to ensure they work correctly
```

#### Deployment Checklist

Before deploying, verify:

- [ ] All tests pass: `pytest tests/`
- [ ] Code is formatted: `black --check *.py tests/*.py`
- [ ] No linting errors: `flake8 *.py tests/*.py`
- [ ] App runs locally: `streamlit run app.py`
- [ ] All features work without errors
- [ ] Performance is acceptable (check with different dataset sizes)

#### Post-Deployment

After deploying to Streamlit Cloud:

1. **Test thoroughly**: Go through all tabs and features
2. **Check logs**: Monitor Streamlit Cloud logs for errors
3. **Verify performance**: Ensure acceptable load times
4. **Test on mobile**: Check mobile responsiveness
5. **Update README**: Add live demo URL

#### Detailed Deployment Guide

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment instructions, including:

- Step-by-step deployment process
- Configuration options
- Troubleshooting guide
- Performance optimization tips
- Monitoring and maintenance
- Custom domain setup

#### Automatic Redeployment

Streamlit Cloud automatically redeploys when you push to your configured branch:

```bash
# Make changes
git add .
git commit -m "Update feature X"
git push origin main

# Streamlit Cloud detects the push and redeploys automatically
# Redeployment typically takes 2-5 minutes
```

#### Environment Differences

**Local vs. Cloud:**

| Aspect | Local | Streamlit Cloud |
|--------|-------|-----------------|
| Python Version | Your local version | 3.9-3.12 (configurable) |
| Resources | Your machine | Shared (free tier) |
| URL | localhost:8501 | your-app.streamlit.app |
| HTTPS | No | Yes |
| Authentication | None | Optional (Pro tier) |

**Note**: The app is designed to work identically in both environments.

#### Deployment Resources

- **Streamlit Cloud Docs**: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md) in this repository

