# Configuration Files Documentation

This directory contains all configuration files for development tools, code quality checks, testing, and build processes for the Linear Regression Guide project.

## 📁 Configuration Files Overview

| File | Purpose | Tools |
|------|---------|-------|
| [`.flake8`](.flake8) | Python code linting and style checking | Flake8 |
| [`pyproject.toml`](pyproject.toml) | Modern Python project configuration | Multiple tools |
| [`mypy.ini`](mypy.ini) | Static type checking configuration | MyPy |
| [`.pre-commit-config.yaml`](.pre-commit-config.yaml) | Git pre-commit hooks | Pre-commit |
| [`pytest.ini`](pytest.ini) | Test framework configuration | Pytest |
| [`.coveragerc`](.coveragerc) | Code coverage configuration | Coverage.py |

## 🔧 Individual Configuration Details

### `.flake8` - Code Linting Configuration

**Purpose**: Configures Flake8 for comprehensive Python code linting and style checking.

**Key Settings**:
```ini
max-line-length = 120
max-complexity = 15
select = E,W,F,C,S,T
extend-ignore = E203,W503,E501,F541,S101,T201,T203
```

**Features**:
- **Line length**: 120 characters (modern standard)
- **Complexity**: Maximum 15 (maintainable functions)
- **Error codes**: pycodestyle (E,W), pyflakes (F), complexity (C), security (S), print statements (T)
- **Black compatibility**: Ignores rules that conflict with Black formatter
- **Per-file ignores**: Test files and scripts have relaxed rules

**Usage**:
```bash
flake8 src/ tests/  # Lint source and test files
```

### `pyproject.toml` - Modern Python Configuration

**Purpose**: Unified configuration file for multiple Python development tools.

**Sections**:
- **`[build-system]`**: Build backend configuration
- **`[project]`**: Project metadata and dependencies
- **`[tool.black]`**: Code formatting (Black)
- **`[tool.isort]`**: Import sorting
- **`[tool.bandit]`**: Security scanning
- **`[tool.coverage.*]`**: Code coverage
- **`[tool.pytest.ini_options]`**: Test configuration
- **`[tool.ruff]`**: Fast Python linter (alternative to flake8)
- **`[tool.commitizen]`**: Conventional commit messages

**Key Features**:
- **PEP 621 compliance**: Modern project metadata
- **Tool integration**: Single file configures multiple tools
- **Black + isort**: Consistent code formatting
- **Security scanning**: Automated vulnerability checks
- **Coverage configuration**: Detailed coverage reporting

### `mypy.ini` - Type Checking Configuration

**Purpose**: Configures MyPy for static type checking of Python code.

**Key Settings**:
```ini
python_version = 3.9
strict_equality = True
warn_return_any = True
no_implicit_optional = True
```

**Module-Specific Configuration**:
- **Streamlit modules**: Fully ignored (complex typing)
- **Data science libs**: numpy, pandas, plotly, statsmodels ignored
- **Test files**: More lenient type checking
- **Core modules**: Strict type checking enabled

**Features**:
- **Balanced strictness**: Practical approach for data science code
- **Library compatibility**: Handles complex third-party typing
- **File-specific rules**: Different strictness per module type
- **Performance optimized**: Caching and incremental checking

### `.pre-commit-config.yaml` - Git Hooks Configuration

**Purpose**: Defines pre-commit hooks that run automatically before commits.

**Hook Categories**:

#### File Processing
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Fix line endings
- **mixed-line-ending**: Ensure consistent line endings

#### Code Quality
- **black**: Format Python code
- **isort**: Sort Python imports
- **flake8**: Lint Python code
- **mypy**: Type check Python code

#### Security
- **python-safety-dependencies-check**: Check for vulnerable dependencies

#### Documentation
- **codespell**: Check for common misspellings
- **prettier**: Format documentation files

**Features**:
- **CI integration**: Automatic fixes and updates
- **Comprehensive checks**: From formatting to security
- **Performance**: Parallel execution where possible

### `pytest.ini` - Test Framework Configuration

**Purpose**: Configures pytest for comprehensive testing.

**Key Features**:
- **Test discovery**: Multiple patterns for flexible test organization
- **Markers**: 20+ test markers for categorization
- **Coverage integration**: Built-in coverage reporting
- **Performance**: Timeout and duration tracking
- **Parallel execution**: Support for pytest-xdist

**Test Markers**:
```ini
# Test types
unit, integration, end_to_end, smoke

# Frameworks
streamlit, playwright, selenium

# Characteristics
slow, fast, performance, regression

# Focus areas
data, plots, ui, api, config

# Special types
property, visual, accessibility

# Environments
ci, local, staging, production

# Data types
synthetic, real, mock, integration_data
```

### `.coveragerc` - Code Coverage Configuration

**Purpose**: Configures coverage.py for accurate code coverage measurement.

**Key Features**:
- **Source specification**: Only track coverage in `src/` directory
- **Exclusion patterns**: Comprehensive list of files to ignore
- **Line exclusions**: Framework-specific code that shouldn't be tested
- **Multiple report formats**: HTML, XML, JSON, terminal
- **Branch coverage**: Track both line and branch coverage

## 🚀 Usage Examples

### Code Quality Check
```bash
# Use flake8 directly
flake8 src/ --config=config/.flake8

# Use pre-commit (recommended)
pre-commit run flake8 --all-files
```

### Type Checking
```bash
# Use mypy directly
mypy src/ --config-file=config/mypy.ini

# Use pre-commit
pre-commit run mypy --all-files
```

### Testing
```bash
# Use pytest directly
pytest --config-file=config/pytest.ini

# Run with coverage
pytest --cov-config=config/.coveragerc --cov=src
```

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

## ⚙️ Configuration Philosophy

### Development-First Approach
- **Practical over perfect**: Balance between code quality and development speed
- **Incremental adoption**: Start with reasonable rules, tighten over time
- **Tool integration**: Single configuration files where possible

### Code Quality Balance
- **Black for formatting**: Opinionated but consistent
- **Flake8 for linting**: Comprehensive but not overwhelming
- **MyPy for typing**: Practical balance for data science code
- **Pre-commit for automation**: Catch issues before they reach CI

### Testing Strategy
- **Comprehensive markers**: Flexible test categorization
- **Coverage integration**: Built-in reporting and thresholds
- **Performance awareness**: Timeout and duration tracking
- **CI/CD ready**: JUnit XML and other CI integrations

## 🔧 Customization

### Adding New Tools
1. Add configuration section to `pyproject.toml`
2. Update pre-commit hooks if applicable
3. Add documentation to this README
4. Test integration with existing workflow

### Modifying Existing Rules
1. Update the relevant configuration file
2. Test the changes locally
3. Run affected pre-commit hooks
4. Update documentation if behavior changes

### Environment-Specific Configuration
- Use environment variables for different settings
- Create separate config files for different environments
- Document environment-specific requirements

## 📋 Maintenance

### Regular Updates
- **Tool versions**: Update to latest stable versions regularly
- **Rule evaluation**: Review and adjust rules based on codebase evolution
- **Performance**: Monitor and optimize pre-commit hook execution time

### Troubleshooting
- **Pre-commit slow**: Use `--all-files` only when necessary
- **False positives**: Adjust ignore rules rather than disabling tools
- **CI failures**: Ensure local pre-commit passes before pushing

---

**Configuration Version**: 2.0
**Last Updated**: January 2026
**Tools Covered**: Black, Flake8, MyPy, Pytest, Coverage, Pre-commit, Ruff, Bandit, Commitizen