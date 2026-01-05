#!/usr/bin/env python3
"""
Pre-commit hook to validate modular separation in the Linear Regression Guide.

This script performs static analysis to ensure that:
- data.py only handles data generation
- statistics.py only performs statistical computations
- plots.py only creates visualizations
- content.py only provides metadata

Usage:
    python scripts/check_modular_separation.py [--fix]
"""

import sys
import ast
from pathlib import Path
from typing import Set, Dict


def analyze_module(file_path: Path) -> Dict[str, Set[str]]:
    """Analyze a Python module to extract imports and function calls."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        imports = set()
        function_calls = set()
        function_defs = set()

        for node in ast.walk(tree):
            # Extract imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])

            # Extract function definitions
            elif isinstance(node, ast.FunctionDef):
                function_defs.add(node.name)

            # Extract function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    function_calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like module.function
                    if isinstance(node.func.value, ast.Name):
                        function_calls.add(f"{node.func.value.id}.{node.func.attr}")

        return {
            "imports": imports,
            "function_calls": function_calls,
            "function_defs": function_defs
        }

    except Exception as e:
        print(f"❌ Failed to analyze {file_path}: {e}", file=sys.stderr)
        return {"imports": set(), "function_calls": set(), "function_defs": set()}


def check_modular_separation(project_root: Path) -> bool:
    """Check modular separation across all modules."""
    src_path = project_root / "src"
    success = True

    modules = {
        "data": src_path / "data.py",
        "statistics": src_path / "statistics.py",
        "plots": src_path / "plots.py",
        "content": src_path / "content.py"
    }

    module_info = {}
    for name, path in modules.items():
        if path.exists():
            module_info[name] = analyze_module(path)
        else:
            print(f"⚠️  Module {name}.py not found")
            continue

    # Check data.py separation
    print("🔍 Checking data.py modular separation...")
    data_info = module_info.get("data", {"imports": set(), "function_calls": set()})

    # data.py should NOT import statistics or plots
    forbidden_imports = {"statistics", "plots", "plotly", "matplotlib", "seaborn"}
    bad_imports = data_info["imports"].intersection(forbidden_imports)
    if bad_imports:
        print(f"❌ data.py imports forbidden modules: {bad_imports}")
        success = False
    else:
        print("✅ data.py correctly avoids forbidden imports")

    # Check statistics.py separation
    print("🔍 Checking statistics.py modular separation...")
    stats_info = module_info.get("statistics", {"imports": set(), "function_calls": set()})

    forbidden_imports = {"plots", "plotly", "matplotlib", "seaborn"}
    bad_imports = stats_info["imports"].intersection(forbidden_imports)
    if bad_imports:
        print(f"❌ statistics.py imports plotting modules: {bad_imports}")
        success = False
    else:
        print("✅ statistics.py correctly avoids plotting imports")

        # Check that statistics.py has EXACTLY the right functions
    stats_functions = stats_info.get("function_defs", set())

    # Required statistics functions
    required_stats = {
        "fit_ols_model", "fit_multiple_ols_model", "compute_regression_statistics",
        "compute_simple_regression_stats", "compute_multiple_regression_stats",
        "compute_residual_diagnostics", "perform_normality_tests", "perform_heteroskedasticity_tests",
        "perform_t_test", "calculate_variance_inflation_factors", "calculate_sensitivity_analysis",
        "get_model_coefficients", "get_model_summary_stats", "get_model_diagnostics",
        "calculate_prediction_interval", "get_data_ranges", "calculate_basic_stats",
        "create_design_matrix", "create_model_summary_dataframe", "format_statistical_value"
    }

    # Forbidden in statistics
    forbidden_in_stats = {"calculate_residual_sizes", "standardize_residuals", "create_plotly"}

    missing_required = required_stats - stats_functions
    forbidden_present = set()
    for func in stats_functions:
        for forbidden in forbidden_in_stats:
            if func.startswith(forbidden):
                forbidden_present.add(func)

    if missing_required:
        print(f"❌ statistics.py MISSING required functions: {missing_required}")
        success = False

    if forbidden_present:
        print(f"❌ statistics.py CONTAINS FORBIDDEN functions: {forbidden_present}")
        success = False

    if not missing_required and not forbidden_present:
        print("✅ statistics.py has exact correct function set")

    # Check plots.py separation
    print("🔍 Checking plots.py modular separation...")
    plots_info = module_info.get("plots", {"function_calls": set()})

    data_gen_calls = [call for call in plots_info["function_calls"]
                     if call.startswith("generate_") or call.startswith("fetch_")]
    if data_gen_calls:
        print(f"❌ plots.py calls data generation functions: {data_gen_calls}")
        success = False
    else:
        print("✅ plots.py correctly avoids data generation calls")

    # Check that visualization prep functions are in plots.py
    plots_functions = plots_info.get("function_defs", set())
    viz_prep_functions = {"calculate_residual_sizes", "standardize_residuals"}
    missing_viz_prep = viz_prep_functions - plots_functions
    if missing_viz_prep:
        print(f"❌ plots.py missing visualization prep functions: {missing_viz_prep}")
        success = False
    else:
        print("✅ plots.py contains visualization preparation functions")

    # Check that data.py has EXACTLY the right functions
    data_functions = data_info.get("function_defs", set())

    # Required data functions
    required_data = {
        "generate_multiple_regression_data", "generate_simple_regression_data",
        "generate_swiss_canton_regression_data", "generate_swiss_weather_regression_data",
        "generate_electronics_market_data", "fetch_bfs_data", "fetch_world_bank_data",
        "fetch_fred_data", "fetch_who_health_data", "fetch_eurostat_data", "fetch_swiss_weather_data",
        "get_swiss_canton_data", "get_available_swiss_datasets", "get_global_regression_datasets",
        "create_dummy_encoded_dataset", "safe_scalar"
    }

    # Forbidden in data
    forbidden_in_data = {"compute_", "fit_", "perform_", "calculate_", "create_plotly"}

    missing_required = required_data - data_functions
    forbidden_present = set()
    for func in data_functions:
        for forbidden in forbidden_in_data:
            if func.startswith(forbidden):
                forbidden_present.add(func)

    if missing_required:
        print(f"❌ data.py MISSING required functions: {missing_required}")
        success = False

    if forbidden_present:
        print(f"❌ data.py CONTAINS FORBIDDEN functions: {forbidden_present}")
        success = False

    if not missing_required and not forbidden_present:
        print("✅ data.py has exact correct function set")

    # Check content.py has EXACTLY the right functions
    content_info = module_info.get("content", {"function_defs": set()})
    content_functions = content_info.get("function_defs", set())

    # Required content functions
    required_content = {
        "get_simple_regression_content", "get_multiple_regression_formulas",
        "get_multiple_regression_descriptions", "get_dataset_info"
    }

    # Forbidden in content
    forbidden_in_content = {"generate_", "fetch_", "compute_", "fit_", "perform_", "calculate_", "create_plotly"}

    missing_required = required_content - content_functions
    forbidden_present = set()
    for func in content_functions:
        for forbidden in forbidden_in_content:
            if func.startswith(forbidden):
                forbidden_present.add(func)

    if missing_required:
        print(f"❌ content.py MISSING required functions: {missing_required}")
        success = False

    if forbidden_present:
        print(f"❌ content.py CONTAINS FORBIDDEN functions: {forbidden_present}")
        success = False

    if not missing_required and not forbidden_present:
        print("✅ content.py has exact correct function set")

    return success


def main():
    """Main entry point for the pre-commit hook."""
    project_root = Path(__file__).parent.parent

    print("🔧 Checking modular separation in Linear Regression Guide...")

    if check_modular_separation(project_root):
        print("🎉 All modular separation checks passed!")
        return 0
    else:
        print("❌ Modular separation violations found!")
        print("\nTo fix these issues:")
        print("1. data.py should only generate data (no statistics or plotting)")
        print("2. statistics.py should only compute statistics (no data generation or plotting)")
        print("3. plots.py should only create visualizations (no data generation or statistics)")
        print("4. content.py should only provide metadata (no data processing)")
        return 1


if __name__ == "__main__":
    sys.exit(main())