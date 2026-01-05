#!/usr/bin/env python3
"""
Architecture validation script for the Linear Regression Guide.

This script validates the modular architecture with STRICT rules.
It can be run during CI/CD or as a standalone validation.

Usage:
    python scripts/validate_architecture.py
"""

import sys
import ast
from pathlib import Path
from typing import Set, Dict, List


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


def validate_architecture() -> bool:
    """Validate the modular architecture with STRICT rules."""
    print("🏗️  STRICTLY Validating Linear Regression Guide Architecture...")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    success = True

    modules = {
        "data": src_path / "data.py",
        "statistics": src_path / "statistics.py",
        "plots": src_path / "plots.py",
        "content": src_path / "content.py",
        "app": src_path / "app.py"
    }

    module_info = {}
    for name, path in modules.items():
        if path.exists():
            print(f"📄 Analyzing {name}.py...")
            module_info[name] = analyze_module(path)
        else:
            print(f"⚠️  Module {name}.py not found at {path}")
            continue

    # STRICT EXACT FUNCTION MEMBERSHIP RULES
    expected_functions = {
        "data": {
            "allowed_only": {
                "generate_multiple_regression_data", "generate_simple_regression_data",
                "generate_swiss_canton_regression_data", "generate_swiss_weather_regression_data",
                "generate_electronics_market_data", "fetch_bfs_data", "fetch_world_bank_data",
                "fetch_fred_data", "fetch_who_health_data", "fetch_eurostat_data", "fetch_swiss_weather_data",
                "get_swiss_canton_data", "get_available_swiss_datasets", "get_global_regression_datasets",
                "create_dummy_encoded_dataset", "safe_scalar"
            },
            "forbidden_any": {"compute_", "fit_", "perform_", "calculate_", "create_plotly", "get_signif"}
        },
        "statistics": {
            "allowed_only": {
                "fit_ols_model", "fit_multiple_ols_model", "compute_regression_statistics",
                "compute_simple_regression_stats", "compute_multiple_regression_stats",
                "compute_residual_diagnostics", "perform_normality_tests", "perform_heteroskedasticity_tests",
                "perform_t_test", "calculate_variance_inflation_factors", "calculate_sensitivity_analysis",
                "get_model_coefficients", "get_model_summary_stats", "get_model_diagnostics",
                "calculate_prediction_interval", "get_data_ranges", "calculate_basic_stats",
                "create_design_matrix", "create_model_summary_dataframe", "format_statistical_value"
            },
            "forbidden_any": {"generate_", "fetch_", "create_plotly", "plt.", "calculate_residual_sizes", "standardize_residuals"}
        },
        "plots": {
            "allowed_only": {
                "create_plotly_scatter", "create_plotly_scatter_with_line", "create_plotly_3d_scatter",
                "create_plotly_3d_surface", "create_plotly_residual_plot", "create_plotly_bar",
                "create_plotly_distribution", "create_r_output_display", "create_r_output_figure",
                "get_signif_stars", "get_signif_color", "calculate_residual_sizes", "standardize_residuals",
                "create_regression_mesh", "get_3d_layout_config", "create_zero_plane"
            },
            "forbidden_any": {"generate_", "fetch_", "compute_", "fit_", "perform_"}
        },
        "content": {
            "allowed_only": {
                "get_simple_regression_content", "get_multiple_regression_formulas",
                "get_multiple_regression_descriptions", "get_dataset_info"
            },
            "forbidden_any": {"generate_", "fetch_", "compute_", "fit_", "perform_", "calculate_", "create_plotly"}
        }
    }

    print("\n" + "="*70)
    print("🔍 VALIDATING EXACT FUNCTION MEMBERSHIP")
    print("="*70)

    for module_name, rules in expected_functions.items():
        print(f"\n🔍 Checking {module_name}.py function membership")

        if module_name not in module_info:
            print(f"⚠️  Skipping {module_name}.py (not found)")
            continue

        info = module_info[module_name]
        module_ok = True

        # Check for EXACTLY ALLOWED functions only
        if "allowed_only" in rules:
            allowed_functions = rules["allowed_only"]

            # Find functions that exist but are NOT in allowed list
            unexpected_functions = info["function_defs"] - allowed_functions
            if unexpected_functions:
                print(f"❌ UNEXPECTED functions in {module_name}.py: {unexpected_functions}")
                print(f"   Only these functions are allowed: {sorted(allowed_functions)}")
                module_ok = False

            # Find missing required functions
            missing_functions = allowed_functions - info["function_defs"]
            if missing_functions:
                print(f"❌ MISSING required functions in {module_name}.py: {missing_functions}")
                module_ok = False

            if not unexpected_functions and not missing_functions:
                print(f"✅ {module_name}.py has exactly the correct {len(allowed_functions)} functions")

        # Check for ANY forbidden function patterns
        if "forbidden_any" in rules:
            forbidden_patterns = rules["forbidden_any"]
            forbidden_found = set()

            for func in info["function_defs"]:
                for pattern in forbidden_patterns:
                    if func.startswith(pattern):
                        forbidden_found.add(func)

            if forbidden_found:
                print(f"❌ FORBIDDEN functions found in {module_name}.py: {forbidden_found}")
                module_ok = False
            else:
                print("✅ No forbidden function patterns found")

        if not module_ok:
            success = False

    # STRICT IMPORT VALIDATION
    print("\n" + "="*70)
    print("🔗 VALIDATING STRICT IMPORT RULES")
    print("="*70)

    strict_import_rules = {
        "data": {"allowed": {"typing", "time", "numpy", "pandas", "streamlit", "statsmodels", "config", "logger"},
                "forbidden": {"statistics", "plots", "plotly", "matplotlib", "seaborn"}},
        "statistics": {"allowed": {"typing", "numpy", "pandas", "scipy", "statsmodels", "streamlit", "logger"},
                      "forbidden": {"plots", "plotly", "matplotlib", "seaborn"}},
        "plots": {"allowed": {"typing", "numpy", "pandas", "plotly", "streamlit", "logger", "data"},
                 "forbidden": {"statistics", "statsmodels", "scipy"}},
        "content": {"allowed": {"typing"},
                   "forbidden": {"data", "statistics", "plots", "numpy", "pandas", "plotly", "matplotlib"}}
    }

    for module_name, rules in strict_import_rules.items():
        print(f"\n🔍 Checking {module_name}.py imports")

        if module_name not in module_info:
            continue

        info = module_info[module_name]
        module_ok = True

        # Check forbidden imports
        if "forbidden" in rules:
            bad_imports = info["imports"].intersection(rules["forbidden"])
            if bad_imports:
                print(f"❌ FORBIDDEN imports in {module_name}.py: {bad_imports}")
                module_ok = False
            else:
                print("✅ No forbidden imports")

        # Check for unexpected imports (not in allowed list)
        if "allowed" in rules:
            # Add common stdlib imports that are always allowed
            allowed_with_stdlib = rules["allowed"] | {"os", "sys", "pathlib", "functools", "collections", "itertools"}
            unexpected_imports = info["imports"] - allowed_with_stdlib
            if unexpected_imports:
                print(f"❌ UNEXPECTED imports in {module_name}.py: {unexpected_imports}")
                print(f"   Only these are allowed: {sorted(allowed_with_stdlib)}")
                module_ok = False

        if not module_ok:
            success = False

    # STRICT FUNCTION CALL VALIDATION
    print("\n" + "="*70)
    print("⚡ VALIDATING STRICT FUNCTION CALLS")
    print("="*70)

    strict_call_rules = {
        "data": {"forbidden": ["create_plotly", "plt.", "plotly.", "matplotlib.", "compute_", "fit_", "perform_"]},
        "statistics": {"forbidden": ["create_plotly", "plt.", "plotly.", "matplotlib.", "generate_", "fetch_"]},
        "plots": {"forbidden": ["generate_", "fetch_", "compute_", "fit_", "perform_", "calculate_"]},
        "content": {"forbidden": ["generate_", "fetch_", "compute_", "fit_", "perform_", "calculate_", "create_plotly"]},
        "app": {"allowed": ["generate_", "fetch_", "compute_", "fit_", "perform_", "calculate_", "create_plotly"]}  # app can call everything
    }

    for module_name, rules in strict_call_rules.items():
        print(f"\n🔍 Checking {module_name}.py function calls")

        if module_name not in module_info:
            continue

        info = module_info[module_name]
        module_ok = True

        if "forbidden" in rules:
            bad_calls = set()
            for call in info["function_calls"]:
                for forbidden in rules["forbidden"]:
                    if forbidden in call:
                        bad_calls.add(call)

            if bad_calls:
                print(f"❌ FORBIDDEN function calls in {module_name}.py: {bad_calls}")
                module_ok = False
            else:
                print("✅ No forbidden function calls")

        if not module_ok:
            success = False

    # VALIDATE DATA FLOW INTEGRITY
    print("\n" + "="*70)
    print("🔄 VALIDATING DATA FLOW INTEGRITY")
    print("="*70)

    # Check that app.py orchestrates correctly
    app_calls = module_info.get("app", {}).get("function_calls", set())

    # app.py should call functions from all modules
    calls_data_functions = any(call.startswith(("generate_", "fetch_", "get_")) for call in app_calls)
    calls_stats_functions = any(call.startswith(("compute_", "fit_", "perform_")) for call in app_calls)
    calls_plot_functions = any(call.startswith(("create_plotly", "calculate_residual")) for call in app_calls)

    data_flow_checks = [
        ("app.py calls data functions", calls_data_functions),
        ("app.py calls statistics functions", calls_stats_functions),
        ("app.py calls plotting functions", calls_plot_functions)
    ]

    for check_name, passed in data_flow_checks:
        if passed:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            success = False

    print("\n" + "="*70)
    if success:
        print("🎉 STRICT Architecture validation PASSED!")
        print("✅ All modules maintain PERFECT separation of concerns")
        print("✅ Exact function membership validated")
        print("✅ Import restrictions enforced")
        print("✅ Data flow integrity confirmed")
    else:
        print("❌ STRICT Architecture validation FAILED!")
        print("🔧 Fix the architectural violations above")
        print("💡 This validation is STRICT - only exact matches are allowed")

    return success


def main():
    """Main entry point."""
    success = validate_architecture()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()