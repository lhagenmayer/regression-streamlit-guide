#!/usr/bin/env python3
"""
Pre-commit-style check to validate modular separation in the Linear Regression Guide.

This script performs static analysis to ensure that:
- data.py only handles data generation
- statistics.py only performs statistical computations
- plots.py only creates visualizations
- content.py only provides metadata

Usage:
    python scripts/check_modular_separation.py
"""

import sys
import ast
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class ModuleSummary:
    imports: Set[str] = field(default_factory=set)
    function_calls: Set[str] = field(default_factory=set)
    function_defs: Set[str] = field(default_factory=set)


@dataclass
class ModuleRules:
    name: str
    path: Path
    forbidden_imports: Set[str] = field(default_factory=set)
    required_functions: Set[str] = field(default_factory=set)
    forbidden_function_prefixes: Set[str] = field(default_factory=set)


def analyze_module(file_path: Path) -> ModuleSummary:
    """Analyze a Python module to extract imports, definitions, and calls."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        imports: Set[str] = set()
        function_calls: Set[str] = set()
        function_defs: Set[str] = set()

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

        return ModuleSummary(
            imports=imports,
            function_calls=function_calls,
            function_defs=function_defs,
        )

    except Exception as e:
        print(f"❌ Failed to analyze {file_path}: {e}", file=sys.stderr)
        return ModuleSummary()


def check_modular_separation(project_root: Path) -> bool:
    """Check modular separation across all core modules."""
    src_path = project_root / "src"
    success = True

    module_rules: Dict[str, ModuleRules] = {
        "data": ModuleRules(
            name="data",
            path=src_path / "data.py",
            forbidden_imports={"statistics", "plots", "plotly", "matplotlib", "seaborn"},
            required_functions={
                "generate_multiple_regression_data",
                "generate_simple_regression_data",
                "generate_swiss_canton_regression_data",
                "generate_swiss_weather_regression_data",
                "generate_electronics_market_data",
                "fetch_bfs_data",
                "fetch_world_bank_data",
                "fetch_fred_data",
                "fetch_who_health_data",
                "fetch_eurostat_data",
                "fetch_swiss_weather_data",
                "get_swiss_canton_data",
                "get_available_swiss_datasets",
                "get_global_regression_datasets",
                "create_dummy_encoded_dataset",
                "safe_scalar",
            },
            forbidden_function_prefixes={"compute_", "fit_", "perform_", "calculate_", "create_plotly"},
        ),
        "statistics": ModuleRules(
            name="statistics",
            path=src_path / "statistics.py",
            forbidden_imports={"plots", "plotly", "matplotlib", "seaborn"},
            required_functions={
                "fit_ols_model",
                "fit_multiple_ols_model",
                "compute_regression_statistics",
                "compute_simple_regression_stats",
                "compute_multiple_regression_stats",
                "compute_residual_diagnostics",
                "perform_normality_tests",
                "perform_heteroskedasticity_tests",
                "perform_t_test",
                "calculate_variance_inflation_factors",
                "calculate_sensitivity_analysis",
                "get_model_coefficients",
                "get_model_summary_stats",
                "get_model_diagnostics",
                "calculate_prediction_interval",
                "get_data_ranges",
                "calculate_basic_stats",
                "create_design_matrix",
                "create_model_summary_dataframe",
                "format_statistical_value",
            },
            forbidden_function_prefixes={"calculate_residual_sizes", "standardize_residuals", "create_plotly"},
        ),
        "plots": ModuleRules(
            name="plots",
            path=src_path / "plots.py",
            required_functions={"calculate_residual_sizes", "standardize_residuals"},
        ),
        "content": ModuleRules(
            name="content",
            path=src_path / "content.py",
            required_functions={
                "get_simple_regression_content",
                "get_multiple_regression_formulas",
                "get_multiple_regression_descriptions",
                "get_dataset_info",
            },
            forbidden_function_prefixes={
                "generate_",
                "fetch_",
                "compute_",
                "fit_",
                "perform_",
                "calculate_",
                "create_plotly",
            },
        ),
    }

    module_info: Dict[str, ModuleSummary] = {}
    for name, rules in module_rules.items():
        if rules.path.exists():
            module_info[name] = analyze_module(rules.path)
        else:
            print(f"⚠️  Module {name}.py not found")
            module_info[name] = ModuleSummary()

    def check_forbidden_imports(rules: ModuleRules, summary: ModuleSummary) -> bool:
        bad_imports = summary.imports.intersection(rules.forbidden_imports)
        if bad_imports:
            print(f"❌ {rules.name}.py imports forbidden modules: {sorted(bad_imports)}")
            return False
        print(f"✅ {rules.name}.py correctly avoids forbidden imports")
        return True

    def check_required_functions(rules: ModuleRules, summary: ModuleSummary) -> bool:
        missing = rules.required_functions - summary.function_defs
        if missing:
            print(f"❌ {rules.name}.py MISSING required functions: {sorted(missing)}")
            return False
        print(f"✅ {rules.name}.py contains required functions")
        return True

    def check_forbidden_functions(rules: ModuleRules, summary: ModuleSummary) -> bool:
        forbidden_present = {
            func
            for func in summary.function_defs
            for prefix in rules.forbidden_function_prefixes
            if func.startswith(prefix)
        }
        if forbidden_present:
            print(f"❌ {rules.name}.py CONTAINS FORBIDDEN functions: {sorted(forbidden_present)}")
            return False
        if rules.forbidden_function_prefixes:
            print(f"✅ {rules.name}.py avoids forbidden function families")
        return True

    print("🔍 Checking data.py modular separation...")
    data_rules = module_rules["data"]
    data_info = module_info["data"]
    success &= check_forbidden_imports(data_rules, data_info)
    success &= check_required_functions(data_rules, data_info)
    success &= check_forbidden_functions(data_rules, data_info)

    print("🔍 Checking statistics.py modular separation...")
    stats_rules = module_rules["statistics"]
    stats_info = module_info["statistics"]
    success &= check_forbidden_imports(stats_rules, stats_info)
    success &= check_required_functions(stats_rules, stats_info)
    success &= check_forbidden_functions(stats_rules, stats_info)

    print("🔍 Checking plots.py modular separation...")
    plots_rules = module_rules["plots"]
    plots_info = module_info["plots"]

    data_gen_calls = [
        call
        for call in plots_info.function_calls
        if call.startswith("generate_") or call.startswith("fetch_")
    ]
    if data_gen_calls:
        print(f"❌ plots.py calls data generation functions: {sorted(data_gen_calls)}")
        success = False
    else:
        print("✅ plots.py correctly avoids data generation calls")

    success &= check_required_functions(plots_rules, plots_info)

    print("🔍 Checking content.py modular separation...")
    content_rules = module_rules["content"]
    content_info = module_info["content"]
    success &= check_required_functions(content_rules, content_info)
    success &= check_forbidden_functions(content_rules, content_info)

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