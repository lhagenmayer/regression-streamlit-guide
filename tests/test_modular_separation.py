"""
Modular separation validation tests for Linear Regression Guide.

This module ensures that the data flow architecture is maintained:
- data.py: Only data generation, no statistics or plotting
- statistics.py: Only statistical computations, no data generation or plotting
- plots.py: Only visualization, no data generation or statistical analysis
- content.py: Only metadata and content, no data processing

Tests use static analysis to validate import relationships and function calls.
"""

import pytest
import ast
import importlib.util
import sys
from pathlib import Path
from typing import Set, Dict, List


class TestModularSeparation:
    """Test suite for validating modular separation in the codebase."""

    @pytest.fixture(scope="class")
    def module_structure(self):
        """Analyze the module structure and imports."""
        project_root = Path(__file__).parent.parent
        src_path = project_root / "src"

        modules = {
            "data": src_path / "data.py",
            "statistics": src_path / "statistics.py",
            "plots": src_path / "plots.py",
            "content": src_path / "content.py",
            "app": src_path / "app.py",
            "config": src_path / "config.py"
        }

        structure = {}
        for name, path in modules.items():
            if path.exists():
                structure[name] = self._analyze_module(path)
            else:
                structure[name] = {"imports": set(), "function_calls": set(), "function_defs": set()}

        return structure

    def _analyze_module(self, file_path: Path) -> Dict[str, Set[str]]:
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
            pytest.fail(f"Failed to analyze {file_path}: {e}")

    @pytest.mark.unit
    def test_data_module_separation(self, module_structure):
        """Test that data.py only handles data generation, no statistics or plotting."""
        data_info = module_structure["data"]

        # data.py should NOT import statistics or plots modules
        forbidden_imports = {"statistics", "plots", "plotly", "matplotlib", "seaborn"}
        overlapping_imports = data_info["imports"].intersection(forbidden_imports)

        assert not overlapping_imports, (
            f"data.py imports forbidden modules: {overlapping_imports}. "
            "data.py should only handle data generation, not statistics or plotting."
        )

        # data.py should NOT call plotting functions
        forbidden_calls = {"create_plotly", "plt.", "plotly.", "matplotlib."}
        plotting_calls = [call for call in data_info["function_calls"]
                         if any(forbidden in call for forbidden in forbidden_calls)]

        assert not plotting_calls, (
            f"data.py calls plotting functions: {plotting_calls}. "
            "data.py should only generate data, not create visualizations."
        )

        # data.py SHOULD contain data generation functions
        data_functions = {"generate_", "fetch_", "get_available_", "get_global_"}
        has_data_functions = any(any(func.startswith(prefix) for func in data_info["function_defs"])
                               for prefix in data_functions)

        assert has_data_functions, "data.py should contain data generation functions"

    @pytest.mark.unit
    def test_statistics_module_separation(self, module_structure):
        """Test that statistics.py only handles statistical computations."""
        stats_info = module_structure["statistics"]

        # statistics.py should NOT import plots or plotly
        forbidden_imports = {"plots", "plotly", "matplotlib", "seaborn"}
        plotting_imports = stats_info["imports"].intersection(forbidden_imports)

        assert not plotting_imports, (
            f"statistics.py imports plotting modules: {plotting_imports}. "
            "statistics.py should only perform statistical computations."
        )

        # statistics.py should NOT call plotting functions
        forbidden_calls = {"create_plotly", "plt.", "plotly.", "matplotlib."}
        plotting_calls = [call for call in stats_info["function_calls"]
                         if any(forbidden in call for forbidden in forbidden_calls)]

        assert not plotting_calls, (
            f"statistics.py calls plotting functions: {plotting_calls}. "
            "statistics.py should only compute statistics, not create visualizations."
        )

        # statistics.py SHOULD contain statistical computation functions
        stat_functions = {"compute_", "fit_", "perform_", "calculate_"}
        has_stat_functions = any(any(func.startswith(prefix) for func in stats_info["function_defs"])
                               for prefix in stat_functions)

        assert has_stat_functions, "statistics.py should contain statistical computation functions"

    @pytest.mark.unit
    def test_plots_module_separation(self, module_structure):
        """Test that plots.py only handles visualization, no data generation or statistics."""
        plots_info = module_structure["plots"]

        # plots.py should NOT import data generation modules for data creation
        # (but may import for data access)
        data_generation_calls = [call for call in plots_info["function_calls"]
                               if call.startswith("generate_") or call.startswith("fetch_")]

        assert not data_generation_calls, (
            f"plots.py calls data generation functions: {data_generation_calls}. "
            "plots.py should only visualize existing data, not generate new data."
        )

        # plots.py should NOT perform statistical computations
        stat_calls = [call for call in plots_info["function_calls"]
                     if call.startswith("compute_") or call.startswith("fit_") or
                        call.startswith("perform_") or call.startswith("calculate_")]

        assert not stat_calls, (
            f"plots.py performs statistical computations: {stat_calls}. "
            "plots.py should only create visualizations from pre-computed statistics."
        )

        # plots.py SHOULD contain plotting functions
        plot_functions = {"create_plotly", "create_r_output", "get_signif"}
        has_plot_functions = any(any(func.startswith(prefix) for func in plots_info["function_defs"])
                               for prefix in plot_functions)

        assert has_plot_functions, "plots.py should contain plotting functions"

    @pytest.mark.unit
    def test_content_module_separation(self, module_structure):
        """Test that content.py only provides metadata and content."""
        content_info = module_structure["content"]

        # content.py should NOT generate data
        data_generation_calls = [call for call in content_info["function_calls"]
                               if call.startswith("generate_") or call.startswith("fetch_")]

        assert not data_generation_calls, (
            f"content.py calls data generation functions: {data_generation_calls}. "
            "content.py should only provide metadata and descriptions."
        )

        # content.py should NOT perform statistical computations
        stat_calls = [call for call in content_info["function_calls"]
                     if call.startswith("compute_") or call.startswith("fit_") or
                        call.startswith("perform_") or call.startswith("calculate_")]

        assert not stat_calls, (
            f"content.py performs statistical computations: {stat_calls}. "
            "content.py should only provide content and metadata."
        )

        # content.py should NOT create plots
        plotting_calls = [call for call in content_info["function_calls"]
                         if call.startswith("create_plotly") or call.startswith("plt.")]

        assert not plotting_calls, (
            f"content.py calls plotting functions: {plotting_calls}. "
            "content.py should only provide content, not visualizations."
        )

        # content.py SHOULD contain content functions
        content_functions = {"get_", "formulas", "descriptions", "content"}
        has_content_functions = any(any(func.startswith(prefix) for func in content_info["function_defs"])
                                  for prefix in content_functions)

        assert has_content_functions, "content.py should contain content/metadata functions"

    @pytest.mark.unit
    def test_data_flow_integrity(self, module_structure):
        """Test that data flows correctly between modules without forbidden cross-dependencies."""

        # Test that data.py functions are used by statistics.py and app.py, but not by plots.py directly
        data_functions = {f"data.{func}" for func in module_structure["data"]["function_defs"]
                         if func.startswith(("generate_", "fetch_", "get_"))}

        stats_calls = module_structure["statistics"]["function_calls"]
        plots_calls = module_structure["plots"]["function_calls"]
        content_calls = module_structure["content"]["function_calls"]

        # Statistics should use data functions
        stats_uses_data = bool(data_functions.intersection(stats_calls))
        # Plots should NOT directly use data generation functions
        plots_avoids_data = not bool(data_functions.intersection(plots_calls))
        # Content should NOT use data generation functions
        content_avoids_data = not bool(data_functions.intersection(content_calls))

        assert stats_uses_data, "statistics.py should use data generation functions"
        assert plots_avoids_data, "plots.py should not directly use data generation functions"
        assert content_avoids_data, "content.py should not use data generation functions"

    @pytest.mark.unit
    def test_statistics_to_plots_flow(self, module_structure):
        """Test that statistics results flow to plots correctly."""

        # Statistics functions that should be used by plots
        stat_functions = {f"statistics.{func}" for func in module_structure["statistics"]["function_defs"]
                         if func.startswith(("compute_", "fit_", "perform_", "calculate_"))}

        plots_calls = module_structure["plots"]["function_calls"]
        app_calls = module_structure["app"]["function_calls"]

        # App should use statistics functions
        app_uses_stats = bool(stat_functions.intersection(app_calls))

        assert app_uses_stats, "app.py should use statistics functions"

    @pytest.mark.unit
    def test_circular_dependency_prevention(self, module_structure):
        """Test that there are no circular dependencies in the data flow."""

        # Check for forbidden import patterns
        forbidden_patterns = [
            ("data", "statistics"),  # data should not import statistics
            ("data", "plots"),       # data should not import plots
            ("statistics", "plots"), # statistics should not import plots
            ("plots", "statistics"), # plots should not import statistics for computation
        ]

        for from_module, to_module in forbidden_patterns:
            from_imports = module_structure[from_module]["imports"]
            if to_module in from_imports:
                pytest.fail(f"Circular dependency detected: {from_module}.py imports {to_module}.py")

    @pytest.mark.unit
    def test_module_purity_runtime(self):
        """Runtime test to ensure modules can be imported independently."""

        # Test that each module can be imported without pulling in forbidden dependencies
        modules_to_test = [
            ("data", ["statistics", "plots"]),
            ("statistics", ["plots"]),
            ("plots", []),
            ("content", ["data", "statistics", "plots"])
        ]

        for module_name, forbidden_modules in modules_to_test:
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    Path(__file__).parent.parent / "src" / f"{module_name}.py"
                )
                module = importlib.util.module_from_spec(spec)

                # Check what modules are in sys.modules before import
                modules_before = set(sys.modules.keys())

                # Execute the module
                spec.loader.exec_module(module)

                # Check what new modules were imported
                modules_after = set(sys.modules.keys())
                new_modules = modules_after - modules_before

                # Check for forbidden imports
                forbidden_found = [mod for mod in new_modules
                                 if any(forbidden in mod for forbidden in forbidden_modules)]

                assert not forbidden_found, (
                    f"Module {module_name}.py imports forbidden modules: {forbidden_found}"
                )

            except Exception as e:
                pytest.fail(f"Failed to test module purity for {module_name}: {e}")


class TestArchitecturalRules:
    """Test architectural rules and constraints."""

    @pytest.mark.unit
    def test_function_naming_conventions(self, module_structure):
        """Test that function naming follows architectural conventions."""

        # Data module should have generation/fetch functions
        data_functions = module_structure["data"]["function_defs"]
        data_gen_functions = [f for f in data_functions if f.startswith(("generate_", "fetch_", "get_"))]
        assert len(data_gen_functions) >= 5, f"data.py should have data generation functions, found: {data_gen_functions}"

        # Statistics module should have computation functions
        stats_functions = module_structure["statistics"]["function_defs"]
        stats_comp_functions = [f for f in stats_functions if f.startswith(("compute_", "fit_", "perform_", "calculate_"))]
        assert len(stats_comp_functions) >= 3, f"statistics.py should have computation functions, found: {stats_comp_functions}"

        # Plots module should have visualization functions
        plots_functions = module_structure["plots"]["function_defs"]
        plots_viz_functions = [f for f in plots_functions if f.startswith(("create_", "get_"))]
        assert len(plots_viz_functions) >= 3, f"plots.py should have visualization functions, found: {plots_viz_functions}"

    @pytest.mark.unit
    def test_import_whitelist(self, module_structure):
        """Test that modules only import from allowed sources."""

        # Define allowed imports for each module
        allowed_imports = {
            "data": {"typing", "time", "numpy", "pandas", "streamlit", "statsmodels", "config", "logger"},
            "statistics": {"numpy", "pandas", "scipy", "statsmodels", "typing", "streamlit", "logger"},
            "plots": {"typing", "numpy", "pandas", "plotly", "streamlit", "logger"},
            "content": {"typing"}
        }

        for module_name, allowed in allowed_imports.items():
            if module_name in module_structure:
                actual_imports = module_structure[module_name]["imports"]
                forbidden_imports = actual_imports - allowed

                # Allow some common standard library imports
                std_lib_allowed = {"os", "sys", "pathlib", "functools", "collections", "itertools"}
                forbidden_imports -= std_lib_allowed

                assert not forbidden_imports, (
                    f"{module_name}.py has forbidden imports: {forbidden_imports}. "
                    f"Allowed: {allowed}"
                )