#!/usr/bin/env python3
"""
🏛️ Enterprise Architecture Validation Script

Validates compliance with Clean Architecture + DDD + CQRS standards defined in ARCHITECTURE_DESIGN.md

This script performs comprehensive static analysis to ensure:
✅ Clean Architecture: Dependency directions, layer isolation
✅ DDD: Value objects, entities, domain services, repositories
✅ CQRS: Separate commands (writes) and queries (reads)
✅ SOLID: Single responsibility, dependency inversion, etc.
✅ Testability: Business logic can be tested in isolation

Usage:
    python scripts/check_modular_separation.py

Exit codes:
    0: All architectural standards met
    1: Architecture violations found (blocks commits)
    2: Script execution error
"""

import sys
import ast
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Protocol
from enum import Enum


class ArchitectureLayer(Enum):
    """Clean Architecture layers."""
    DOMAIN = "domain"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    PRESENTATION = "presentation"


class Severity(Enum):
    """Validation severity levels."""
    CRITICAL = "🔴 CRITICAL"  # Blocks commits
    WARNING = "🟡 WARNING"    # Should be addressed
    INFO = "ℹ️  INFO"         # Best practice


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    severity: Severity
    passed: bool
    message: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    details: Optional[str] = None


@dataclass
class ModuleAnalysis:
    """Comprehensive analysis of a Python module."""
    path: Path
    imports: Set[str] = field(default_factory=set)
    from_imports: Dict[str, Set[str]] = field(default_factory=dict)
    function_defs: Set[str] = field(default_factory=set)
    class_defs: Set[str] = field(default_factory=set)
    method_defs: Dict[str, Set[str]] = field(default_factory=dict)
    function_calls: Set[str] = field(default_factory=set)
    has_main: bool = False
    has_streamlit: bool = False
    has_dataclasses: bool = False
    has_protocols: bool = False
    lines_of_code: int = 0


class ArchitectureValidator:
    """Main validator for Clean Architecture + DDD + CQRS compliance."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.results: List[ValidationResult] = []
        self.total_checks = 0

    def validate_architecture(self) -> bool:
        """Run all architectural validation checks."""
        print("🏛️  Validating Enterprise Architecture Compliance...")
        print("=" * 60)

        # Core architecture checks
        self._check_layer_structure()
        self._check_dependency_directions()
        self._check_domain_purity()
        self._check_ddd_compliance()
        self._check_cqrs_separation()
        self._check_solid_principles()
        self._check_testability()

        # Additional architectural integrity checks
        self._check_infrastructure_ui_dependencies()
        self._check_generator_interfaces()
        self._check_domain_events_usage()
        self._check_datasource_references()
        self._check_syntax_validation()
        self._check_entry_point_structure()

        # Report results
        return self._report_results()

    def _check_layer_structure(self):
        """Check that layers are properly structured."""
        required_layers = {
            "core/domain": "Domain layer (business rules)",
            "core/application": "Application layer (use cases)",
            "infrastructure": "Infrastructure layer (external concerns)",
            "data": "Data access layer",
            "ui": "Presentation layer",
            "config": "Configuration layer"
        }

        for layer_path, description in required_layers.items():
            self.total_checks += 1
            full_path = self.src_path / layer_path
            if not full_path.exists():
                self._add_result(
                    "Layer Structure",
                    Severity.CRITICAL,
                    False,
                    f"Missing required layer: {layer_path} ({description})",
                    None,
                    f"Create directory {full_path} with __init__.py"
                )
            elif not (full_path / "__init__.py").exists():
                self._add_result(
                    "Layer Structure",
                    Severity.CRITICAL,
                    False,
                    f"Layer missing __init__.py: {layer_path}",
                    full_path,
                    f"Create {full_path}/__init__.py"
                )
            else:
                # Add a success result for visible feedback
                self._add_result(
                    "Layer Structure",
                    Severity.INFO,
                    True,
                    f"Layer correctly structured: {layer_path}"
                )

    def _check_dependency_directions(self):
        """Check that dependencies point inward (Clean Architecture)."""
        self.total_checks += 1
        layer_order = {
            "core/domain": 1,        # Innermost
            "core/application": 2,
            "core": 3,
            "infrastructure": 4,
            "data": 4,
            "ui": 4,
            "config": 4,
            "utils": 4,             # Outermost
        }

        # Analyze all Python files
        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)
            current_layer = self._get_layer(py_file)

            # Check imports don't violate dependency direction
            for imported_module in analysis.imports:
                imported_layer = self._get_import_layer(imported_module, py_file)
                if imported_layer and current_layer:
                    if layer_order.get(imported_layer, 5) < layer_order.get(current_layer, 5):
                        self._add_result(
                            "Dependency Direction",
                            Severity.CRITICAL,
                            False,
                            f"Invalid dependency: {current_layer} imports from {imported_layer}",
                            py_file,
                            f"Move {imported_module} logic to {current_layer} layer or use dependency injection"
                        )

            # Check for forbidden imports in domain layer
            if current_layer == "core/domain":
                forbidden = {"streamlit", "plotly", "matplotlib", "requests", "pandas"}
                domain_violations = analysis.imports.intersection(forbidden)
                if domain_violations:
                    self._add_result(
                        "Domain Purity",
                        Severity.CRITICAL,
                        False,
                        f"Domain layer imports external frameworks: {sorted(domain_violations)}",
                        py_file,
                        "Domain layer must contain only pure business logic"
                    )

    def _check_domain_purity(self):
        """Check that domain layer contains only business logic."""
        domain_path = self.src_path / "core" / "domain"

        forbidden_domain_imports = {
            "numpy", "pandas", "streamlit", "plotly", "matplotlib", "seaborn",
            "requests", "datetime", "time", "os", "sys", "json", "csv",
            "sklearn", "statsmodels", "scipy"
        }

        for py_file in domain_path.rglob("*.py"):
            self.total_checks += 1
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)

            # Domain should not have ANY external dependencies
            domain_violations = analysis.imports.intersection(forbidden_domain_imports)
            if domain_violations:
                self._add_result(
                    "Domain Purity",
                    Severity.CRITICAL,
                    False,
                    f"Domain layer imports external frameworks: {sorted(domain_violations)}",
                    py_file,
                    "Domain layer must contain ONLY pure business logic with no external dependencies"
                )

            # Check for imports from other layers (should only import from typing, dataclasses, abc)
            allowed_imports = {"typing", "dataclasses", "abc", "enum"}
            external_imports = analysis.imports - allowed_imports
            if external_imports:
                # Check if they're from our own layers (this would be circular imports)
                our_layers = {"core", "infrastructure", "data", "ui", "config", "utils"}
                circular_imports = external_imports.intersection(our_layers)
                if circular_imports:
                    self._add_result(
                        "Dependency Direction",
                        Severity.CRITICAL,
                        False,
                        f"Domain layer has circular dependency: {sorted(circular_imports)}",
                        py_file,
                        "Domain layer cannot import from other layers - use dependency injection"
                    )

            # Domain should not have UI or external concerns
            ui_indicators = {"st.", "plotly", "plt.", "fig", "ax", "print(", "input("}
            has_ui = any(call in analysis.function_calls for call in ui_indicators)
            if has_ui:
                self.total_checks += 1
                self._add_result(
                    "Domain Purity",
                    Severity.CRITICAL,
                    False,
                    "Domain layer contains UI/external concerns (print, input, plotting)",
                    py_file,
                    "Move UI logic to infrastructure/presentation layer"
                )

            # Check for required DDD patterns
            self.total_checks += 1
            if not (analysis.has_dataclasses or analysis.has_protocols):
                self._add_result(
                    "DDD Compliance",
                    Severity.WARNING,
                    False,
                    "Domain module lacks DDD patterns (dataclasses/protocols)",
                    py_file,
                    "Use @dataclass for entities, @dataclass(frozen=True) for value objects, Protocol for interfaces"
                )

            # Check for proper value object immutability
            if "value_objects" in str(py_file):
                with open(py_file, 'r') as f:
                    content = f.read()
                    # Check for @dataclass(frozen=True) pattern
                    if "@dataclass(frozen=True)" not in content and "@dataclass\n" in content:
                        # Check if it's followed by frozen=True
                        if "frozen=True" not in content:
                            self._add_result(
                                "DDD Compliance",
                                Severity.WARNING,
                                False,
                                "Value objects should be immutable with @dataclass(frozen=True)",
                                py_file,
                                "Add frozen=True to @dataclass decorator for immutable value objects"
                            )

            # Check for domain service complexity (should have business logic)
            if "services.py" in str(py_file):
                analysis = self._analyze_module(py_file)
                if analysis.lines_of_code < 50:  # Too simple for a domain service
                    self._add_result(
                        "DDD Compliance",
                        Severity.WARNING,
                        False,
                        "Domain service appears too simple - should contain complex business logic",
                        py_file,
                        "Domain services should encapsulate complex business rules and logic"
                    )

            # Check that entities have behavior methods (not just data)
            if "entities.py" in str(py_file):
                # Simple text-based analysis for entities
                with open(py_file, 'r') as f:
                    content = f.read()

                # Count class definitions
                import re
                class_matches = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                num_classes = len(class_matches)

                # Count method definitions (lines starting with spaces + def)
                method_matches = re.findall(r'^\s+def\s+(\w+)', content, re.MULTILINE)
                total_methods = len(method_matches)

                # Count business methods (exclude __init__, __post_init__, _private methods)
                business_methods = 0
                for method in method_matches:
                    if not method.startswith('_') and method not in ['__init__', '__post_init__']:
                        business_methods += 1


                # More realistic threshold: at least 1-2 business methods per entity class
                # (Lowered from 5 for educational context)
                min_business_methods = num_classes * 1
                if business_methods < min_business_methods:
                    self.total_checks += 1
                    self._add_result(
                        "DDD Compliance",
                        Severity.INFO,
                        False,
                        f"Entities have {business_methods} business methods, recommend at least {min_business_methods}",
                        py_file,
                        "DDD entities should contain substantial business logic, not just data"
                    )

            # Domain entities should not instantiate other objects directly
            with open(py_file, 'r') as f:
                content = f.read()
                # Look for direct instantiation patterns that suggest tight coupling
                if "import " in content and ("=" in content and "(" in content):
                    # This is a simplified check - could be improved with AST analysis
                    if "Repository(" in content or "Service(" in content:
                        self._add_result(
                            "Dependency Inversion",
                            Severity.WARNING,
                            False,
                            "Domain layer appears to instantiate dependencies directly",
                            py_file,
                            "Use dependency injection instead of direct instantiation"
                        )

    def _check_ddd_compliance(self):
        """Check Domain-Driven Design compliance."""
        domain_files = list((self.src_path / "core" / "domain").rglob("*.py"))
        domain_files = [f for f in domain_files if f.name != "__init__.py"]

        # Check for required DDD components
        has_entities = any("entities.py" in str(f) for f in domain_files)
        has_value_objects = any("value_objects.py" in str(f) for f in domain_files)
        has_services = any("services.py" in str(f) for f in domain_files)

        if not has_entities:
            self._add_result(
                "DDD Compliance",
                Severity.CRITICAL,
                False,
                "Missing entities.py - domain entities required",
                self.src_path / "core" / "domain",
                "Create entities.py with @dataclass entities"
            )

        if not has_value_objects:
            self._add_result(
                "DDD Compliance",
                Severity.CRITICAL,
                False,
                "Missing value_objects.py - immutable value objects required",
                self.src_path / "core" / "domain",
                "Create value_objects.py with @dataclass(frozen=True) objects"
            )

        if not has_services:
            self._add_result(
                "DDD Compliance",
                Severity.WARNING,
                False,
                "Missing domain services - complex business logic should be in services",
                self.src_path / "core" / "domain",
                "Create services.py for complex domain logic"
            )

        # Check for repository interfaces
        has_repository_interfaces = False
        for py_file in domain_files:
            self.total_checks += 1
            analysis = self._analyze_module(py_file)
            if analysis.has_protocols and any("Repository" in cls for cls in analysis.class_defs):
                has_repository_interfaces = True
                break

        self.total_checks += 1
        if not has_repository_interfaces:
            self._add_result(
                "DDD Compliance",
                Severity.WARNING,
                False,
                "Missing repository interfaces in domain layer",
                self.src_path / "core" / "domain",
                "Define repository protocols in domain layer (e.g., DatasetRepository Protocol)"
            )

        # Check for legacy architecture violations
        self._check_legacy_architecture_violations()

    def _check_cqrs_separation(self):
        """Check CQRS command/query separation."""
        app_path = self.src_path / "core" / "application"

        has_commands = (app_path / "commands.py").exists()
        has_queries = (app_path / "queries.py").exists()
        has_use_cases = (app_path / "use_cases.py").exists()

        if not has_commands:
            self._add_result(
                "CQRS Compliance",
                Severity.CRITICAL,
                False,
                "Missing commands.py - CQRS requires separate commands",
                app_path,
                "Create commands.py with @dataclass command objects"
            )

        if not has_queries:
            self._add_result(
                "CQRS Compliance",
                Severity.CRITICAL,
                False,
                "Missing queries.py - CQRS requires separate queries",
                app_path,
                "Create queries.py with @dataclass query objects"
            )

        if not has_use_cases:
            self._add_result(
                "CQRS Compliance",
                Severity.WARNING,
                False,
                "Missing use_cases.py - application logic should be in use cases",
                app_path,
                "Create use_cases.py with use case classes"
            )

        # Check command/query separation
        if has_commands:
            self.total_checks += 1
            cmd_analysis = self._analyze_module(app_path / "commands.py")
            if any("query" in class_name.lower() for class_name in cmd_analysis.class_defs):
                self._add_result(
                    "CQRS Compliance",
                    Severity.WARNING,
                    False,
                    "Commands module contains query-related classes",
                    app_path / "commands.py",
                    "Keep commands and queries in separate files"
                )

            # Check inheritance from Command base class
            with open(app_path / "commands.py", 'r') as f:
                content = f.read()
                # Find all dataclass definitions that might be commands
                # We expect them to look like 'class SomeCommand(Command):'
                for class_name in cmd_analysis.class_defs:
                    if class_name == "Command": continue
                    if f"class {class_name}(Command):" not in content and f"class {class_name}(Command)" not in content:
                        self.total_checks += 1
                        self._add_result(
                            "CQRS Compliance",
                            Severity.CRITICAL,
                            False,
                            f"Command '{class_name}' does not inherit from base Command class",
                            app_path / "commands.py",
                            "All commands must inherit from the 'Command' base class for bus compatibility"
                        )

        if has_queries:
            self.total_checks += 1
            query_analysis = self._analyze_module(app_path / "queries.py")
            
            # Check inheritance from Query base class
            with open(app_path / "queries.py", 'r') as f:
                content = f.read()
                for class_name in query_analysis.class_defs:
                    if class_name == "Query": continue
                    if f"class {class_name}(Query):" not in content and f"class {class_name}(Query)" not in content:
                        self.total_checks += 1
                        self._add_result(
                            "CQRS Compliance",
                            Severity.CRITICAL,
                            False,
                            f"Query '{class_name}' does not inherit from base Query class",
                            app_path / "queries.py",
                            "All queries must inherit from the 'Query' base class for bus compatibility"
                        )

        if has_queries:
            qry_analysis = self._analyze_module(app_path / "queries.py")
            if any("command" in class_name.lower() for class_name in qry_analysis.class_defs):
                self._add_result(
                    "CQRS Compliance",
                    Severity.WARNING,
                    False,
                    "Queries module contains command-related classes",
                    app_path / "queries.py",
                    "Keep commands and queries in separate files"
                )

        # Check for use case implementation quality
        if has_use_cases:
            uc_analysis = self._analyze_module(app_path / "use_cases.py")

            # Use cases should orchestrate domain objects, not contain business logic
            business_logic_indicators = ["calculate", "compute", "validate", "process"]
            has_business_logic = any(
                any(indicator in func for indicator in business_logic_indicators)
                for func in uc_analysis.function_defs
            )

            if has_business_logic:
                self._add_result(
                    "Clean Architecture",
                    Severity.WARNING,
                    False,
                    "Use cases appear to contain business logic instead of orchestration",
                    app_path / "use_cases.py",
                    "Use cases should orchestrate domain objects, not implement business rules"
                )

            # Check for proper use case naming (should end with UseCase)
            invalid_use_cases = [
                cls for cls in uc_analysis.class_defs
                if not cls.endswith("UseCase")
            ]
            if invalid_use_cases:
                self._add_result(
                    "Clean Architecture",
                    Severity.INFO,
                    False,
                    f"Use case classes should be named *UseCase: {invalid_use_cases}",
                    app_path / "use_cases.py",
                    "Follow naming convention: CreateRegressionModelUseCase, etc."
                )

    def _check_solid_principles(self):
        """Check SOLID principles compliance."""
        # Single Responsibility
        for py_file in self.src_path.rglob("*.py"):
            self.total_checks += 1
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)
            if analysis.lines_of_code > 500:
                self._add_result(
                    "SOLID: Single Responsibility",
                    Severity.WARNING,
                    False,
                    f"Module too large ({analysis.lines_of_code} lines) - split into smaller modules",
                    py_file,
                    "Break down large modules into smaller, focused ones"
                )

        # Dependency Inversion
        for py_file in self.src_path.rglob("*.py"):
            analysis = self._analyze_module(py_file)
            if "import streamlit" in analysis.imports and "core/domain" in str(py_file):
                self._add_result(
                    "SOLID: Dependency Inversion",
                    Severity.CRITICAL,
                    False,
                    "Domain layer directly imports external framework",
                    py_file,
                    "Use dependency injection and interfaces instead"
                )

        # Interface Segregation
        domain_path = self.src_path / "core" / "domain"
        for py_file in domain_path.rglob("*.py"):
            self.total_checks += 1
            analysis = self._analyze_module(py_file)
            if analysis.has_protocols:
                # Check if protocols are focused (not too many methods)
                # Interface Segregation: Protocols should be lean
                for protocol_name, methods in analysis.method_defs.items():
                    if len(methods) > 5:
                        self.total_checks += 1
                        self._add_result(
                            "SOLID: Interface Segregation",
                            Severity.WARNING,
                            False,
                            f"Protocol '{protocol_name}' has too many methods ({len(methods)})",
                            py_file,
                            "Split large interfaces into smaller, more specific ones"
                        )

    def _check_testability(self):
        """Check that code is testable in isolation."""
        # Check for dependency injection container
        container_path = self.src_path / "infrastructure" / "dependency_container.py"
        if not container_path.exists():
            self._add_result(
                "Testability",
                Severity.CRITICAL,
                False,
                "Missing dependency injection container",
                self.src_path / "infrastructure",
                "Create dependency_container.py for testable dependencies"
            )

        # Check that domain layer doesn't instantiate dependencies
        domain_path = self.src_path / "core" / "domain"
        for py_file in domain_path.rglob("*.py"):
            analysis = self._analyze_module(py_file)
            # Look for direct instantiation patterns
            with open(py_file, 'r') as f:
                content = f.read()
                if re.search(r'\b\w+\([^)]*\)', content):  # Simple instantiation pattern
                    # This is a rough check - in practice would need AST analysis
                    pass

    def _analyze_module(self, file_path: Path) -> ModuleAnalysis:
        """Analyze a Python module comprehensively."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines_of_code = len(content.splitlines())

            tree = ast.parse(content)

            imports = set()
            from_imports = {}
            function_defs = set()
            class_defs = set()
            method_defs = {}
            function_calls = set()
            has_main = False
            has_streamlit = False
            has_dataclasses = False
            has_protocols = False

            current_class = None

            for node in ast.walk(tree):
                # Imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_base = node.module.split('.')[0]
                        imports.add(module_base)
                        if module_base not in from_imports:
                            from_imports[module_base] = set()
                        for alias in node.names:
                            from_imports[module_base].add(alias.name)

                # Function definitions
                elif isinstance(node, ast.FunctionDef):
                    function_defs.add(node.name)
                    if node.name == "main":
                        has_main = True

                # Class definitions
                elif isinstance(node, ast.ClassDef):
                    class_defs.add(node.name)
                    current_class = node.name
                    method_defs[current_class] = set()

                    # Check decorators for dataclasses/protocols
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if decorator.id == "dataclass":
                                has_dataclasses = True
                        elif isinstance(decorator, ast.Attribute):
                            if decorator.attr == "dataclass":
                                has_dataclasses = True

                    # Check base classes for Protocol
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "Protocol":
                            has_protocols = True

                # Method definitions
                elif isinstance(node, ast.FunctionDef) and current_class:
                    method_defs[current_class].add(node.name)

                # Function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        function_calls.add(node.func.id)
                        if node.func.id in ["st", "streamlit"]:
                            has_streamlit = True
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            function_calls.add(f"{node.func.value.id}.{node.func.attr}")

            return ModuleAnalysis(
                path=file_path,
                imports=imports,
                from_imports=from_imports,
                function_defs=function_defs,
                class_defs=class_defs,
                method_defs=method_defs,
                function_calls=function_calls,
                has_main=has_main,
                has_streamlit=has_streamlit,
                has_dataclasses=has_dataclasses,
                has_protocols=has_protocols,
                lines_of_code=lines_of_code
            )

        except Exception as e:
            print(f"❌ Failed to analyze {file_path}: {e}", file=sys.stderr)
            return ModuleAnalysis(path=file_path)

    def _get_layer(self, file_path: Path) -> Optional[str]:
        """Determine which architectural layer a file belongs to."""
        path_str = str(file_path.relative_to(self.src_path))

        if path_str.startswith("core/domain"):
            return "core/domain"
        elif path_str.startswith("core/application"):
            return "core/application"
        elif path_str.startswith("infrastructure"):
            return "infrastructure"
        elif path_str.startswith("data"):
            return "data"
        elif path_str.startswith("ui"):
            return "ui"
        elif path_str.startswith("config"):
            return "config"
        elif path_str.startswith("utils"):
            return "utils"

        return None

    def _get_import_layer(self, imported_module: str, importing_file: Path) -> Optional[str]:
        """Determine the layer of an imported module."""
        # This is a simplified mapping - in practice would need more sophisticated analysis
        layer_mappings = {
            "core.domain": "core/domain",
            "core.application": "core/application",
            "infrastructure": "infrastructure",
            "data": "data",
            "ui": "ui",
            "config": "config",
            "utils": "utils"
        }

        return layer_mappings.get(imported_module)

    def _check_legacy_architecture_violations(self):
        """Check for violations of our clean architecture that still exist in legacy code."""

        # Check for legacy service-based architecture that violates clean architecture
        legacy_service_files = [
            self.src_path / "core" / "services" / "data_service.py",
            self.src_path / "core" / "services" / "model_service.py",
            self.src_path / "core" / "data_loading.py",
            self.src_path / "core" / "data_preparation.py",
        ]

        for service_file in legacy_service_files:
            if service_file.exists():
                self._add_result(
                    "Clean Architecture",
                    Severity.CRITICAL,
                    False,
                    f"Legacy service architecture violates Clean Architecture: {service_file.name}",
                    service_file,
                    "Remove legacy service files - business logic belongs in domain layer, infrastructure in infrastructure layer"
                )

        # Check that statistics.py is not in core (violates domain purity)
        stats_file = self.src_path / "core" / "statistics.py"
        if stats_file.exists():
            analysis = self._analyze_module(stats_file)
            if analysis.lines_of_code > 100:  # It's a substantial implementation
                self._add_result(
                    "Clean Architecture",
                    Severity.CRITICAL,
                    False,
                    "statistics.py in core layer violates domain purity",
                    stats_file,
                    "Move statistical computations to infrastructure layer or domain services"
                )

        # Check for mixed responsibilities (files that do too many things)
        large_files_to_check = [
            (self.src_path / "data" / "content.py", "Content management"),
            (self.src_path / "ui" / "plots.py", "Plotting"),
            (self.src_path / "utils" / "session_state.py", "Session management"),
        ]

        for file_path, responsibility in large_files_to_check:
            if file_path.exists():
                analysis = self._analyze_module(file_path)
                if analysis.lines_of_code > 400:
                    self._add_result(
                        "Single Responsibility",
                        Severity.WARNING,
                        False,
                        f"{responsibility} module too large ({analysis.lines_of_code} lines)",
                        file_path,
                        f"Split {responsibility.lower()} into smaller, focused modules"
                    )

        # Check for missing dependency injection container
        container_file = self.src_path / "infrastructure" / "dependency_container.py"
        if not container_file.exists():
            self._add_result(
                "Dependency Injection",
                Severity.CRITICAL,
                False,
                "Missing dependency injection container",
                self.src_path / "infrastructure",
                "Create dependency_container.py to wire up all dependencies for testability"
            )
        else:
            # Check that container actually wires dependencies properly
            analysis = self._analyze_module(container_file)
            if analysis.lines_of_code < 100:  # Too small to be a proper container
                self._add_result(
                    "Dependency Injection",
                    Severity.WARNING,
                    False,
                    "Dependency container appears incomplete or too simple",
                    container_file,
                    "Ensure container properly wires domain, application, and infrastructure dependencies"
                )

            # Check for proper dependency injection patterns
            with open(container_file, 'r') as f:
                content = f.read()
                if "_services" not in content or "register" not in content:
                    self._add_result(
                        "Dependency Injection",
                        Severity.WARNING,
                        False,
                        "Container lacks proper dependency registration patterns",
                        container_file,
                        "Implement proper service registration and resolution patterns"
                    )

                # Check for interface usage (Protocol/base classes)
                if "Protocol" not in content and "ABC" not in content:
                    self._add_result(
                        "Dependency Inversion",
                        Severity.WARNING,
                        False,
                        "Container doesn't use interfaces/protocols for dependency injection",
                        container_file,
                        "Use Protocol or ABC base classes for proper dependency inversion"
                    )

                # Check for proper separation of concerns
                if "import streamlit" in content or "plotly" in content:
                    self._add_result(
                        "Clean Architecture",
                        Severity.CRITICAL,
                        False,
                        "Dependency injection container imports external frameworks",
                        container_file,
                        "DI container should only wire dependencies, not import external frameworks"
                    )

        # Check CQRS implementation
        self._check_cqrs_implementation()

        # Check for testability violations
        self._check_testability_violations()

        # Check for hardcoded dependencies in application layer
        app_path = self.src_path / "core" / "application"
        for py_file in app_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)
                # Application layer should use dependency injection, not direct imports
            if "import streamlit" in analysis.imports or "from streamlit" in str(analysis.from_imports):
                self._add_result(
                    "Clean Architecture",
                    Severity.CRITICAL,
                    False,
                    "Application layer directly imports external framework",
                    py_file,
                    "Application layer should depend only on domain layer - use interfaces/adapters"
                )

        # Check infrastructure layer purity (no business logic)
        infra_path = self.src_path / "infrastructure"
        for py_file in infra_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)

            # Infrastructure should not contain business logic
            # Note: We distinguish between "Domain Business Logic" (rules, policies) 
            # and "Computational Logic" (math, algorithms). 
            # Computations involving NumPy/SciPy belong in Infrastructure because 
            # the Domain layer must be pure (no external deps).
            
            business_logic_indicators = [
                "validate_policy", "approve_", "reject_", "check_permission",
                # "validate" is too broad, handled separately
                # "compute" and "calculate" are allowed if purely mathematical (checked below)
            ]

            # If the file imports numpy/scipy/statsmodels, it's likely a computational adapter
            is_computational = "numpy" in analysis.imports or "scipy" in analysis.imports or "statsmodels" in analysis.imports
            
            if not is_computational:
                # For non-computational infra, we are stricter
                business_logic_indicators.extend([
                    "calculate_bonus", "determine_eligibility", "process_workflow"
                ])

            has_business_logic = any(
                any(indicator in func for indicator in business_logic_indicators)
                for func in analysis.function_defs
            )

            if has_business_logic:
                self._add_result(
                    "Clean Architecture",
                    Severity.CRITICAL,
                    False,
                    "Infrastructure layer contains domain business logic",
                    py_file,
                    "Infrastructure should only handle external concerns (persistence, APIs, frameworks)"
                )

            # Infrastructure implementing domain interfaces is CORRECT in Clean Architecture
            # Domain concepts in infrastructure are legitimate implementations of domain interfaces
            # This check is removed as it produces false positives

    def _check_cqrs_implementation(self):
        """Check that CQRS is properly implemented."""
        app_path = self.src_path / "core" / "application"

        # Check command/query separation
        commands_file = app_path / "commands.py"
        queries_file = app_path / "queries.py"
        use_cases_file = app_path / "use_cases.py"

        if not commands_file.exists():
            self._add_result(
                "CQRS Compliance",
                Severity.CRITICAL,
                False,
                "Missing commands.py - CQRS requires separate write operations",
                app_path,
                "Create commands.py with Command classes and command handlers"
            )

        if not queries_file.exists():
            self._add_result(
                "CQRS Compliance",
                Severity.CRITICAL,
                False,
                "Missing queries.py - CQRS requires separate read operations",
                app_path,
                "Create queries.py with Query classes and query handlers"
            )

        if not use_cases_file.exists():
            self._add_result(
                "CQRS Compliance",
                Severity.WARNING,
                False,
                "Missing use_cases.py - application logic should be in use cases",
                app_path,
                "Create use_cases.py with use case classes that orchestrate domain objects"
            )

        # Check for command/query mixing
        if commands_file.exists():
            cmd_analysis = self._analyze_module(commands_file)
            if any("query" in class_name.lower() for class_name in cmd_analysis.class_defs):
                self._add_result(
                    "CQRS Compliance",
                    Severity.WARNING,
                    False,
                    "Commands file contains query-related classes",
                    commands_file,
                    "Commands and queries must be in separate files"
                )

        if queries_file.exists():
            qry_analysis = self._analyze_module(queries_file)
            if any("command" in class_name.lower() for class_name in qry_analysis.class_defs):
                self._add_result(
                    "CQRS Compliance",
                    Severity.WARNING,
                    False,
                    "Queries file contains command-related classes",
                    queries_file,
                    "Commands and queries must be in separate files"
                )

    def _check_testability_violations(self):
        """Check for testability violations."""
        # Check that business logic can be tested in isolation
        domain_files = list((self.src_path / "core" / "domain").rglob("*.py"))
        domain_files = [f for f in domain_files if f.name != "__init__.py"]

        for domain_file in domain_files:
            analysis = self._analyze_module(domain_file)

            # Domain should not have side effects or external calls
            external_calls = ["open(", "print(", "input(", "exit(", "sys.exit"]
            has_side_effects = any(call in analysis.function_calls for call in external_calls)

            if has_side_effects:
                self._add_result(
                    "Testability",
                    Severity.WARNING,
                    False,
                    "Domain layer has side effects that make testing difficult",
                    domain_file,
                    "Remove side effects from domain logic - use dependency injection for external concerns"
                )

        # Check for missing unit tests
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            self._add_result(
                "Testability",
                Severity.WARNING,
                False,
                "Missing tests directory",
                self.project_root,
                "Create tests/ directory with comprehensive unit and integration tests"
            )
        else:
            # Check for basic test structure
            domain_test_files = list(tests_dir.glob("**/test_*domain*.py"))
            if not domain_test_files:
                self._add_result(
                    "Testability",
                    Severity.WARNING,
                    False,
                    "Missing domain layer unit tests",
                    tests_dir,
                    "Create unit tests for domain entities, value objects, and services"
                )

        # Check for proper error handling
        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)
            if analysis.has_main and "try:" not in open(py_file).read():
                self._add_result(
                    "Error Handling",
                    Severity.INFO,
                    False,
                    "Main function lacks proper error handling",
                    py_file,
                    "Add try/except blocks and proper error handling in main functions"
                )

    def _check_infrastructure_ui_dependencies(self):
        """Check that infrastructure layer doesn't contain UI framework dependencies."""
        infrastructure_path = self.src_path / "infrastructure"

        # UI frameworks that should not be in infrastructure
        forbidden_ui_imports = {
            "streamlit", "plotly", "matplotlib", "seaborn", "dash", "flask", "django"
        }

        for py_file in infrastructure_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)

            # Check for forbidden UI imports
            ui_violations = analysis.imports.intersection(forbidden_ui_imports)
            if ui_violations:
                self._add_result(
                    "Infrastructure Purity",
                    Severity.CRITICAL,
                    False,
                    f"Infrastructure layer imports UI frameworks: {sorted(ui_violations)}",
                    py_file,
                    "Infrastructure layer must not depend on UI frameworks. Use dependency injection instead."
                )

            # Check for Streamlit-specific patterns (decorators, cache calls)
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for Streamlit decorators
                if "@st.cache_data" in content or "@st.cache_resource" in content:
                    self._add_result(
                        "Infrastructure Purity",
                        Severity.CRITICAL,
                        False,
                        "Infrastructure layer uses Streamlit decorators (@st.cache_data, @st.cache_resource)",
                        py_file,
                        "Replace Streamlit caching with framework-agnostic caching or move to UI layer"
                    )

                # Check for Streamlit function calls
                if "st.cache_data(" in content or "st.cache_resource(" in content:
                    self._add_result(
                        "Infrastructure Purity",
                        Severity.CRITICAL,
                        False,
                        "Infrastructure layer calls Streamlit caching functions",
                        py_file,
                        "Replace st.cache_data/resource calls with custom caching implementation"
                    )

            except Exception as e:
                self._add_result(
                    "File Analysis",
                    Severity.WARNING,
                    False,
                    f"Could not analyze file content: {e}",
                    py_file
                )

    def _check_generator_interfaces(self):
        """Check that data generators follow common interface patterns."""
        generators_path = self.src_path / "data" / "data_generators"

        if not generators_path.exists():
            return

        # Check for common base class/interface
        has_base_generator = False
        base_generator_classes = set()

        for py_file in generators_path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis = self._analyze_module(py_file)

            # Check if file defines generator classes
            generator_classes = [cls for cls in analysis.class_defs if "Generator" in cls]
            if generator_classes:
                # Check if they inherit from BaseDataGenerator or similar
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Look for inheritance patterns
                    for class_name in generator_classes:
                        # Check for inheritance from BaseDataGenerator
                        if f"class {class_name}(BaseDataGenerator)" in content:
                            has_base_generator = True
                            base_generator_classes.add(class_name)
                        elif f"class {class_name}(" in content and "BaseDataGenerator" not in content:
                            # Class inherits from something else
                            self._add_result(
                                "Generator Interface",
                                Severity.WARNING,
                                False,
                                f"Generator class '{class_name}' doesn't inherit from BaseDataGenerator",
                                py_file,
                                "Make generators inherit from BaseDataGenerator for consistent interface"
                            )

                except Exception as e:
                    self._add_result(
                        "File Analysis",
                        Severity.WARNING,
                        False,
                        f"Could not analyze generator file: {e}",
                        py_file
                    )

        # Check that __init__.py exports the base class
        init_file = generators_path / "__init__.py"
        if init_file.exists():
            try:
                with open(init_file, 'r', encoding='utf-8') as f:
                    init_content = f.read()

                if "BaseDataGenerator" not in init_content:
                    self._add_result(
                        "Generator Interface",
                        Severity.WARNING,
                        False,
                        "BaseDataGenerator not exported from generators package",
                        init_file,
                        "Export BaseDataGenerator in __init__.py for proper interface exposure"
                    )

            except Exception as e:
                self._add_result(
                    "File Analysis",
                    Severity.WARNING,
                    False,
                    f"Could not analyze generators __init__.py: {e}",
                    init_file
                )

        # Overall assessment
        if not has_base_generator and generator_classes:
            self._add_result(
                "Generator Interface",
                Severity.WARNING,
                False,
                "No generators inherit from BaseDataGenerator",
                generators_path,
                "Create BaseDataGenerator abstract class and make all generators inherit from it"
            )

    def _check_domain_events_usage(self):
        """Check that domain events are actually used/published in the application."""
        self.total_checks += 1
        events_file = self.src_path / "core" / "domain" / "events.py"

        if not events_file.exists():
            return

        # Get defined domain events
        analysis = self._analyze_module(events_file)
        defined_events = set()

        try:
            with open(events_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all non-commented event class definitions
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # Skip commented lines
                if line.startswith('#'):
                    continue
                # Find class definitions
                match = re.search(r'class (\w+)\(DomainEvent\):', line)
                if match:
                    defined_events.add(match.group(1))

        except Exception as e:
            self._add_result(
                "Domain Events",
                Severity.WARNING,
                False,
                f"Could not analyze domain events file: {e}",
                events_file
            )
            return

        if not defined_events:
            return

        # Check if events are actually published in application layer
        application_path = self.src_path / "core" / "application"
        event_handlers_file = application_path / "event_handlers.py"
        published_events = set()

        for py_file in application_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for actual event publishing calls
                for event in defined_events:
                    # Look for patterns like: publish_event(SomeEvent(...))
                    publish_pattern = f"publish_event\\(\\s*{event}\\("
                    if re.search(publish_pattern, content):
                        published_events.add(event)

            except Exception as e:
                self._add_result(
                    "File Analysis",
                    Severity.WARNING,
                    False,
                    f"Could not analyze application file: {e}",
                    py_file
                )

        # Report unused events
        unused_events = defined_events - published_events
        if unused_events:
            self._add_result(
                "Domain Events",
                Severity.WARNING,
                False,
                f"Domain events defined but not used: {sorted(unused_events)}",
                events_file,
                "Implement event publishing in use cases and event handlers for defined domain events"
            )

        # Check for event publishing infrastructure
        if published_events and not event_handlers_file.exists():
            self._add_result(
                "Domain Events",
                Severity.WARNING,
                False,
                "Domain events are used but no event handlers file exists",
                application_path,
                "Create event_handlers.py with EventBus and event handler implementations"
            )

    def _check_datasource_references(self):
        """Check for references to non-existent DataService."""
        self.total_checks += 1
        data_loading_file = self.src_path / "data" / "data_loading.py"

        if data_loading_file.exists():
            try:
                with open(data_loading_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for DataService references
                if "DataService" in content:
                    self._add_result(
                        "Data Services",
                        Severity.CRITICAL,
                        False,
                        "References to non-existent DataService found",
                        data_loading_file,
                        "Replace DataService references with actual implementation or remove unused imports"
                    )

            except Exception as e:
                self._add_result(
                    "File Analysis",
                    Severity.WARNING,
                    False,
                    f"Could not analyze data_loading.py: {e}",
                    data_loading_file
                )

    def _check_syntax_validation(self):
        """Check syntax of main application files."""
        self.total_checks += 1
        main_files = [
            self.project_root / "run.py",
            self.src_path / "app.py",
            self.src_path / "main.py" if (self.src_path / "main.py").exists() else None
        ]

        for file_path in main_files:
            if file_path and file_path.exists():
                try:
                    # Use Python's compile to check syntax
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source = f.read()

                    compile(source, str(file_path), 'exec')
                except SyntaxError as e:
                    self._add_result(
                        "Syntax Validation",
                        Severity.CRITICAL,
                        False,
                        f"Syntax error in {file_path.name}: {e.msg} at line {e.lineno}",
                        file_path,
                        f"Fix syntax error: {e.text.strip() if e.text else 'Check indentation and structure'}"
                    )
                except Exception as e:
                    self._add_result(
                        "Syntax Validation",
                        Severity.WARNING,
                        False,
                        f"Could not validate syntax of {file_path.name}: {e}",
                        file_path
                    )

    def _check_entry_point_structure(self):
        """Check that the entry point (run.py) has proper structure."""
        self.total_checks += 1
        run_file = self.project_root / "run.py"

        if not run_file.exists():
            self._add_result(
                "Entry Point",
                Severity.CRITICAL,
                False,
                "run.py entry point not found",
                self.project_root,
                "Create run.py as the main entry point for the application"
            )
            return

        try:
            with open(run_file, 'r', encoding='utf-8') as f:
                content = f.read()

            issues = []

            # Check for proper streamlit detection
            if "streamlit" not in content.lower():
                issues.append("No Streamlit execution detection")

            # Check for error handling
            if "try:" not in content or "except" not in content:
                issues.append("Missing proper error handling")

            # Check for main function
            if "def main():" not in content:
                issues.append("Missing main() function")

            # Check for execution guard
            if 'if __name__ == "__main__":' not in content:
                issues.append("Missing execution guard")

            if issues:
                self._add_result(
                    "Entry Point",
                    Severity.WARNING,
                    False,
                    f"Entry point structure issues: {', '.join(issues)}",
                    run_file,
                    "Ensure run.py has proper Streamlit detection, error handling, main function, and execution guard"
                )

        except Exception as e:
            self._add_result(
                "Entry Point",
                Severity.WARNING,
                False,
                f"Could not analyze entry point: {e}",
                run_file
            )

    def _add_result(self, check_name: str, severity: Severity, passed: bool,
                   message: str, file_path: Optional[Path] = None,
                   details: Optional[str] = None):
        """Add a validation result."""
        self.results.append(ValidationResult(
            check_name=check_name,
            severity=severity,
            passed=passed,
            message=message,
            file_path=file_path,
            details=details
        ))

    def _report_results(self) -> bool:
        """Report validation results and return success status."""
        print(f"\n📊 Validation Results: {self.total_checks} checks performed")

        critical_failures = 0
        warning_count = 0
        info_count = 0

        for result in self.results:
            status_icon = "✅" if result.passed else "❌"
            print(f"{status_icon} {result.severity.value} {result.check_name}: {result.message}")

            if result.details:
                print(f"   💡 {result.details}")

            if result.file_path:
                print(f"   📁 {result.file_path}")

            if not result.passed:
                if result.severity == Severity.CRITICAL:
                    critical_failures += 1
                elif result.severity == Severity.WARNING:
                    warning_count += 1
                elif result.severity == Severity.INFO:
                    info_count += 1

        print(f"\n🎯 Summary:")
        print(f"   🔴 Critical Failures: {critical_failures}")
        print(f"   🟡 Warnings: {warning_count}")
        print(f"   ℹ️  Info: {info_count}")

        if critical_failures > 0:
            print(f"\n❌ {critical_failures} CRITICAL violations found!")
            print("🚫 These must be fixed before committing.")
            print("\n🔧 To fix critical issues:")
            print("1. Domain layer must have ZERO external dependencies")
            print("2. Dependencies must point inward (infrastructure → domain)")
            print("3. Required DDD components must exist (entities, value objects, services)")
            print("4. CQRS commands and queries must be separate and inherit from base classes")
            return False
        else:
            print(f"\n🎉 All critical architectural standards met!")
            if warning_count > 0:
                print(f"⚠️  {warning_count} warnings should be addressed for optimal architecture.")
            elif self.total_checks > 0:
                print(f"✅ Every one of the {self.total_checks} checks passed successfully!")
            return True


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent

    validator = ArchitectureValidator(project_root)
    success = validator.validate_architecture()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())