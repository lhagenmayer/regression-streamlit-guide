"""
Factory Pattern - Objekt-Erstellung.

Factories kapseln die komplexe Logik zur Erstellung von Domain Objects.
Sie stellen sicher, dass Objekte immer in einem gültigen Zustand erstellt werden.

Vorteile:
- Konsistente Objekterstellung
- Validierung bei Erstellung
- Verstecken von Erstellungskomplexität
- Testbarkeit durch Factory-Austausch
"""

from typing import Dict, List, Any, Optional, Protocol, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uuid
import random
import math

from .value_objects import DatasetConfig, RegressionParameters, ModelMetrics, StatisticalSummary
from .entities import Dataset, RegressionModel
from .aggregates import DatasetAggregate, RegressionModelAggregate
from .result import Result, Error, ValidationResult


T = TypeVar('T')


# ============================================================================
# Value Object Factories
# ============================================================================

class DatasetConfigFactory:
    """Factory für DatasetConfig Value Objects."""

    @staticmethod
    def create_synthetic(
        name: str,
        description: str,
        variables: Dict[str, str],
        n_observations: int
    ) -> Result[DatasetConfig]:
        """Create config for synthetic dataset."""
        try:
            config = DatasetConfig(
                name=name,
                dataset_type="synthetic",
                source="generated",
                description=description,
                variables=variables,
                n_observations=n_observations,
                api_available=False
            )
            return Result.success(config)
        except ValueError as e:
            return Result.failure(Error("INVALID_CONFIG", str(e)))

    @staticmethod
    def create_from_api(
        name: str,
        source: str,
        api_docs: str,
        variables: Dict[str, str],
        n_observations: int,
        python_package: Optional[str] = None
    ) -> Result[DatasetConfig]:
        """Create config for API-sourced dataset."""
        try:
            config = DatasetConfig(
                name=name,
                dataset_type="api",
                source=source,
                description=f"Data from {source} API",
                variables=variables,
                n_observations=n_observations,
                api_available=True,
                python_package=python_package,
                api_docs=api_docs
            )
            return Result.success(config)
        except ValueError as e:
            return Result.failure(Error("INVALID_CONFIG", str(e)))

    @staticmethod
    def create_from_file(
        name: str,
        file_path: str,
        variables: Dict[str, str],
        n_observations: int
    ) -> Result[DatasetConfig]:
        """Create config for file-based dataset."""
        try:
            config = DatasetConfig(
                name=name,
                dataset_type="file",
                source=file_path,
                description=f"Data loaded from {file_path}",
                variables=variables,
                n_observations=n_observations,
                api_available=False
            )
            return Result.success(config)
        except ValueError as e:
            return Result.failure(Error("INVALID_CONFIG", str(e)))


class RegressionParametersFactory:
    """Factory für RegressionParameters Value Objects."""

    @staticmethod
    def create(
        intercept: float,
        coefficients: Dict[str, float],
        noise_level: float = 0.1,
        seed: int = 42,
        confidence_level: float = 0.95
    ) -> Result[RegressionParameters]:
        """Create regression parameters with validation."""
        try:
            params = RegressionParameters(
                intercept=intercept,
                coefficients=dict(coefficients),  # Copy to ensure immutability
                noise_level=noise_level,
                seed=seed,
                confidence_level=confidence_level
            )
            return Result.success(params)
        except ValueError as e:
            return Result.failure(Error("INVALID_PARAMS", str(e)))

    @staticmethod
    def create_for_simple_regression(
        intercept: float,
        slope: float,
        feature_name: str,
        noise_level: float = 0.1,
        seed: int = 42
    ) -> Result[RegressionParameters]:
        """Create parameters for simple regression."""
        return RegressionParametersFactory.create(
            intercept=intercept,
            coefficients={feature_name: slope},
            noise_level=noise_level,
            seed=seed
        )

    @staticmethod
    def create_random(
        feature_names: List[str],
        intercept_range: tuple = (-10, 10),
        coeff_range: tuple = (-5, 5),
        seed: int = 42
    ) -> Result[RegressionParameters]:
        """Create random regression parameters for testing."""
        random.seed(seed)

        intercept = random.uniform(*intercept_range)
        coefficients = {
            name: random.uniform(*coeff_range)
            for name in feature_names
        }

        return RegressionParametersFactory.create(
            intercept=intercept,
            coefficients=coefficients,
            seed=seed
        )


class ModelMetricsFactory:
    """Factory für ModelMetrics Value Objects."""

    @staticmethod
    def calculate_from_residuals(
        y_actual: List[float],
        y_predicted: List[float],
        n_features: int
    ) -> Result[ModelMetrics]:
        """Calculate metrics from actual and predicted values."""
        if len(y_actual) != len(y_predicted):
            return Result.failure(Error(
                "LENGTH_MISMATCH",
                "Actual and predicted must have same length"
            ))

        n = len(y_actual)
        if n == 0:
            return Result.failure(Error("EMPTY_DATA", "No data points"))

        p = n_features + 1  # +1 for intercept

        # Calculate residuals
        residuals = [a - p for a, p in zip(y_actual, y_predicted)]

        # Sum of squares
        ss_res = sum(r ** 2 for r in residuals)
        y_mean = sum(y_actual) / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_actual)

        # Metrics
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p) if n > p else r_squared
        mse = ss_res / (n - p) if n > p else ss_res / n
        rmse = math.sqrt(mse) if mse > 0 else 0
        mae = sum(abs(r) for r in residuals) / n

        # F-statistic
        if ss_res > 0 and n > p and p > 1:
            ms_model = (ss_tot - ss_res) / (p - 1)
            ms_res = ss_res / (n - p)
            f_statistic = ms_model / ms_res if ms_res > 0 else 0
        else:
            f_statistic = 0

        # Simplified p-value (would use F-distribution in real implementation)
        f_p_value = 0.01 if f_statistic > 4 else 0.1

        try:
            metrics = ModelMetrics(
                r_squared=r_squared,
                adj_r_squared=adj_r_squared,
                mse=mse,
                rmse=rmse,
                mae=mae,
                f_statistic=f_statistic,
                f_p_value=f_p_value
            )
            return Result.success(metrics)
        except ValueError as e:
            return Result.failure(Error("INVALID_METRICS", str(e)))


# ============================================================================
# Entity Factories
# ============================================================================

class DatasetFactory:
    """
    Factory für Dataset Entities.

    Erstellt Datasets mit validierter Konfiguration und Daten.
    """

    @staticmethod
    def generate_id(name: str) -> str:
        """Generate unique dataset ID."""
        base = name.lower().replace(" ", "_")
        unique = str(uuid.uuid4())[:8]
        return f"dataset_{base}_{unique}"

    @classmethod
    def create(
        cls,
        name: str,
        dataset_type: str,
        source: str,
        description: str,
        data: Dict[str, List[float]],
        variables: Optional[Dict[str, str]] = None
    ) -> Result[Dataset]:
        """
        Create a new Dataset entity.

        Validates configuration and data before creation.
        """
        if not data:
            return Result.failure(Error("EMPTY_DATA", "Dataset must have data"))

        # Infer variables if not provided
        if variables is None:
            variables = {name: "numeric" for name in data.keys()}

        # Get observation count
        n_observations = len(next(iter(data.values())))

        # Create config
        try:
            config = DatasetConfig(
                name=name,
                dataset_type=dataset_type,
                source=source,
                description=description,
                variables=variables,
                n_observations=n_observations
            )
        except ValueError as e:
            return Result.failure(Error("INVALID_CONFIG", str(e)))

        # Create dataset
        dataset = Dataset(
            id=cls.generate_id(name),
            config=config,
            data=dict(data)  # Copy
        )

        # Validate
        issues = dataset.validate()
        if issues:
            # Filter warnings from errors
            errors = [i for i in issues if "small sample" not in i.lower()]
            if errors:
                return Result.failure(Error("VALIDATION_ERROR", "; ".join(errors)))

        return Result.success(dataset)

    @classmethod
    def create_synthetic(
        cls,
        name: str,
        n_observations: int,
        features: List[str],
        target: str = "y",
        intercept: float = 0.0,
        coefficients: Optional[Dict[str, float]] = None,
        noise_std: float = 1.0,
        seed: int = 42
    ) -> Result[Dataset]:
        """
        Create a synthetic dataset for regression analysis.

        Generates X values and calculates y = intercept + sum(coeff * x) + noise.
        """
        random.seed(seed)

        if coefficients is None:
            coefficients = {f: random.uniform(-5, 5) for f in features}

        # Generate feature data
        data = {}
        for feature in features:
            data[feature] = [random.gauss(0, 10) for _ in range(n_observations)]

        # Generate target
        y = [intercept] * n_observations
        for i in range(n_observations):
            for feature in features:
                coeff = coefficients.get(feature, 0)
                y[i] += coeff * data[feature][i]
            y[i] += random.gauss(0, noise_std)

        data[target] = y

        return cls.create(
            name=name,
            dataset_type="synthetic",
            source="generated",
            description=f"Synthetic data with {len(features)} features",
            data=data
        )


class RegressionModelFactory:
    """
    Factory für RegressionModel Entities.

    Erstellt und fittet Regressionsmodelle.
    """

    @staticmethod
    def generate_id() -> str:
        """Generate unique model ID."""
        return f"model_{str(uuid.uuid4())[:8]}"

    @classmethod
    def create(
        cls,
        dataset: Dataset,
        target_variable: str,
        feature_variables: List[str],
        parameters: RegressionParameters
    ) -> Result[RegressionModel]:
        """
        Create and fit a regression model.

        Validates inputs and calculates fitted values and metrics.
        """
        # Validate target variable
        if target_variable not in dataset.data:
            return Result.failure(Error(
                "TARGET_NOT_FOUND",
                f"Target '{target_variable}' not in dataset"
            ))

        # Validate feature variables
        missing = [f for f in feature_variables if f not in dataset.data]
        if missing:
            return Result.failure(Error(
                "FEATURES_NOT_FOUND",
                f"Features not found: {missing}"
            ))

        # Get data
        y = dataset.get_variable(target_variable)
        X_data = {f: dataset.get_variable(f) for f in feature_variables}

        # Calculate fitted values
        fitted, residuals = cls._fit_model(y, X_data, feature_variables, parameters)

        # Calculate metrics
        metrics_result = ModelMetricsFactory.calculate_from_residuals(
            y, fitted, len(feature_variables)
        )

        if metrics_result.is_failure:
            return Result.failure(metrics_result.error)

        # Create model
        model = RegressionModel(
            id=cls.generate_id(),
            dataset_id=dataset.id,
            model_type="multiple" if len(feature_variables) > 1 else "simple",
            parameters=parameters,
            fitted_values=fitted,
            residuals=residuals,
            metrics=metrics_result.value,
            feature_names=list(feature_variables)
        )

        return Result.success(model)

    @staticmethod
    def _fit_model(
        y: List[float],
        X_data: Dict[str, List[float]],
        feature_names: List[str],
        parameters: RegressionParameters
    ) -> tuple:
        """
        Fit model and return (fitted_values, residuals).

        Uses provided parameters for coefficients.
        """
        n = len(y)
        random.seed(parameters.seed)

        # Calculate fitted values
        fitted = [parameters.intercept] * n

        for i in range(n):
            for feature_name in feature_names:
                coeff = parameters.coefficients.get(feature_name, 0)
                fitted[i] += coeff * X_data[feature_name][i]

        # Add noise
        fitted = [f + random.gauss(0, parameters.noise_level) for f in fitted]

        # Calculate residuals
        residuals = [y[i] - fitted[i] for i in range(n)]

        return fitted, residuals

    @classmethod
    def create_from_ols(
        cls,
        dataset: Dataset,
        target_variable: str,
        feature_variables: List[str],
        confidence_level: float = 0.95
    ) -> Result[RegressionModel]:
        """
        Create model using OLS (Ordinary Least Squares).

        Estimates coefficients from data instead of using provided parameters.
        This is a simplified implementation - real implementation would use
        numpy/statsmodels for proper OLS.
        """
        # Validate variables
        if target_variable not in dataset.data:
            return Result.failure(Error(
                "TARGET_NOT_FOUND",
                f"Target '{target_variable}' not in dataset"
            ))

        missing = [f for f in feature_variables if f not in dataset.data]
        if missing:
            return Result.failure(Error(
                "FEATURES_NOT_FOUND",
                f"Features not found: {missing}"
            ))

        # Get data
        y = dataset.get_variable(target_variable)
        X_data = {f: dataset.get_variable(f) for f in feature_variables}

        n = len(y)

        # Simple OLS for single feature (for demonstration)
        if len(feature_variables) == 1:
            feature = feature_variables[0]
            x = X_data[feature]

            x_mean = sum(x) / n
            y_mean = sum(y) / n

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            coefficients = {feature: slope}
        else:
            # For multiple regression, use simplified approximation
            # Real implementation would use matrix operations
            y_mean = sum(y) / n
            intercept = y_mean

            coefficients = {}
            for feature in feature_variables:
                x = X_data[feature]
                x_mean = sum(x) / n

                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

                coeff = numerator / denominator if denominator != 0 else 0
                coefficients[feature] = coeff / len(feature_variables)  # Simplified

        # Create parameters
        params_result = RegressionParametersFactory.create(
            intercept=intercept,
            coefficients=coefficients,
            noise_level=0.0,  # No added noise for OLS
            confidence_level=confidence_level
        )

        if params_result.is_failure:
            return Result.failure(params_result.error)

        return cls.create(
            dataset=dataset,
            target_variable=target_variable,
            feature_variables=feature_variables,
            parameters=params_result.value
        )


# ============================================================================
# Aggregate Factories
# ============================================================================

class DatasetAggregateFactory:
    """Factory für DatasetAggregate."""

    @staticmethod
    def create(
        name: str,
        dataset_type: str,
        source: str,
        description: str,
        data: Dict[str, List[float]],
        variables: Optional[Dict[str, str]] = None
    ) -> Result[DatasetAggregate]:
        """Create a new DatasetAggregate."""
        if not data:
            return Result.failure(Error("EMPTY_DATA", "Dataset must have data"))

        if variables is None:
            variables = {name: "numeric" for name in data.keys()}

        n_observations = len(next(iter(data.values())))

        try:
            config = DatasetConfig(
                name=name,
                dataset_type=dataset_type,
                source=source,
                description=description,
                variables=variables,
                n_observations=n_observations
            )
        except ValueError as e:
            return Result.failure(Error("INVALID_CONFIG", str(e)))

        return DatasetAggregate.create(
            id=f"dataset_{name.lower().replace(' ', '_')}_{str(uuid.uuid4())[:8]}",
            config=config,
            data=dict(data)
        )


class RegressionModelAggregateFactory:
    """Factory für RegressionModelAggregate."""

    @classmethod
    def create(
        cls,
        dataset: DatasetAggregate,
        target_variable: str,
        feature_variables: List[str],
        parameters: RegressionParameters
    ) -> Result[RegressionModelAggregate]:
        """Create a new RegressionModelAggregate."""
        # Validate target
        if not dataset.has_variable(target_variable):
            return Result.failure(Error(
                "TARGET_NOT_FOUND",
                f"Target '{target_variable}' not in dataset"
            ))

        # Validate features
        missing = [f for f in feature_variables if not dataset.has_variable(f)]
        if missing:
            return Result.failure(Error(
                "FEATURES_NOT_FOUND",
                f"Features not found: {missing}"
            ))

        # Get data
        y_result = dataset.get_variable(target_variable)
        if y_result.is_failure:
            return Result.failure(y_result.error)
        y = y_result.value

        X_data = {}
        for feature in feature_variables:
            x_result = dataset.get_variable(feature)
            if x_result.is_failure:
                return Result.failure(x_result.error)
            X_data[feature] = x_result.value

        # Fit model
        fitted, residuals = cls._fit_model(y, X_data, feature_variables, parameters)

        # Calculate metrics
        metrics_result = ModelMetricsFactory.calculate_from_residuals(
            y, fitted, len(feature_variables)
        )

        if metrics_result.is_failure:
            return Result.failure(metrics_result.error)

        return RegressionModelAggregate.create(
            id=f"model_{str(uuid.uuid4())[:8]}",
            dataset_id=dataset.id,
            model_type="multiple" if len(feature_variables) > 1 else "simple",
            parameters=parameters,
            metrics=metrics_result.value,
            feature_names=list(feature_variables),
            fitted_values=fitted,
            residuals=residuals
        )

    @staticmethod
    def _fit_model(
        y: List[float],
        X_data: Dict[str, List[float]],
        feature_names: List[str],
        parameters: RegressionParameters
    ) -> tuple:
        """Fit model and return (fitted_values, residuals)."""
        n = len(y)
        random.seed(parameters.seed)

        fitted = [parameters.intercept] * n

        for i in range(n):
            for feature_name in feature_names:
                coeff = parameters.coefficients.get(feature_name, 0)
                fitted[i] += coeff * X_data[feature_name][i]

        fitted = [f + random.gauss(0, parameters.noise_level) for f in fitted]
        residuals = [y[i] - fitted[i] for i in range(n)]

        return fitted, residuals
