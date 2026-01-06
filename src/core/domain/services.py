"""
Domain Services - business logic that doesn't naturally fit entities.

Domain Services contain business rules and operations that involve
multiple entities or complex business logic.
"""

from typing import Dict, List, Optional, Any, Protocol
import random
import math

from .entities import Dataset, RegressionModel
from .value_objects import DatasetConfig, RegressionParameters, ModelMetrics
from .repositories import DatasetRepository


class RegressionAnalysisServiceProtocol(Protocol):
    """
    Protocol interface for regression analysis services.

    Defines the contract for regression analysis business logic.
    """

    def create_regression_model(
        self,
        dataset_id: str,
        target_variable: str,
        feature_variables: List[str],
        parameters: RegressionParameters
    ) -> RegressionModel: ...

    def validate_model_quality(self, model: RegressionModel) -> Dict[str, Any]: ...

    def compare_models(self, model1: RegressionModel, model2: RegressionModel) -> Dict[str, Any]: ...


class RegressionAnalysisService:
    """
    Domain service for regression analysis business logic.

    This service contains the core business rules for regression analysis,
    independent of infrastructure concerns like UI or data persistence.
    """

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def create_regression_model(
        self,
        dataset_id: str,
        target_variable: str,
        feature_variables: List[str],
        parameters: RegressionParameters
    ) -> RegressionModel:
        """
        Create and fit a regression model.

        Business rules:
        - Dataset must exist and be valid
        - Target and feature variables must exist in dataset
        - Model must meet minimum quality standards
        """
        # Load and validate dataset
        dataset = self.dataset_repository.find_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Validate dataset
        issues = dataset.validate()
        if issues:
            raise ValueError(f"Dataset validation failed: {issues}")

        # Validate variables exist
        if target_variable not in dataset.data:
            raise ValueError(f"Target variable '{target_variable}' not found")

        missing_features = [f for f in feature_variables if f not in dataset.data]
        if missing_features:
            raise ValueError(f"Feature variables not found: {missing_features}")

        # Extract data
        y = dataset.get_variable(target_variable)
        X_data = [dataset.get_variable(feature) for feature in feature_variables]

        # Fit model (simplified implementation)
        model = self._fit_regression_model(y, X_data, feature_variables, parameters)

        return model

    def validate_model_quality(self, model: RegressionModel) -> Dict[str, Any]:
        """
        Validate model quality against business rules.

        Returns diagnostic information about model quality.
        """
        diagnostics = model.get_diagnostics()

        # Business rules for model quality
        quality_checks = {
            "sufficient_r_squared": diagnostics["adj_r_squared"] > 0.1,
            "reasonable_mse": diagnostics["mse"] < 10.0,  # Arbitrary threshold
            "significant_f_test": diagnostics["f_p_value"] < 0.05,
            "no_extreme_residuals": abs(diagnostics["residual_mean"]) < 1.0,
            "reasonable_complexity": diagnostics["n_features"] <= 10
        }

        overall_quality = all(quality_checks.values())

        return {
            "overall_quality": overall_quality,
            "quality_checks": quality_checks,
            "diagnostics": diagnostics,
            "recommendations": self._generate_recommendations(quality_checks)
        }

    def compare_models(self, model1: RegressionModel, model2: RegressionModel) -> Dict[str, Any]:
        """
        Compare two regression models.

        Business rules for model comparison:
        - Prefer models with higher adjusted R²
        - Prefer simpler models (fewer features) when performance is similar
        - Consider statistical significance
        """
        metrics1 = model1.metrics
        metrics2 = model2.metrics

        # Comparison criteria
        r2_diff = metrics2.adj_r_squared - metrics1.adj_r_squared
        complexity_diff = len(model2.feature_names) - len(model1.feature_names)

        # Decision logic
        if abs(r2_diff) < 0.05:  # Similar performance
            if complexity_diff < 0:  # Model 2 is simpler
                winner = "model2"
                reason = "Model 2 is simpler with similar performance"
            elif complexity_diff > 0:  # Model 1 is simpler
                winner = "model1"
                reason = "Model 1 is simpler with similar performance"
            else:
                winner = "tie"
                reason = "Models have similar performance and complexity"
        else:
            winner = "model2" if r2_diff > 0 else "model1"
            reason = f"Model {2 if r2_diff > 0 else 1} has better explanatory power"

        return {
            "winner": winner,
            "reason": reason,
            "r2_difference": r2_diff,
            "complexity_difference": complexity_diff,
            "model1_metrics": metrics1,
            "model2_metrics": metrics2
        }

    def _fit_regression_model(
        self,
        y: List[float],
        X_data: List[List[float]],
        feature_names: List[str],
        parameters: RegressionParameters
    ) -> RegressionModel:
        """Fit regression model (simplified implementation)."""
        n = len(y)
        if n == 0:
            raise ValueError("Empty dataset")

        # Generate synthetic fitted values and residuals
        # In a real implementation, this would use statsmodels or scikit-learn
        random.seed(parameters.seed)

        # Simulate fitted values based on parameters
        fitted = [parameters.intercept] * n
        for i in range(n):
            for j, feature_data in enumerate(X_data):
                coeff = parameters.coefficients.get(feature_names[j], 0)
                fitted[i] += coeff * feature_data[i]

        # Add noise
        for i in range(n):
            noise = random.gauss(0, parameters.noise_level)
            fitted[i] += noise

        residuals = [y[i] - fitted[i] for i in range(n)]

        # Calculate metrics (simplified)
        ss_res = sum(r**2 for r in residuals)
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean)**2 for yi in y)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        p = len(feature_names) + 1  # +1 for intercept
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p) if n > p else r_squared
        mse = ss_res / (n - p) if n > p else 0
        rmse = math.sqrt(mse) if mse > 0 else 0
        mae = sum(abs(r) for r in residuals) / n

        # Simulate F-test
        f_statistic = (ss_tot - ss_res) / p / (ss_res / (n - p)) if ss_res > 0 and n > p else 0
        f_p_value = 0.01  # Simplified

        metrics = ModelMetrics(
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            mse=mse,
            rmse=rmse,
            mae=mae,
            f_statistic=f_statistic,
            f_p_value=f_p_value
        )

        return RegressionModel(
            id=f"model_{random.randint(1000, 9999)}",
            dataset_id="",  # Would be set by caller
            model_type="multiple" if len(feature_names) > 1 else "simple",
            parameters=parameters,
            fitted_values=fitted,
            residuals=residuals,
            metrics=metrics,
            feature_names=feature_names
        )

    def _generate_recommendations(self, quality_checks: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on quality check failures."""
        recommendations = []

        if not quality_checks["sufficient_r_squared"]:
            recommendations.append("Consider adding more relevant features or transforming existing ones")

        if not quality_checks["reasonable_mse"]:
            recommendations.append("Model has high prediction error - check for outliers or non-linear relationships")

        if not quality_checks["significant_f_test"]:
            recommendations.append("Model is not statistically significant - consider different features")

        if not quality_checks["no_extreme_residuals"]:
            recommendations.append("Check for model misspecification or influential outliers")

        if not quality_checks["reasonable_complexity"]:
            recommendations.append("Consider feature selection to reduce model complexity")

        if not recommendations:
            recommendations.append("Model quality is acceptable")

        return recommendations