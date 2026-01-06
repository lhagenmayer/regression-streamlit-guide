"""
Use Cases - Application layer business logic.

Use cases orchestrate domain objects to fulfill specific business goals.
They define what the application does from a business perspective.
"""

from typing import Dict, List, Optional, Protocol
from abc import ABC, abstractmethod

from ..domain.entities import Dataset, RegressionModel
from ..domain.services import RegressionAnalysisService
from ..domain.repositories import DatasetRepository
from ..domain.value_objects import DatasetConfig, RegressionParameters
from ..domain.events import DatasetCreated, RegressionModelCreated, RegressionModelValidated, ModelsCompared
from .event_handlers import publish_event


# Use Cases
class CreateDatasetUseCase:
    """Application service for dataset operations."""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def create_dataset(self, config: DatasetConfig, data: Dict[str, any]) -> Dataset:
        """Create a new dataset."""
        dataset = Dataset(
            id=f"dataset_{config.name.lower().replace(' ', '_')}",
            config=config,
            data=data
        )

        # Validate before saving
        issues = dataset.validate()
        if issues:
            raise ValueError(f"Invalid dataset: {issues}")

        self.dataset_repository.save(dataset)

        # Publish domain event
        publish_event(DatasetCreated(dataset=dataset))

        return dataset

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get dataset by ID."""
        return self.dataset_repository.find_by_id(dataset_id)

    def list_datasets(self) -> List[Dataset]:
        """List all available datasets."""
        return self.dataset_repository.list_all()


# Use Cases
class CreateRegressionModelUseCase:
    """
    Use Case: Create a regression model.

    This encapsulates the entire workflow of creating a regression model,
    from data validation to model fitting and quality assessment.
    """

    def __init__(
        self,
        dataset_service: CreateDatasetUseCase,
        regression_service: RegressionAnalysisService
    ):
        self.dataset_service = dataset_service
        self.regression_service = regression_service

    def execute(
        self,
        dataset_id: str,
        target_variable: str,
        feature_variables: List[str],
        parameters: RegressionParameters
    ) -> Dict[str, any]:
        """
        Execute the create regression model use case.

        Steps:
        1. Validate dataset exists and contains required variables
        2. Create and fit the model
        3. Analyze model quality
        4. Return comprehensive results
        """
        # Validate dataset
        dataset = self.dataset_service.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Validate variables exist
        if target_variable not in dataset.data:
            raise ValueError(f"Target variable '{target_variable}' not found in dataset")

        missing_features = [f for f in feature_variables if f not in dataset.data]
        if missing_features:
            raise ValueError(f"Feature variables not found: {missing_features}")

        # Create model
        model = self.regression_service.create_model(
            dataset_id, target_variable, feature_variables, parameters
        )

        # Publish model created event
        publish_event(RegressionModelCreated(model=model))

        # Analyze quality
        quality_report = self.regression_service.analyze_model_quality(model)

        # Publish validation event
        overall_quality = quality_report.get("overall_quality", False)
        publish_event(RegressionModelValidated(
            model_id=model.id,
            quality_score=model.metrics.adj_r_squared if overall_quality else 0.0,
            recommendations=quality_report.get("recommendations", [])
        ))

        # Generate insights
        insights = self._generate_insights(model, quality_report)

        return {
            "model": model,
            "quality_report": quality_report,
            "insights": insights,
            "dataset_info": {
                "id": dataset.id,
                "name": dataset.config.name,
                "n_observations": dataset.config.n_observations,
                "variables_used": [target_variable] + feature_variables
            }
        }

    def _generate_insights(self, model: RegressionModel, quality_report: Dict) -> List[str]:
        """Generate human-readable insights about the model."""
        insights = []

        metrics = model.metrics

        # R² insights
        if metrics.adj_r_squared > 0.8:
            insights.append("Excellent model fit - explains most of the variance")
        elif metrics.adj_r_squared > 0.6:
            insights.append("Good model fit with substantial explanatory power")
        elif metrics.adj_r_squared > 0.3:
            insights.append("Fair model fit - some explanatory power but room for improvement")
        else:
            insights.append("Poor model fit - consider different variables or transformations")

        # Feature importance
        significant_features = [
            name for name in model.feature_names
            if model.is_significant(name)
        ]

        if significant_features:
            insights.append(f"Significant predictors: {', '.join(significant_features)}")
        else:
            insights.append("No statistically significant predictors found")

        # Recommendations
        if quality_report["recommendations"]:
            insights.extend(quality_report["recommendations"])

        return insights


class AnalyzeModelQualityUseCase:
    """Use Case: Analyze regression model quality."""

    def __init__(self, regression_service: RegressionAnalysisService):
        self.regression_service = regression_service

    def execute(self, model: RegressionModel) -> Dict[str, any]:
        """Execute model quality analysis."""
        quality_report = self.regression_service.analyze_model_quality(model)

        # Add actionable recommendations
        recommendations = []
        if not quality_report["quality_checks"]["sufficient_r_squared"]:
            recommendations.extend([
                "Add interaction terms between variables",
                "Consider polynomial transformations",
                "Include additional relevant predictors"
            ])

        if not quality_report["quality_checks"]["significant_f_test"]:
            recommendations.append("Overall model is not statistically significant")

        return {
            "quality_report": quality_report,
            "recommendations": recommendations,
            "actionable_insights": self._create_actionable_insights(quality_report)
        }

    def _create_actionable_insights(self, quality_report: Dict) -> List[str]:
        """Create actionable insights for model improvement."""
        insights = []

        diagnostics = quality_report["diagnostics"]

        if diagnostics["residual_std"] > diagnostics["rmse"] * 2:
            insights.append("High residual variance suggests heteroskedasticity")

        if diagnostics["f_p_value"] > 0.1:
            insights.append("Consider removing non-significant terms")

        if diagnostics["adj_r_squared"] < 0.5:
            insights.append("Low explanatory power - explore additional variables")

        return insights


class CompareModelsUseCase:
    """Use Case: Compare two regression models."""

    def __init__(self, regression_service: RegressionAnalysisService):
        self.regression_service = regression_service

    def execute(self, model1: RegressionModel, model2: RegressionModel) -> Dict[str, any]:
        """Execute model comparison."""
        comparison = self.regression_service.compare_models(model1, model2)

        # Publish comparison event
        publish_event(ModelsCompared(
            model1_id=model1.id,
            model2_id=model2.id,
            winner=comparison.get("winner", "tie"),
            reason=comparison.get("reason", "Comparison completed")
        ))

        # Add business interpretation
        interpretation = self._interpret_comparison(comparison)

        return {
            "comparison": comparison,
            "interpretation": interpretation,
            "recommendation": self._make_recommendation(comparison)
        }

    def _interpret_comparison(self, comparison: Dict) -> str:
        """Provide human interpretation of model comparison."""
        winner = comparison["winner"]
        reason = comparison["reason"]
        r2_diff = comparison["r2_difference"]

        if winner == "tie":
            return f"Models perform similarly ({reason}). Consider the simpler model (Occam's razor)."
        else:
            model_num = "2" if winner == "model2" else "1"
            r2_desc = "better" if r2_diff > 0 else "worse"
            return f"Model {model_num} is preferred: {reason} (ΔR² = {r2_diff:.3f})"

    def _make_recommendation(self, comparison: Dict) -> str:
        """Make a specific recommendation based on comparison."""
        if comparison["winner"] == "model2":
            return "Use Model 2 for better predictive performance"
        elif comparison["winner"] == "model1":
            return "Use Model 1 for better parsimony"
        else:
            complexity_diff = comparison["complexity_difference"]
            if complexity_diff <= 0:
                return "Use Model 1 (simpler with equivalent performance)"
            else:
                return "Use Model 2 (better performance justifies added complexity)"


class LoadDatasetUseCase:
    """Use Case: Load a dataset for analysis."""

    def __init__(self, dataset_service: CreateDatasetUseCase):
        self.dataset_service = dataset_service

    def execute(self, dataset_id: str) -> Dict[str, any]:
        """Execute dataset loading."""
        dataset = self.dataset_service.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Validate dataset
        issues = dataset.validate()

        return {
            "dataset": dataset,
            "is_valid": len(issues) == 0,
            "validation_issues": issues,
            "summary": {
                "n_variables": len(dataset.data),
                "n_observations": dataset.config.n_observations,
                "variable_names": list(dataset.data.keys())
            }
        }


class GenerateSyntheticDataUseCase:
    """Use Case: Generate synthetic dataset."""

    def __init__(self, dataset_service: CreateDatasetUseCase):
        self.dataset_service = dataset_service

    def execute(
        self,
        config: DatasetConfig,
        parameters: Dict[str, any]
    ) -> Dict[str, any]:
        """Execute synthetic data generation."""
        # This would orchestrate data generation through domain services
        # For now, return a placeholder
        return {
            "config": config,
            "parameters": parameters,
            "status": "not_implemented",
            "message": "Synthetic data generation would be implemented here"
        }