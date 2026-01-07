"""
Aggregate Root Pattern - DDD Aggregates.

Aggregates sind Cluster von Domain Objects, die als eine Einheit behandelt werden.
Der Aggregate Root ist der einzige Zugriffspunkt auf das Aggregate und
verantwortlich für die Konsistenz innerhalb des Aggregates.

Vorteile:
- Konsistenzgrenzen klar definiert
- Transaktionale Integrität
- Events werden am Aggregate gesammelt
- Versionierung für Optimistic Locking
"""

from typing import List, TypeVar, Generic, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .events import DomainEvent, EventMetadata
from .result import Result, ValidationResult, Error


T = TypeVar('T', bound='AggregateRoot')


@dataclass
class AggregateRoot(ABC):
    """
    Base class for Aggregate Roots.

    Ein Aggregate Root:
    - Ist der einzige Einstiegspunkt zum Aggregate
    - Garantiert Konsistenz innerhalb des Aggregates
    - Sammelt Domain Events
    - Hat eine Version für Optimistic Locking
    """
    id: str
    _version: int = field(default=0, repr=False)
    _events: List[DomainEvent] = field(default_factory=list, repr=False)

    @property
    def version(self) -> int:
        """Get current version for optimistic locking."""
        return self._version

    @property
    def uncommitted_events(self) -> List[DomainEvent]:
        """Get all uncommitted domain events."""
        return self._events.copy()

    def add_event(self, event: DomainEvent) -> None:
        """Add a domain event to be published after commit."""
        self._events.append(event)

    def clear_events(self) -> List[DomainEvent]:
        """Clear and return all uncommitted events."""
        events = self._events.copy()
        self._events.clear()
        return events

    def increment_version(self) -> None:
        """Increment version after successful persistence."""
        self._version += 1

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Validate aggregate consistency."""
        pass

    def is_valid(self) -> bool:
        """Check if aggregate is in valid state."""
        return self.validate().is_valid

    def _create_event_metadata(self) -> EventMetadata:
        """Create event metadata for this aggregate."""
        return EventMetadata.create(
            aggregate_id=self.id,
            aggregate_type=self.__class__.__name__
        )


# ============================================================================
# Dataset Aggregate
# ============================================================================

@dataclass
class DatasetAggregate(AggregateRoot):
    """
    Dataset Aggregate Root.

    Das Dataset Aggregate enthält:
    - Dataset Konfiguration (Value Object)
    - Die eigentlichen Daten
    - Metadata

    Invarianten:
    - Alle Variablen haben gleiche Länge
    - Keine leeren Variablennamen
    - Mindestens eine Variable
    """
    from .value_objects import DatasetConfig

    config: 'DatasetConfig' = None
    data: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate dataset aggregate consistency."""
        from .specifications import (
            HasMinimumSampleSize,
            HasSufficientVariation
        )

        result = ValidationResult.valid()

        # Check config exists
        if self.config is None:
            result.add_error("MISSING_CONFIG", "Dataset configuration is required")
            return result

        # Check data is not empty
        if not self.data:
            result.add_error("EMPTY_DATA", "Dataset must contain at least one variable")
            return result

        # Check variable lengths are consistent
        lengths = [len(v) for v in self.data.values()]
        if len(set(lengths)) > 1:
            result.add_error(
                "INCONSISTENT_LENGTHS",
                f"Variables have inconsistent lengths: {lengths}"
            )

        # Check no empty variable names
        if "" in self.data:
            result.add_error("EMPTY_VAR_NAME", "Variable names cannot be empty")

        return result

    def add_variable(self, name: str, values: List[float]) -> Result[None]:
        """
        Add a variable to the dataset.

        Returns Result indicating success or failure.
        """
        if not name or not name.strip():
            return Result.failure(Error("INVALID_NAME", "Variable name cannot be empty"))

        if name in self.data:
            return Result.failure(Error("DUPLICATE_NAME", f"Variable '{name}' already exists"))

        # Check length consistency
        if self.data:
            expected_length = len(next(iter(self.data.values())))
            if len(values) != expected_length:
                return Result.failure(Error(
                    "LENGTH_MISMATCH",
                    f"Variable length {len(values)} doesn't match existing {expected_length}"
                ))

        self.data[name] = list(values)  # Make a copy

        # Add event
        from .events import DatasetUpdated
        self.add_event(DatasetUpdated.create(
            dataset_id=self.id,
            changes={"added_variable": name, "length": len(values)}
        ))

        return Result.success(None)

    def remove_variable(self, name: str) -> Result[None]:
        """Remove a variable from the dataset."""
        if name not in self.data:
            return Result.failure(Error("NOT_FOUND", f"Variable '{name}' not found"))

        if len(self.data) == 1:
            return Result.failure(Error(
                "LAST_VARIABLE",
                "Cannot remove the last variable from dataset"
            ))

        del self.data[name]

        from .events import DatasetUpdated
        self.add_event(DatasetUpdated.create(
            dataset_id=self.id,
            changes={"removed_variable": name}
        ))

        return Result.success(None)

    def get_variable(self, name: str) -> Result[List[float]]:
        """Get variable data by name."""
        if name not in self.data:
            return Result.failure(Error("NOT_FOUND", f"Variable '{name}' not found"))
        return Result.success(self.data[name].copy())

    def get_sample_size(self) -> int:
        """Get number of observations."""
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))

    def get_variable_names(self) -> List[str]:
        """Get all variable names."""
        return list(self.data.keys())

    def has_variable(self, name: str) -> bool:
        """Check if variable exists."""
        return name in self.data

    def create_subset(self, variable_names: List[str]) -> Result['DatasetAggregate']:
        """Create a subset with selected variables."""
        missing = [n for n in variable_names if n not in self.data]
        if missing:
            return Result.failure(Error(
                "MISSING_VARIABLES",
                f"Variables not found: {missing}"
            ))

        subset_data = {name: self.data[name].copy() for name in variable_names}

        return Result.success(DatasetAggregate(
            id=f"{self.id}_subset",
            config=self.config,
            data=subset_data
        ))

    @classmethod
    def create(
        cls,
        id: str,
        config: 'DatasetConfig',
        data: Dict[str, List[float]]
    ) -> Result['DatasetAggregate']:
        """
        Factory method to create a valid Dataset Aggregate.

        Returns Result with the aggregate or validation errors.
        """
        aggregate = cls(id=id, config=config, data=data)

        validation = aggregate.validate()
        if validation.is_invalid:
            return validation.to_result(None)

        # Add creation event
        from .events import DatasetCreated
        aggregate.add_event(DatasetCreated(
            dataset_id=id,
            dataset_name=config.name,
            n_variables=len(data),
            n_observations=aggregate.get_sample_size(),
            source=config.source,
            _metadata=aggregate._create_event_metadata()
        ))

        return Result.success(aggregate)


# ============================================================================
# RegressionModel Aggregate
# ============================================================================

@dataclass
class RegressionModelAggregate(AggregateRoot):
    """
    Regression Model Aggregate Root.

    Das RegressionModel Aggregate enthält:
    - Model Parameter (Value Object)
    - Model Metrics (Value Object)
    - Fitted Values und Residuals

    Invarianten:
    - fitted_values und residuals haben gleiche Länge
    - Mindestens ein Feature
    - Gültige Metrics
    """
    from .value_objects import RegressionParameters, ModelMetrics

    dataset_id: str = ""
    model_type: str = "simple"
    parameters: 'RegressionParameters' = None
    metrics: 'ModelMetrics' = None
    feature_names: List[str] = field(default_factory=list)
    fitted_values: List[float] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        """Validate model aggregate consistency."""
        result = ValidationResult.valid()

        if self.parameters is None:
            result.add_error("MISSING_PARAMS", "Model parameters are required")

        if self.metrics is None:
            result.add_error("MISSING_METRICS", "Model metrics are required")

        if not self.feature_names:
            result.add_error("NO_FEATURES", "Model must have at least one feature")

        if len(self.fitted_values) != len(self.residuals):
            result.add_error(
                "LENGTH_MISMATCH",
                "Fitted values and residuals must have same length"
            )

        return result

    def get_coefficient(self, feature_name: str) -> Result[float]:
        """Get coefficient for a specific feature."""
        if feature_name not in self.feature_names:
            return Result.failure(Error(
                "FEATURE_NOT_FOUND",
                f"Feature '{feature_name}' not in model"
            ))

        coeff = self.parameters.coefficients.get(feature_name, 0.0)
        return Result.success(coeff)

    def predict(self, input_data: Dict[str, List[float]]) -> Result[List[float]]:
        """
        Make predictions using the model.

        Returns Result with predictions or error.
        """
        # Validate input
        missing = [f for f in self.feature_names if f not in input_data]
        if missing:
            return Result.failure(Error(
                "MISSING_FEATURES",
                f"Input missing features: {missing}"
            ))

        # Get prediction length
        first_values = input_data[self.feature_names[0]]
        n_predictions = len(first_values)

        # Check all inputs have same length
        for feature in self.feature_names:
            if len(input_data[feature]) != n_predictions:
                return Result.failure(Error(
                    "LENGTH_MISMATCH",
                    f"All input features must have same length"
                ))

        # Calculate predictions
        predictions = [self.parameters.intercept] * n_predictions

        for i in range(n_predictions):
            for feature_name in self.feature_names:
                coeff = self.parameters.coefficients.get(feature_name, 0)
                predictions[i] += coeff * input_data[feature_name][i]

        # Add prediction event
        from .events import ModelPredictionMade
        self.add_event(ModelPredictionMade.create(
            model_id=self.id,
            n_predictions=n_predictions,
            input_variables=self.feature_names
        ))

        return Result.success(predictions)

    def get_model_equation(self) -> str:
        """Get model equation as string."""
        terms = [f"{self.parameters.intercept:.4f}"]

        for name in self.feature_names:
            coeff = self.parameters.coefficients.get(name, 0)
            sign = "+" if coeff >= 0 else ""
            terms.append(f"{sign}{coeff:.4f}*{name}")

        return "y = " + " ".join(terms)

    def is_good_fit(self, min_r_squared: float = 0.5, alpha: float = 0.05) -> bool:
        """Business rule: Check if model has acceptable fit."""
        return (
            self.metrics.adj_r_squared >= min_r_squared and
            self.metrics.f_p_value < alpha
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Calculate normalized feature importance."""
        if not self.feature_names:
            return {}

        coefficients = {
            name: abs(self.parameters.coefficients.get(name, 0))
            for name in self.feature_names
        }

        max_coeff = max(coefficients.values()) if coefficients else 1.0
        if max_coeff == 0:
            return {name: 0.0 for name in self.feature_names}

        return {
            name: coeff / max_coeff
            for name, coeff in sorted(
                coefficients.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }

    @classmethod
    def create(
        cls,
        id: str,
        dataset_id: str,
        model_type: str,
        parameters: 'RegressionParameters',
        metrics: 'ModelMetrics',
        feature_names: List[str],
        fitted_values: List[float],
        residuals: List[float]
    ) -> Result['RegressionModelAggregate']:
        """
        Factory method to create a valid RegressionModel Aggregate.

        Returns Result with the aggregate or validation errors.
        """
        aggregate = cls(
            id=id,
            dataset_id=dataset_id,
            model_type=model_type,
            parameters=parameters,
            metrics=metrics,
            feature_names=list(feature_names),
            fitted_values=list(fitted_values),
            residuals=list(residuals)
        )

        validation = aggregate.validate()
        if validation.is_invalid:
            return validation.to_result(None)

        # Add creation event
        from .events import RegressionModelCreated
        aggregate.add_event(RegressionModelCreated(
            model_id=id,
            dataset_id=dataset_id,
            model_type=model_type,
            feature_names=tuple(feature_names),
            r_squared=metrics.r_squared,
            adj_r_squared=metrics.adj_r_squared,
            _metadata=aggregate._create_event_metadata()
        ))

        return Result.success(aggregate)
