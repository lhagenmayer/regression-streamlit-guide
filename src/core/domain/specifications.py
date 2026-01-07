"""
Specification Pattern - Composable Business Rules.

Das Specification Pattern ermöglicht die Kombination von Geschäftsregeln
zu komplexeren Regeln durch logische Operatoren (AND, OR, NOT).

Vorteile:
- Wiederverwendbare Geschäftsregeln
- Testbare einzelne Spezifikationen
- Lesbare Kombination von Regeln
- Saubere Trennung von Validierung und Logik
"""

from typing import TypeVar, Generic, List, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .result import ValidationResult, ValidationError


T = TypeVar('T')


class Specification(ABC, Generic[T]):
    """
    Abstract base class for specifications.

    Eine Spezifikation definiert eine Geschäftsregel, die auf ein Objekt
    angewendet werden kann.
    """

    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool:
        """Check if candidate satisfies this specification."""
        pass

    @abstractmethod
    def get_failure_message(self) -> str:
        """Get the failure message for this specification."""
        pass

    def get_error_code(self) -> str:
        """Get error code for this specification."""
        return self.__class__.__name__.upper()

    def validate(self, candidate: T) -> ValidationResult:
        """
        Validate candidate and return ValidationResult.

        Returns valid result if satisfied, invalid with error message otherwise.
        """
        if self.is_satisfied_by(candidate):
            return ValidationResult.valid()
        return ValidationResult.invalid(
            code=self.get_error_code(),
            message=self.get_failure_message()
        )

    def and_(self, other: 'Specification[T]') -> 'AndSpecification[T]':
        """Combine with another specification using AND."""
        return AndSpecification(self, other)

    def or_(self, other: 'Specification[T]') -> 'OrSpecification[T]':
        """Combine with another specification using OR."""
        return OrSpecification(self, other)

    def not_(self) -> 'NotSpecification[T]':
        """Negate this specification."""
        return NotSpecification(self)

    def __and__(self, other: 'Specification[T]') -> 'AndSpecification[T]':
        """Support & operator for AND combination."""
        return self.and_(other)

    def __or__(self, other: 'Specification[T]') -> 'OrSpecification[T]':
        """Support | operator for OR combination."""
        return self.or_(other)

    def __invert__(self) -> 'NotSpecification[T]':
        """Support ~ operator for NOT."""
        return self.not_()


@dataclass
class AndSpecification(Specification[T]):
    """Specification that requires both specs to be satisfied."""
    left: Specification[T]
    right: Specification[T]

    def is_satisfied_by(self, candidate: T) -> bool:
        return (self.left.is_satisfied_by(candidate) and
                self.right.is_satisfied_by(candidate))

    def get_failure_message(self) -> str:
        messages = []
        if not self.left.is_satisfied_by:
            messages.append(self.left.get_failure_message())
        if not self.right.is_satisfied_by:
            messages.append(self.right.get_failure_message())
        return " AND ".join(messages) if messages else "Both conditions must be satisfied"

    def validate(self, candidate: T) -> ValidationResult:
        """Validate both specifications and combine errors."""
        left_result = self.left.validate(candidate)
        right_result = self.right.validate(candidate)
        return ValidationResult.combine(left_result, right_result)


@dataclass
class OrSpecification(Specification[T]):
    """Specification that requires at least one spec to be satisfied."""
    left: Specification[T]
    right: Specification[T]

    def is_satisfied_by(self, candidate: T) -> bool:
        return (self.left.is_satisfied_by(candidate) or
                self.right.is_satisfied_by(candidate))

    def get_failure_message(self) -> str:
        return f"Either: {self.left.get_failure_message()} OR {self.right.get_failure_message()}"


@dataclass
class NotSpecification(Specification[T]):
    """Specification that negates another specification."""
    spec: Specification[T]

    def is_satisfied_by(self, candidate: T) -> bool:
        return not self.spec.is_satisfied_by(candidate)

    def get_failure_message(self) -> str:
        return f"NOT: {self.spec.get_failure_message()}"


# ============================================================================
# Dataset Specifications
# ============================================================================

class DatasetSpecification(Specification['Dataset']):
    """Base class for dataset specifications."""
    pass


class HasMinimumSampleSize(DatasetSpecification):
    """Specification: Dataset must have minimum sample size."""

    def __init__(self, min_size: int = 10):
        self.min_size = min_size

    def is_satisfied_by(self, dataset: 'Dataset') -> bool:
        return dataset.get_sample_size() >= self.min_size

    def get_failure_message(self) -> str:
        return f"Dataset must have at least {self.min_size} observations"


class HasRequiredVariables(DatasetSpecification):
    """Specification: Dataset must contain specific variables."""

    def __init__(self, required_variables: List[str]):
        self.required_variables = required_variables

    def is_satisfied_by(self, dataset: 'Dataset') -> bool:
        return all(dataset.has_variable(var) for var in self.required_variables)

    def get_failure_message(self) -> str:
        return f"Dataset must contain variables: {self.required_variables}"


class HasNoMissingValues(DatasetSpecification):
    """Specification: Dataset must not contain missing values."""

    def is_satisfied_by(self, dataset: 'Dataset') -> bool:
        return dataset._calculate_completeness() == 1.0

    def get_failure_message(self) -> str:
        return "Dataset must not contain missing values"


class HasSufficientVariation(DatasetSpecification):
    """Specification: All numeric variables must have variation."""

    def is_satisfied_by(self, dataset: 'Dataset') -> bool:
        for name, data in dataset.data.items():
            if data and len(set(data)) == 1:
                return False
        return True

    def get_failure_message(self) -> str:
        return "All variables must have some variation (no constant values)"


class IsReadyForAnalysis(DatasetSpecification):
    """
    Composite specification: Dataset is ready for statistical analysis.

    Combines multiple business rules:
    - Minimum sample size
    - No validation issues
    - Sufficient variation
    """

    def __init__(self, min_sample_size: int = 10):
        self.min_sample_size = min_sample_size

    def is_satisfied_by(self, dataset: 'Dataset') -> bool:
        has_min_size = HasMinimumSampleSize(self.min_sample_size)
        has_variation = HasSufficientVariation()
        no_missing = HasNoMissingValues()

        combined = has_min_size & has_variation
        return combined.is_satisfied_by(dataset)

    def get_failure_message(self) -> str:
        return "Dataset is not ready for analysis"


# ============================================================================
# Model Specifications
# ============================================================================

class ModelSpecification(Specification['RegressionModel']):
    """Base class for model specifications."""
    pass


class HasMinimumRSquared(ModelSpecification):
    """Specification: Model must have minimum R² value."""

    def __init__(self, min_r_squared: float = 0.1):
        self.min_r_squared = min_r_squared

    def is_satisfied_by(self, model: 'RegressionModel') -> bool:
        return model.metrics.adj_r_squared >= self.min_r_squared

    def get_failure_message(self) -> str:
        return f"Model must have R² >= {self.min_r_squared}"


class IsStatisticallySignificant(ModelSpecification):
    """Specification: Model must be statistically significant."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def is_satisfied_by(self, model: 'RegressionModel') -> bool:
        return model.metrics.f_p_value < self.alpha

    def get_failure_message(self) -> str:
        return f"Model must be statistically significant (p < {self.alpha})"


class HasReasonableComplexity(ModelSpecification):
    """Specification: Model must not be overly complex."""

    def __init__(self, max_features: int = 10):
        self.max_features = max_features

    def is_satisfied_by(self, model: 'RegressionModel') -> bool:
        return len(model.feature_names) <= self.max_features

    def get_failure_message(self) -> str:
        return f"Model must not have more than {self.max_features} features"


class IsProductionReady(ModelSpecification):
    """
    Composite specification: Model is ready for production deployment.

    Business rules:
    - Statistically significant
    - Reasonable R² value
    - Not overly complex
    """

    def __init__(
        self,
        min_r_squared: float = 0.3,
        alpha: float = 0.05,
        max_features: int = 10
    ):
        self.min_r_squared = min_r_squared
        self.alpha = alpha
        self.max_features = max_features

    def is_satisfied_by(self, model: 'RegressionModel') -> bool:
        significant = IsStatisticallySignificant(self.alpha)
        good_fit = HasMinimumRSquared(self.min_r_squared)
        reasonable = HasReasonableComplexity(self.max_features)

        combined = significant & good_fit & reasonable
        return combined.is_satisfied_by(model)

    def get_failure_message(self) -> str:
        return "Model does not meet production requirements"

    def validate(self, model: 'RegressionModel') -> ValidationResult:
        """Validate all production requirements."""
        result = ValidationResult.valid()

        significant = IsStatisticallySignificant(self.alpha)
        good_fit = HasMinimumRSquared(self.min_r_squared)
        reasonable = HasReasonableComplexity(self.max_features)

        result.merge(significant.validate(model))
        result.merge(good_fit.validate(model))
        result.merge(reasonable.validate(model))

        return result


# ============================================================================
# Specification Factory
# ============================================================================

class SpecificationFactory:
    """
    Factory for creating common specification combinations.

    Provides convenient methods for creating frequently used specifications.
    """

    @staticmethod
    def dataset_for_regression(
        target_variable: str,
        feature_variables: List[str],
        min_sample_size: int = 30
    ) -> DatasetSpecification:
        """
        Create specification for dataset suitable for regression analysis.

        Requires:
        - All specified variables present
        - Minimum sample size
        - No missing values
        - Sufficient variation
        """
        all_variables = [target_variable] + feature_variables
        return (
            HasRequiredVariables(all_variables) &
            HasMinimumSampleSize(min_sample_size) &
            HasSufficientVariation()
        )

    @staticmethod
    def model_for_production(
        min_r_squared: float = 0.3,
        significance_level: float = 0.05,
        max_features: int = 10
    ) -> ModelSpecification:
        """Create specification for production-ready model."""
        return IsProductionReady(
            min_r_squared=min_r_squared,
            alpha=significance_level,
            max_features=max_features
        )

    @staticmethod
    def model_for_exploration() -> ModelSpecification:
        """
        Create lenient specification for exploratory analysis.

        Only requires statistical significance.
        """
        return IsStatisticallySignificant(alpha=0.1)
