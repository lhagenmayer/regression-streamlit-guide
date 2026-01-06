"""
Entities - domain objects with identity.

Entities have identity and can change over time. They represent
core business objects like RegressionModel and Dataset.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

from .value_objects import DatasetConfig, RegressionParameters, StatisticalSummary, ModelMetrics


@dataclass
class Dataset:
    """Dataset entity - represents a collection of data for analysis."""
    id: str
    config: DatasetConfig
    data: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_variable(self, name: str, data: List[float], metadata: Optional[Dict] = None):
        """Add a variable to the dataset."""
        self.data[name] = data
        if metadata:
            self.metadata[name] = metadata

    def get_variable(self, name: str) -> List[float]:
        """Get a variable by name."""
        if name not in self.data:
            raise ValueError(f"Variable '{name}' not found in dataset")
        return self.data[name]

    def get_summary(self, variable_name: str) -> StatisticalSummary:
        """Get statistical summary for a variable."""
        data = self.get_variable(variable_name)
        return StatisticalSummary.from_list(data)

    def validate(self) -> List[str]:
        """Validate the dataset and return list of issues."""
        issues = []

        if not self.data:
            issues.append("Dataset contains no variables")
            return issues

        # Check that all variables have same length
        lengths = [len(data) for data in self.data.values()]
        if len(set(lengths)) > 1:
            issues.append("Variables have inconsistent lengths")

        # Check for missing values (None or float('nan'))
        for name, data in self.data.items():
            if any(x is None or (isinstance(x, float) and str(x) == 'nan') for x in data):
                issues.append(f"Variable '{name}' contains missing values")

        # Check for reasonable data ranges
        for name, data in self.data.items():
            if len(data) > 0:
                numeric_data = [x for x in data if isinstance(x, (int, float)) and not (isinstance(x, float) and str(x) == 'nan')]
                if numeric_data:
                    data_range = max(numeric_data) - min(numeric_data)
                    if data_range == 0:
                        issues.append(f"Variable '{name}' has no variation (constant value)")
                    elif data_range > 1e10:  # Extremely large range
                        issues.append(f"Variable '{name}' has extremely large value range")

        # Check for minimum sample size
        n_obs = lengths[0] if lengths else 0
        if n_obs < 5:
            issues.append(f"Dataset has very small sample size ({n_obs} observations)")
        elif n_obs < 30:
            issues.append(f"Dataset has small sample size ({n_obs} observations) - results may be unreliable")

        return issues

    def get_variable_names(self) -> List[str]:
        """Get list of all variable names."""
        return list(self.data.keys())

    def get_sample_size(self) -> int:
        """Get the sample size (number of observations)."""
        if not self.data:
            return 0
        first_var = next(iter(self.data.values()))
        return len(first_var)

    def has_variable(self, name: str) -> bool:
        """Check if dataset contains a specific variable."""
        return name in self.data

    def get_variable_types(self) -> Dict[str, str]:
        """Infer variable types based on data."""
        types = {}
        for name, data in self.data.items():
            if not data:
                types[name] = "empty"
            elif all(isinstance(x, (int, float)) and not (isinstance(x, float) and str(x) == 'nan') for x in data[:10]):
                # Check if all are integers
                if all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in data[:10]):
                    types[name] = "integer"
                else:
                    types[name] = "numeric"
            else:
                types[name] = "categorical"
        return types

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        return {
            "sample_size": self.get_sample_size(),
            "n_variables": len(self.data),
            "variable_names": self.get_variable_names(),
            "variable_types": self.get_variable_types(),
            "validation_issues": self.validate(),
            "completeness": self._calculate_completeness(),
            "summary_stats": {name: self.get_summary(name) for name in self.data.keys()}
        }

    def is_ready_for_analysis(self) -> bool:
        """Business rule: Check if dataset is ready for statistical analysis."""
        issues = self.validate()
        return len(issues) == 0 and self.get_sample_size() >= 10

    def get_analysis_recommendations(self) -> List[str]:
        """Generate analysis recommendations based on data characteristics."""
        recommendations = []

        sample_size = self.get_sample_size()
        if sample_size < 30:
            recommendations.append("Consider collecting more data for reliable statistical inference")
        elif sample_size > 1000:
            recommendations.append("Large dataset - consider sampling for faster analysis")

        var_types = self.get_variable_types()
        numeric_vars = [name for name, vtype in var_types.items() if vtype in ['numeric', 'integer']]
        categorical_vars = [name for name, vtype in var_types.items() if vtype == 'categorical']

        if len(numeric_vars) > len(categorical_vars):
            recommendations.append("Primarily numeric data - suitable for regression analysis")
        elif len(categorical_vars) > 0:
            recommendations.append("Contains categorical variables - consider dummy encoding")

        completeness = self._calculate_completeness()
        if completeness < 0.95:
            recommendations.append(".1%")

        return recommendations

    def create_subset(self, variable_names: List[str]) -> 'Dataset':
        """Create a subset dataset with selected variables."""
        if not all(name in self.data for name in variable_names):
            missing = [name for name in variable_names if name not in self.data]
            raise ValueError(f"Variables not found: {missing}")

        subset_data = {name: self.data[name].copy() for name in variable_names}
        subset_config = self.config

        return Dataset(
            id=f"{self.id}_subset",
            config=subset_config,
            data=subset_data
        )

    def merge_with(self, other: 'Dataset') -> 'Dataset':
        """Merge this dataset with another dataset."""
        if self.get_sample_size() != other.get_sample_size():
            raise ValueError("Datasets must have same sample size for merging")

        merged_data = self.data.copy()
        for name, data in other.data.items():
            if name in merged_data:
                raise ValueError(f"Variable '{name}' already exists in dataset")
            merged_data[name] = data.copy()

        return Dataset(
            id=f"{self.id}_merged_{other.id}",
            config=self.config,  # Keep original config
            data=merged_data
        )

    def _calculate_completeness(self) -> float:
        """Calculate data completeness percentage."""
        if not self.data:
            return 0.0

        total_cells = sum(len(data) for data in self.data.values())
        if total_cells == 0:
            return 0.0

        missing_cells = sum(
            1 for data in self.data.values()
            for x in data
            if x is None or (isinstance(x, float) and str(x) == 'nan')
        )

        return (total_cells - missing_cells) / total_cells

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for numeric variables."""
        numeric_vars = [name for name, vtype in self.get_variable_types().items()
                       if vtype in ['numeric', 'integer']]

        if len(numeric_vars) < 2:
            return {}

        correlations = {}
        for i, var1 in enumerate(numeric_vars):
            correlations[var1] = {}
            for var2 in numeric_vars[i:]:
                if var1 == var2:
                    correlations[var1][var2] = 1.0
                else:
                    data1 = self.get_variable(var1)
                    data2 = self.get_variable(var2)
                    # Simple correlation calculation
                    n = len(data1)
                    mean1 = sum(data1) / n
                    mean2 = sum(data2) / n

                    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(data1, data2))
                    denom1 = sum((x - mean1) ** 2 for x in data1) ** 0.5
                    denom2 = sum((y - mean2) ** 2 for y in data2) ** 0.5

                    corr = numerator / (denom1 * denom2) if denom1 * denom2 != 0 else 0
                    correlations[var1][var2] = corr
                    correlations[var2] = correlations.get(var2, {})
                    correlations[var2][var1] = corr

        return correlations

    def detect_outliers(self, variable_name: str, method: str = "iqr") -> List[int]:
        """Detect outlier indices using specified method."""
        data = self.get_variable(variable_name)
        if not data:
            return []

        if method == "iqr":
            # IQR method
            sorted_data = sorted(data)
            n = len(sorted_data)
            q1_idx = int(n * 0.25)
            q3_idx = int(n * 0.75)
            q1 = sorted_data[q1_idx]
            q3 = sorted_data[q3_idx]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        else:
            # Z-score method (simplified)
            mean_val = sum(data) / len(data)
            std_val = (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
            outliers = [i for i, x in enumerate(data) if abs((x - mean_val) / std_val) > 3]

        return outliers

    def normalize_variable(self, variable_name: str, method: str = "zscore") -> List[float]:
        """Normalize a variable using specified method."""
        data = self.get_variable(variable_name)
        if not data:
            return []

        if method == "zscore":
            mean_val = sum(data) / len(data)
            std_val = (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
            if std_val == 0:
                return [0.0] * len(data)
            return [(x - mean_val) / std_val for x in data]

        elif method == "minmax":
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                return [0.0] * len(data)
            return [(x - min_val) / (max_val - min_val) for x in data]

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def split_train_test(self, test_ratio: float = 0.2, random_seed: int = 42) -> tuple:
        """Split dataset into training and test sets."""
        import random
        random.seed(random_seed)

        n_total = self.get_sample_size()
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_test

        indices = list(range(n_total))
        random.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        def split_data(data_list, train_idx, test_idx):
            return ([data_list[i] for i in train_idx], [data_list[i] for i in test_idx])

        train_data = {}
        test_data = {}

        for var_name, var_data in self.data.items():
            train_split, test_split = split_data(var_data, train_indices, test_indices)
            train_data[var_name] = train_split
            test_data[var_name] = test_split

        train_dataset = Dataset(
            id=f"{self.id}_train",
            config=self.config,
            data=train_data
        )

        test_dataset = Dataset(
            id=f"{self.id}_test",
            config=self.config,
            data=test_data
        )

        return train_dataset, test_dataset


@dataclass
class RegressionModel:
    """Regression model entity - represents a fitted regression model."""
    id: str
    dataset_id: str
    model_type: str  # 'simple', 'multiple'
    parameters: RegressionParameters
    fitted_values: List[float]
    residuals: List[float]
    metrics: ModelMetrics
    feature_names: List[str]

    def predict(self, new_data: Dict[str, List[float]]) -> List[float]:
        """Make predictions using the fitted model."""
        # Simple implementation - in reality would use statsmodels
        if not new_data:
            return []

        # Get length from first feature data
        first_feature = next(iter(new_data.values()))
        n_predictions = len(first_feature)

        predictions = [self.parameters.intercept] * n_predictions

        for i in range(n_predictions):
            for feature_name in self.feature_names:
                if feature_name in new_data:
                    coeff = self.parameters.coefficients.get(feature_name, 0)
                    feature_value = new_data[feature_name][i] if i < len(new_data[feature_name]) else 0
                    predictions[i] += coeff * feature_value

        # Add noise for realism (simplified)
        import random
        random.seed(self.parameters.seed)
        for i in range(len(predictions)):
            noise = random.gauss(0, self.parameters.noise_level)
            predictions[i] += noise

        return predictions

    def get_coefficient(self, feature_name: str) -> float:
        """Get coefficient for a specific feature."""
        return self.parameters.coefficients.get(feature_name, 0)

    def is_significant(self, feature_name: str, alpha: float = 0.05) -> bool:
        """Check if a feature coefficient is statistically significant."""
        # Simplified - in reality would check t-statistics
        coeff = abs(self.get_coefficient(feature_name))
        return coeff > 0.1  # Arbitrary threshold for demo

    def get_model_equation(self) -> str:
        """Get the model equation as a string."""
        terms = [f"{self.parameters.intercept:.3f}"]
        for name in self.feature_names:
            coeff = self.get_coefficient(name)
            sign = "+" if coeff >= 0 else ""
            terms.append(f"{sign}{coeff:.3f}*{name}")

        return " = ".join(["y", " + ".join(terms)])

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics."""
        if not self.residuals:
            residual_mean = 0.0
            residual_std = 0.0
        else:
            residual_mean = sum(self.residuals) / len(self.residuals)
            residual_variance = sum((x - residual_mean) ** 2 for x in self.residuals) / len(self.residuals)
            residual_std = residual_variance ** 0.5

        return {
            "model_type": self.model_type,
            "r_squared": self.metrics.r_squared,
            "adj_r_squared": self.metrics.adj_r_squared,
            "mse": self.metrics.mse,
            "rmse": self.metrics.rmse,
            "mae": self.metrics.mae,
            "f_statistic": self.metrics.f_statistic,
            "f_p_value": self.metrics.f_p_value,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "n_features": len(self.feature_names)
        }

    def is_good_fit(self) -> bool:
        """Business rule: Determine if model has acceptable fit."""
        return self.metrics.adj_r_squared > 0.5 and self.metrics.f_p_value < 0.05

    def has_multicollinearity_concerns(self) -> bool:
        """Business rule: Check for potential multicollinearity issues."""
        # Simple heuristic: high condition number or too many features
        return len(self.feature_names) > 5  # Arbitrary business rule

    def get_confidence_intervals(self, confidence_level: float = 0.95) -> Dict[str, List[float]]:
        """Calculate confidence intervals for coefficients."""
        # Simplified implementation - in practice would use statistical formulas
        critical_value = 2.0 if confidence_level == 0.95 else 1.96

        intervals = {"intercept": [self.parameters.intercept, self.parameters.intercept]}  # Simplified

        for feature_name in self.feature_names:
            coeff = self.get_coefficient(feature_name)
            # Assume some standard error (simplified)
            se = abs(coeff) * 0.1 if coeff != 0 else 0.1
            margin = critical_value * se
            intervals[feature_name] = [coeff - margin, coeff + margin]

        return intervals

    def calculate_prediction_accuracy(self, actual_values: List[float], predicted_values: List[float]) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        if len(actual_values) != len(predicted_values):
            raise ValueError("Actual and predicted values must have same length")

        n = len(actual_values)
        if n == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

        # Mean Absolute Error
        mae = sum(abs(a - p) for a, p in zip(actual_values, predicted_values)) / n

        # Root Mean Squared Error
        mse = sum((a - p) ** 2 for a, p in zip(actual_values, predicted_values)) / n
        rmse = mse ** 0.5

        # Mean Absolute Percentage Error (avoid division by zero)
        mape_values = []
        for a, p in zip(actual_values, predicted_values):
            if abs(a) > 1e-10:  # Avoid division by very small numbers
                mape_values.append(abs((a - p) / a))

        mape = sum(mape_values) / len(mape_values) if mape_values else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "sample_size": n
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance based on coefficient magnitudes."""
        importance = {}
        max_coeff = max(abs(self.get_coefficient(name)) for name in self.feature_names) or 1.0

        for name in self.feature_names:
            coeff = abs(self.get_coefficient(name))
            importance[name] = coeff / max_coeff if max_coeff > 0 else 0.0

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def validate_for_production(self) -> List[str]:
        """Business validation for production deployment."""
        issues = []

        # Check model quality
        if not self.is_good_fit():
            issues.append("Model fit is below acceptable standards")

        # Check for multicollinearity
        if self.has_multicollinearity_concerns():
            issues.append("Potential multicollinearity issues detected")

        # Check sample size adequacy
        if len(self.fitted_values) < 30:
            issues.append("Sample size too small for reliable inference")

        # Check for extreme coefficients
        for name in self.feature_names:
            coeff = self.get_coefficient(name)
            if abs(coeff) > 1000:  # Arbitrary threshold
                issues.append(f"Extremely large coefficient for {name}")

        return issues