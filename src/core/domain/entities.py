"""
Domain Entities.
Objects with identity and lifecycle.
Pure Python - NO external dependencies (no datetime, uuid module used only for id generation).
"""
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

from .value_objects import (
    RegressionParameters, 
    RegressionMetrics, 
    DatasetMetadata,
    RegressionType,
    ModelQuality,
)


@dataclass
class RegressionModel:
    """
    Core Domain Entity representing a trained regression model.
    Has identity (id) and holds the state of the analysis.
    
    Note: We use str for created_at to avoid datetime dependency.
    Infrastructure layer can convert to/from datetime as needed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at_iso: str = ""  # ISO format string, set by infrastructure
    regression_type: RegressionType = RegressionType.SIMPLE
    dataset_metadata: Optional[DatasetMetadata] = None
    
    # State (initially None until trained)
    parameters: Optional[RegressionParameters] = None
    metrics: Optional[RegressionMetrics] = None
    
    # Results
    residuals: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    
    def is_trained(self) -> bool:
        """Check if model has been successfully trained."""
        return self.parameters is not None and self.metrics is not None

    def get_equation_string(self) -> str:
        """Domain logic to generate equation string representation."""
        if not self.is_trained():
            return "Not trained"
            
        parts = [f"{self.parameters.intercept:.4f}"]
        for name, coef in self.parameters.coefficients.items():
            sign = "+" if coef >= 0 else "-"
            parts.append(f"{sign} {abs(coef):.4f}·{name}")
            
        return "ŷ = " + " ".join(parts)
    
    def get_quality(self) -> Optional[ModelQuality]:
        """Get model quality classification."""
        if not self.is_trained():
            return None
        return self.metrics.quality
    
    def get_r_squared(self) -> Optional[float]:
        """Get R² value if trained."""
        if not self.is_trained():
            return None
        return self.metrics.r_squared
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if model is statistically significant at given alpha level."""
        if not self.is_trained():
            return False
        return self.metrics.is_significant(alpha)
    
    def validate(self) -> List[str]:
        """Validate entity state, return list of errors."""
        errors = []
        if self.is_trained():
            if not self.predictions:
                errors.append("Trained model should have predictions")
            if not self.residuals:
                errors.append("Trained model should have residuals")
            if len(self.predictions) != len(self.residuals):
                errors.append("Predictions and residuals must have same length")
        return errors
