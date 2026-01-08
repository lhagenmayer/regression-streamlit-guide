"""
Domain Interfaces (Ports).
Using Protocols for structural subtyping.
Single Responsibility Principle - each interface has ONE job.
"""
from typing import Protocol, List, Dict, Any, Optional
from .entities import RegressionModel
from .value_objects import DatasetMetadata, RegressionType, Result


# =============================================================================
# Data Provider Interfaces (Single Responsibility Split)
# =============================================================================

class IDatasetFetcher(Protocol):
    """Interface for fetching a single dataset."""
    
    def fetch(self, dataset_id: str, n: int, **kwargs) -> Result:
        """
        Fetch raw data.
        Returns Success(Dict) or Failure(error_message).
        """
        ...


class IDatasetLister(Protocol):
    """Interface for listing available datasets."""
    
    def list_all(self) -> List[DatasetMetadata]:
        """List all available datasets."""
        ...


class IDataProvider(IDatasetFetcher, IDatasetLister, Protocol):
    """Combined interface for data operations (backward compatible)."""
    
    def get_dataset(self, dataset_id: str, n: int, **kwargs) -> Dict[str, Any]:
        """Legacy method - use fetch() for Result-based error handling."""
        ...
        
    def get_all_datasets(self) -> Dict[str, List[Dict[str, str]]]:
        """List all datasets grouped by type."""
        ...
        
    def get_raw_data(self, dataset_id: str) -> Dict[str, Any]:
        """Get raw tabular data for a dataset."""
        ...


# =============================================================================
# Regression Service Interfaces (Single Responsibility Split)
# =============================================================================

class ISimpleRegressionTrainer(Protocol):
    """Interface for training simple regression models."""
    
    def train(self, x: List[float], y: List[float]) -> RegressionModel:
        """Train simple regression: y = β₀ + β₁x."""
        ...


class IMultipleRegressionTrainer(Protocol):
    """Interface for training multiple regression models."""
    
    def train(
        self, 
        x: List[List[float]], 
        y: List[float], 
        variable_names: List[str]
    ) -> RegressionModel:
        """Train multiple regression: y = β₀ + β₁x₁ + β₂x₂ + ..."""
        ...


class IRegressionService(Protocol):
    """Combined interface for regression operations (backward compatible)."""
    
    def train_simple(self, x: List[float], y: List[float]) -> RegressionModel:
        """Train simple regression model."""
        ...
        
    def train_multiple(
        self, 
        x: List[List[float]], 
        y: List[float], 
        variable_names: List[str]
    ) -> RegressionModel:
        """Train multiple regression model."""
        ...


# =============================================================================
# Model Repository Interface
# =============================================================================

class IModelRepository(Protocol):
    """Interface for persisting and retrieving models."""
    
    def save(self, model: RegressionModel) -> str:
        """Save model, return model_id."""
        ...
    
    def get(self, model_id: str) -> Optional[RegressionModel]:
        """Retrieve model by ID."""
        ...
    
    def delete(self, model_id: str) -> bool:
        """Delete model, return success status."""
        ...


# =============================================================================
# Prediction Interface
# =============================================================================

class IPredictor(Protocol):
    """Protocol for making predictions."""
    def predict(self, model: RegressionModel, data: Dict[str, Any]) -> float:
        """Make a prediction using the model."""
        ...


class IClassificationService(Protocol):
    """Protocol for classification operations."""
    
    def train_logistic(self, X: np.ndarray, y: np.ndarray) -> ClassificationResult:
        """Train a logistic regression model."""
        ...
        
    def train_knn(self, X: np.ndarray, y: np.ndarray, k: int) -> ClassificationResult:
        """Train a K-Nearest Neighbors model."""
        ...
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
        """Calculate classification performance metrics."""
        ...
