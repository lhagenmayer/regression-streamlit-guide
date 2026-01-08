"""
Application Use Cases.
Orchestrate domain objects and infrastructure services using dependency injection.
"""
from typing import Dict, Any
import numpy as np
from .dtos import RegressionRequestDTO, RegressionResponseDTO, ClassificationRequestDTO, ClassificationResponseDTO
from ..domain.interfaces import IDataProvider, IRegressionService, IClassificationService
from ..domain.entities import RegressionModel
from ..domain.value_objects import RegressionType


class RunRegressionUseCase:
    """
    Use Case: Run a regression analysis (Simple or Multiple).
    
    Follows Clean Architecture: orchestrates domain objects without
    implementing business logic itself.
    """
    
    def __init__(self, data_provider: IDataProvider, regression_service: IRegressionService):
        self.data_provider = data_provider
        self.regression_service = regression_service
        
    def execute(self, request: RegressionRequestDTO) -> RegressionResponseDTO:
        """
        Execute regression analysis pipeline.
        
        1. Fetch data via IDataProvider
        2. Train model via IRegressionService
        3. Build response DTO
        """
        # 1. Fetch Data via Interface
        data_result = self.data_provider.get_dataset(
            dataset_id=request.dataset_id,
            n=request.n_observations,
            noise=request.noise_level,
            seed=request.seed,
            true_intercept=request.true_intercept or 0.6,
            true_slope=request.true_slope or 0.52,
            regression_type=request.regression_type.name.lower()  # Convert Enum to string for infrastructure
        )
        
        # 2. Perform Regression via Service
        if request.regression_type == RegressionType.MULTIPLE:
            # Multiple regression
            x_data = [data_result["x1"], data_result["x2"]]
            y_data = data_result["y"]
            variable_names = [data_result.get("x1_label", "x1"), data_result.get("x2_label", "x2")]
            
            model = self.regression_service.train_multiple(x_data, y_data, variable_names)
        else:
            # Simple regression (default)
            x_data = data_result["x"]
            y_data = data_result["y"]
            
            model = self.regression_service.train_simple(x_data, y_data)
        
        # 3. Add Metadata
        model.dataset_metadata = data_result.get("metadata")
        model.regression_type = request.regression_type
        
        # 4. Construct Response DTO
        return self._build_response(model, data_result, request.regression_type)

    def _build_response(
        self, 
        model: RegressionModel, 
        data_raw: Dict[str, Any],
        regression_type: RegressionType
    ) -> RegressionResponseDTO:
        """Build immutable response DTO from model and raw data."""
        params = model.parameters
        metrics = model.metrics
        
        # Determine x_data based on regression type
        if regression_type == RegressionType.MULTIPLE:
            x_data = tuple([
                tuple(data_raw.get("x1", [])), 
                tuple(data_raw.get("x2", []))
            ])
            x_label = f"{data_raw.get('x1_label', 'x1')} & {data_raw.get('x2_label', 'x2')}"
        else:
            x_data = tuple(data_raw.get("x", []))
            x_label = data_raw.get("x_label", "x")
        
        return RegressionResponseDTO(
            model_id=model.id,
            success=model.is_trained(),
            coefficients=params.coefficients,
            metrics={
                "r_squared": metrics.r_squared,
                "r_squared_adj": metrics.r_squared_adj,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "f_statistic": metrics.f_statistic,
                "p_value": metrics.p_value
            },
            x_data=x_data,
            y_data=tuple(data_raw.get("y", [])),
            residuals=tuple(model.residuals),
            predictions=tuple(model.predictions),
            x_label=x_label,
            y_label=data_raw.get("y_label", "y"),
            title=data_raw.get("context_title", ""),
            description=data_raw.get("context_description", ""),
            quality=model.get_quality(),
            is_significant=model.is_significant(),
            extra=data_raw.get("extra", {})
        )


class RunClassificationUseCase:
    """
    Use Case: Run a classification analysis (Logistic / KNN).
    """
    
    def __init__(self, data_provider: IDataProvider, classification_service: IClassificationService):
        self.data_provider = data_provider
        self.classification_service = classification_service
        # Lazy load splitter to avoid heavy imports in init if not needed?
        # Better to have it injected, but for now we follow the pattern
        from ...infrastructure.services.data_splitting import DataSplitterService
        self.splitter_service = DataSplitterService()
        
    def execute(self, request: ClassificationRequestDTO) -> ClassificationResponseDTO:
        # 1. Fetch Data
        data_raw = self.data_provider.get_dataset(
            dataset_id=request.dataset_id,
            n=request.n_observations,
            noise=request.noise_level,
            seed=request.seed,
            analysis_type="classification"
        )
        
        # 2. Extract arrays
        if "X" not in data_raw or "y" not in data_raw:
             raise ValueError("Dataset does not contain X and y for classification")
             
        X = np.array(data_raw["X"])
        y = np.array(data_raw["y"])
        
        # 3. Split Data
        from ..domain.value_objects import SplitConfig
        config = SplitConfig(
            train_size=request.train_size,
            stratify=request.stratify,
            seed=request.seed
        )
        X_train, X_test, y_train, y_test = self.splitter_service.split_data(X, y, config)
        
        # 4. Train Model (on Training Set)
        if request.method == "knn":
            # For KNN, "training" is just storing X_train, y_train
            train_result = self.classification_service.train_knn(X_train, y_train, k=request.k_neighbors)
        else:
            train_result = self.classification_service.train_logistic(X_train, y_train)
            
        # 5. Evaluate on Test Set
        test_metrics = self.classification_service.evaluate(
            X_test, 
            y_test, 
            train_result.model_params, 
            request.method
        )
        
        # 6. Predict on Full Data for Visualization
        # Use simple utility in service, or re-run "train" in prediction mode? 
        # But `train` returns Result with metrics. 
        if request.method == "knn":
            full_preds, full_probs = self.classification_service.predict_knn(X, train_result.model_params)
        else:
            full_preds, full_probs = self.classification_service.predict_logistic(X, train_result.model_params)
            
        # 6. Build Response
        return ClassificationResponseDTO(
            success=train_result.is_success,
            method=request.method,
            classes=tuple(train_result.classes),
            metrics={ # Train Metrics
                "accuracy": train_result.metrics.accuracy,
                "precision": train_result.metrics.precision,
                "recall": train_result.metrics.recall,
                "f1": train_result.metrics.f1_score,
                "confusion_matrix": train_result.metrics.confusion_matrix.tolist()
            },
            test_metrics={ # Test Metrics
                "accuracy": test_metrics.accuracy,
                "precision": test_metrics.precision,
                "recall": test_metrics.recall,
                "f1": test_metrics.f1_score,
                "confusion_matrix": test_metrics.confusion_matrix.tolist()
            },
            parameters=train_result.model_params,
            # We return FULL data for visualization context
            X_data=tuple(map(tuple, X)), 
            y_data=tuple(y),
            
            # Predictions on FULL dataset
            predictions=tuple(full_preds),
            probabilities=tuple(full_probs) if full_probs is not None else (),
            
            feature_names=tuple(data_raw.get("feature_names", [])),
            target_names=tuple(data_raw.get("target_names", [])),
            dataset_name=data_raw.get("name", "Dataset"),
            dataset_description=data_raw.get("description", ""),
            extra={}
        )


class PreviewSplitUseCase:
    """
    Use Case: Preview data split statistics.
    """
    
    def __init__(self, data_provider: IDataProvider, splitter_service):
        self.data_provider = data_provider
        self.splitter_service = splitter_service
        
    def execute(self, dataset_id: str, n: int, noise: float, seed: int, train_size: float, stratify: bool) -> Any:
        # 1. Fetch Data
        data_raw = self.data_provider.get_dataset(
            dataset_id=dataset_id,
            n=n,
            noise=noise,
            seed=seed,
            analysis_type="classification"
        )
        
        if "y" not in data_raw:
             raise ValueError("Dataset does not contain y for splitting")
             
        y = np.array(data_raw["y"])
        
        # 2. Config
        from ..domain.value_objects import SplitConfig
        config = SplitConfig(
            train_size=train_size,
            stratify=stratify,
            seed=seed
        )
        
        # 3. Calculate Stats
        stats = self.splitter_service.preview_split(y, config)
        
        return stats
