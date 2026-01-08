"""
Application Use Cases.
Orchestrate domain objects and infrastructure services using dependency injection.
"""
from typing import Dict, Any
from .dtos import RegressionRequestDTO, RegressionResponseDTO
from ..domain.interfaces import IDataProvider, IRegressionService
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
