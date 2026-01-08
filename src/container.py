"""
Dependency Injection Container.
Wires up all implementations to interfaces.
This is the ONLY place where concrete classes are instantiated.
"""
from src.core.application import RunRegressionUseCase
from src.infrastructure import DataProviderImpl, RegressionServiceImpl
from src.infrastructure.services.classification import ClassificationServiceImpl
from src.infrastructure.services.ml_bridge import MLBridgeService


class Container:
    """
    Simple DI Container for the application.
    Provides configured Use Cases with injected dependencies.
    """
    
    def __init__(self):
        # Infrastructure implementations
        self._data_provider = DataProviderImpl()
        self._regression_service = RegressionServiceImpl()
        self._classification_service = ClassificationServiceImpl()
        self._ml_bridge_service = MLBridgeService()
    
    @property
    def run_regression_use_case(self) -> RunRegressionUseCase:
        """Get configured RunRegressionUseCase."""
        return RunRegressionUseCase(
            data_provider=self._data_provider,
            regression_service=self._regression_service
        )
    
    @property
    def classification_service(self) -> ClassificationServiceImpl:
        """Get Classification Service."""
        return self._classification_service
    
    @property
    def ml_bridge_service(self) -> MLBridgeService:
        """Get ML Bridge Service."""
        return self._ml_bridge_service

    @property
    def run_classification_use_case(self):
        """Get configured RunClassificationUseCase."""
        from src.core.application.use_cases import RunClassificationUseCase
        return RunClassificationUseCase(
            data_provider=self._data_provider,
            classification_service=self._classification_service
        )
    
    @property
    def preview_split_use_case(self):
        """Get configured PreviewSplitUseCase."""
        from src.infrastructure.services.data_splitting import DataSplitterService
        from src.core.application.use_cases import PreviewSplitUseCase
        
        # We instantiate DataSplitterService lazily or here. Does it have dependencies? No.
        splitter = DataSplitterService() 
        return PreviewSplitUseCase(
            data_provider=self._data_provider,
            splitter_service=splitter
        )

# Singleton instance
_container = None

def get_container() -> Container:
    """Get or create the DI container singleton."""
    global _container
    if _container is None:
        _container = Container()
    return _container
