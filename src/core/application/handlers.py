"""
Command and Query Handlers - CQRS Implementation.

Handler verarbeiten Commands (Schreiboperationen) und Queries (Leseoperationen)
getrennt voneinander. Dies ermöglicht:
- Unabhängige Skalierung von Lese- und Schreiboperationen
- Optimierte Datenmodelle für jeweilige Operationen
- Klare Verantwortlichkeiten
"""

from typing import TypeVar, Generic, Protocol, Dict, Any, List, Optional, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .commands import (
    Command,
    CreateDatasetCommand,
    UpdateDatasetCommand,
    DeleteDatasetCommand,
    CreateRegressionModelCommand,
    UpdateModelParametersCommand,
    DeleteModelCommand,
    GenerateSyntheticDataCommand
)
from .queries import (
    Query,
    GetDatasetByIdQuery,
    ListDatasetsQuery,
    GetModelByIdQuery,
    ListModelsQuery,
    GetDatasetStatisticsQuery,
    GetModelDiagnosticsQuery,
    CompareModelsQuery,
    GetAvailableDataSourcesQuery
)
from ..domain.result import Result, Error, ValidationError
from ..domain.events import EventDispatcher, DomainEvent
from ..domain.aggregates import DatasetAggregate, RegressionModelAggregate
from ..domain.repositories import DatasetRepository, RegressionModelRepository


TCommand = TypeVar('TCommand', bound=Command)
TQuery = TypeVar('TQuery', bound=Query)
TResult = TypeVar('TResult')


# ============================================================================
# Handler Protocols
# ============================================================================

class CommandHandler(Protocol[TCommand, TResult]):
    """Protocol for command handlers."""

    def handle(self, command: TCommand) -> Result[TResult]:
        """Handle a command and return result."""
        ...


class QueryHandler(Protocol[TQuery, TResult]):
    """Protocol for query handlers."""

    def handle(self, query: TQuery) -> Result[TResult]:
        """Handle a query and return result."""
        ...


# ============================================================================
# Command Handlers
# ============================================================================

class CreateDatasetHandler:
    """
    Handler for CreateDatasetCommand.

    Erstellt ein neues Dataset und speichert es im Repository.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        self.dataset_repository = dataset_repository
        self.event_dispatcher = event_dispatcher

    def handle(self, command: CreateDatasetCommand) -> Result[DatasetAggregate]:
        """
        Handle dataset creation.

        Steps:
        1. Validate command data
        2. Create DatasetAggregate
        3. Persist to repository
        4. Dispatch domain events
        """
        # Create aggregate using factory
        result = DatasetAggregate.create(
            id=f"dataset_{command.config.name.lower().replace(' ', '_')}",
            config=command.config,
            data={}  # Initial empty data
        )

        if result.is_failure:
            return result

        aggregate = result.value

        # Persist
        try:
            self.dataset_repository.save(aggregate)
        except Exception as e:
            return Result.failure(Error("PERSISTENCE_ERROR", str(e)))

        # Dispatch events
        if self.event_dispatcher:
            self.event_dispatcher.dispatch_all(aggregate.clear_events())

        return Result.success(aggregate)


class CreateRegressionModelHandler:
    """
    Handler for CreateRegressionModelCommand.

    Erstellt ein Regressionsmodell aus einem bestehenden Dataset.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        model_repository: RegressionModelRepository,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        self.dataset_repository = dataset_repository
        self.model_repository = model_repository
        self.event_dispatcher = event_dispatcher

    def handle(self, command: CreateRegressionModelCommand) -> Result[RegressionModelAggregate]:
        """
        Handle model creation.

        Steps:
        1. Load and validate dataset
        2. Validate variables exist
        3. Fit model
        4. Create aggregate
        5. Persist and dispatch events
        """
        # Load dataset
        dataset = self.dataset_repository.find_by_id(command.dataset_id)
        if not dataset:
            return Result.failure(Error(
                "DATASET_NOT_FOUND",
                f"Dataset '{command.dataset_id}' not found"
            ))

        # Validate target variable
        if not hasattr(dataset, 'has_variable'):
            # Handle both entity and aggregate
            has_var = command.target_variable in getattr(dataset, 'data', {})
        else:
            has_var = dataset.has_variable(command.target_variable)

        if not has_var:
            return Result.failure(Error(
                "TARGET_NOT_FOUND",
                f"Target variable '{command.target_variable}' not in dataset"
            ))

        # Validate feature variables
        data = getattr(dataset, 'data', {})
        missing = [f for f in command.feature_variables if f not in data]
        if missing:
            return Result.failure(Error(
                "FEATURES_NOT_FOUND",
                f"Feature variables not found: {missing}"
            ))

        # Fit model (simplified - in real impl would use statsmodels)
        y = data[command.target_variable]
        X_data = {f: data[f] for f in command.feature_variables}

        fitted_values, residuals, metrics = self._fit_model(
            y, X_data, command.parameters
        )

        # Create model aggregate
        import uuid
        model_id = f"model_{str(uuid.uuid4())[:8]}"

        result = RegressionModelAggregate.create(
            id=model_id,
            dataset_id=command.dataset_id,
            model_type="multiple" if len(command.feature_variables) > 1 else "simple",
            parameters=command.parameters,
            metrics=metrics,
            feature_names=command.feature_variables,
            fitted_values=fitted_values,
            residuals=residuals
        )

        if result.is_failure:
            return result

        aggregate = result.value

        # Persist
        try:
            self.model_repository.save(aggregate)
        except Exception as e:
            return Result.failure(Error("PERSISTENCE_ERROR", str(e)))

        # Dispatch events
        if self.event_dispatcher:
            self.event_dispatcher.dispatch_all(aggregate.clear_events())

        return Result.success(aggregate)

    def _fit_model(
        self,
        y: List[float],
        X_data: Dict[str, List[float]],
        parameters
    ) -> tuple:
        """
        Fit regression model (simplified implementation).

        Returns (fitted_values, residuals, metrics).
        """
        from ..domain.value_objects import ModelMetrics
        import random
        import math

        n = len(y)
        feature_names = list(X_data.keys())

        # Calculate fitted values
        fitted = [parameters.intercept] * n
        for i in range(n):
            for feature_name in feature_names:
                coeff = parameters.coefficients.get(feature_name, 0)
                fitted[i] += coeff * X_data[feature_name][i]

        # Add noise
        random.seed(parameters.seed)
        fitted = [f + random.gauss(0, parameters.noise_level) for f in fitted]

        # Calculate residuals
        residuals = [y[i] - fitted[i] for i in range(n)]

        # Calculate metrics
        ss_res = sum(r ** 2 for r in residuals)
        y_mean = sum(y) / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        p = len(feature_names) + 1
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p) if n > p else r_squared

        mse = ss_res / (n - p) if n > p else 0
        rmse = math.sqrt(mse) if mse > 0 else 0
        mae = sum(abs(r) for r in residuals) / n

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

        return fitted, residuals, metrics


class DeleteDatasetHandler:
    """Handler for DeleteDatasetCommand."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        self.dataset_repository = dataset_repository
        self.event_dispatcher = event_dispatcher

    def handle(self, command: DeleteDatasetCommand) -> Result[None]:
        """Handle dataset deletion."""
        # Check exists
        dataset = self.dataset_repository.find_by_id(command.dataset_id)
        if not dataset:
            return Result.failure(Error(
                "NOT_FOUND",
                f"Dataset '{command.dataset_id}' not found"
            ))

        # Delete
        try:
            self.dataset_repository.delete(command.dataset_id)
        except Exception as e:
            return Result.failure(Error("DELETE_ERROR", str(e)))

        # Dispatch event
        if self.event_dispatcher:
            from ..domain.events import DatasetDeleted
            self.event_dispatcher.dispatch(DatasetDeleted.create(command.dataset_id))

        return Result.success(None)


class DeleteModelHandler:
    """Handler for DeleteModelCommand."""

    def __init__(
        self,
        model_repository: RegressionModelRepository,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        self.model_repository = model_repository
        self.event_dispatcher = event_dispatcher

    def handle(self, command: DeleteModelCommand) -> Result[None]:
        """Handle model deletion."""
        model = self.model_repository.find_by_id(command.model_id)
        if not model:
            return Result.failure(Error(
                "NOT_FOUND",
                f"Model '{command.model_id}' not found"
            ))

        try:
            self.model_repository.delete(command.model_id)
        except Exception as e:
            return Result.failure(Error("DELETE_ERROR", str(e)))

        return Result.success(None)


# ============================================================================
# Query Handlers
# ============================================================================

class GetDatasetByIdHandler:
    """Handler for GetDatasetByIdQuery."""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def handle(self, query: GetDatasetByIdQuery) -> Result[Any]:
        """Handle dataset retrieval by ID."""
        dataset = self.dataset_repository.find_by_id(query.dataset_id)
        if not dataset:
            return Result.failure(Error(
                "NOT_FOUND",
                f"Dataset '{query.dataset_id}' not found"
            ))
        return Result.success(dataset)


class ListDatasetsHandler:
    """Handler for ListDatasetsQuery."""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def handle(self, query: ListDatasetsQuery) -> Result[List[Any]]:
        """Handle dataset listing."""
        datasets = self.dataset_repository.list_all()

        # Apply pagination if specified
        if query.offset:
            datasets = datasets[query.offset:]
        if query.limit:
            datasets = datasets[:query.limit]

        return Result.success(datasets)


class GetModelByIdHandler:
    """Handler for GetModelByIdQuery."""

    def __init__(self, model_repository: RegressionModelRepository):
        self.model_repository = model_repository

    def handle(self, query: GetModelByIdQuery) -> Result[Any]:
        """Handle model retrieval by ID."""
        model = self.model_repository.find_by_id(query.model_id)
        if not model:
            return Result.failure(Error(
                "NOT_FOUND",
                f"Model '{query.model_id}' not found"
            ))
        return Result.success(model)


class ListModelsHandler:
    """Handler for ListModelsQuery."""

    def __init__(self, model_repository: RegressionModelRepository):
        self.model_repository = model_repository

    def handle(self, query: ListModelsQuery) -> Result[List[Any]]:
        """Handle model listing."""
        if query.dataset_id:
            models = self.model_repository.find_by_dataset_id(query.dataset_id)
        else:
            models = self.model_repository.list_all()

        # Apply pagination
        if query.offset:
            models = models[query.offset:]
        if query.limit:
            models = models[:query.limit]

        return Result.success(models)


class GetDatasetStatisticsHandler:
    """Handler for GetDatasetStatisticsQuery."""

    def __init__(self, dataset_repository: DatasetRepository):
        self.dataset_repository = dataset_repository

    def handle(self, query: GetDatasetStatisticsQuery) -> Result[Dict[str, Any]]:
        """Handle dataset statistics calculation."""
        dataset = self.dataset_repository.find_by_id(query.dataset_id)
        if not dataset:
            return Result.failure(Error(
                "NOT_FOUND",
                f"Dataset '{query.dataset_id}' not found"
            ))

        # Calculate statistics
        from ..domain.value_objects import StatisticalSummary

        data = getattr(dataset, 'data', {})
        statistics = {}

        for var_name, values in data.items():
            if values:
                summary = StatisticalSummary.from_list(values)
                statistics[var_name] = {
                    "count": summary.count,
                    "mean": summary.mean,
                    "std": summary.std,
                    "min": summary.min_val,
                    "max": summary.max_val,
                    "median": summary.median,
                    "q25": summary.q25,
                    "q75": summary.q75
                }

        return Result.success({
            "dataset_id": query.dataset_id,
            "n_variables": len(data),
            "n_observations": len(next(iter(data.values()))) if data else 0,
            "variable_statistics": statistics
        })


class GetModelDiagnosticsHandler:
    """Handler for GetModelDiagnosticsQuery."""

    def __init__(self, model_repository: RegressionModelRepository):
        self.model_repository = model_repository

    def handle(self, query: GetModelDiagnosticsQuery) -> Result[Dict[str, Any]]:
        """Handle model diagnostics retrieval."""
        model = self.model_repository.find_by_id(query.model_id)
        if not model:
            return Result.failure(Error(
                "NOT_FOUND",
                f"Model '{query.model_id}' not found"
            ))

        # Get diagnostics
        diagnostics = {
            "model_id": query.model_id,
            "model_type": model.model_type,
            "n_features": len(model.feature_names),
            "feature_names": list(model.feature_names),
            "metrics": {
                "r_squared": model.metrics.r_squared,
                "adj_r_squared": model.metrics.adj_r_squared,
                "mse": model.metrics.mse,
                "rmse": model.metrics.rmse,
                "mae": model.metrics.mae,
                "f_statistic": model.metrics.f_statistic,
                "f_p_value": model.metrics.f_p_value
            },
            "is_good_fit": model.is_good_fit() if hasattr(model, 'is_good_fit') else None
        }

        # Residual analysis
        if model.residuals:
            residuals = model.residuals
            residual_mean = sum(residuals) / len(residuals)
            residual_std = (sum((r - residual_mean) ** 2 for r in residuals) / len(residuals)) ** 0.5

            diagnostics["residual_analysis"] = {
                "mean": residual_mean,
                "std": residual_std,
                "min": min(residuals),
                "max": max(residuals)
            }

        return Result.success(diagnostics)


class CompareModelsHandler:
    """Handler for CompareModelsQuery."""

    def __init__(self, model_repository: RegressionModelRepository):
        self.model_repository = model_repository

    def handle(self, query: CompareModelsQuery) -> Result[Dict[str, Any]]:
        """Handle model comparison."""
        model1 = self.model_repository.find_by_id(query.model1_id)
        model2 = self.model_repository.find_by_id(query.model2_id)

        if not model1:
            return Result.failure(Error("NOT_FOUND", f"Model '{query.model1_id}' not found"))
        if not model2:
            return Result.failure(Error("NOT_FOUND", f"Model '{query.model2_id}' not found"))

        # Compare metrics
        r2_diff = model2.metrics.adj_r_squared - model1.metrics.adj_r_squared
        complexity_diff = len(model2.feature_names) - len(model1.feature_names)

        # Determine winner
        if abs(r2_diff) < 0.05:
            if complexity_diff < 0:
                winner = "model2"
                reason = "Model 2 is simpler with similar performance"
            elif complexity_diff > 0:
                winner = "model1"
                reason = "Model 1 is simpler with similar performance"
            else:
                winner = "tie"
                reason = "Models have similar performance and complexity"
        else:
            winner = "model2" if r2_diff > 0 else "model1"
            reason = f"Better explanatory power (ΔR² = {r2_diff:.4f})"

        return Result.success({
            "model1_id": query.model1_id,
            "model2_id": query.model2_id,
            "winner": winner,
            "reason": reason,
            "r2_difference": r2_diff,
            "complexity_difference": complexity_diff,
            "model1_metrics": {
                "r_squared": model1.metrics.r_squared,
                "adj_r_squared": model1.metrics.adj_r_squared,
                "n_features": len(model1.feature_names)
            },
            "model2_metrics": {
                "r_squared": model2.metrics.r_squared,
                "adj_r_squared": model2.metrics.adj_r_squared,
                "n_features": len(model2.feature_names)
            }
        })


# ============================================================================
# Mediator Pattern - Command/Query Dispatcher
# ============================================================================

class Mediator:
    """
    Mediator für Command/Query Routing.

    Leitet Commands und Queries an die entsprechenden Handler weiter.
    Ermöglicht lose Kopplung zwischen Aufrufern und Handlern.
    """

    def __init__(self):
        self._command_handlers: Dict[Type[Command], Any] = {}
        self._query_handlers: Dict[Type[Query], Any] = {}

    def register_command_handler(
        self,
        command_type: Type[Command],
        handler: Any
    ) -> None:
        """Register a handler for a command type."""
        self._command_handlers[command_type] = handler

    def register_query_handler(
        self,
        query_type: Type[Query],
        handler: Any
    ) -> None:
        """Register a handler for a query type."""
        self._query_handlers[query_type] = handler

    def send_command(self, command: Command) -> Result:
        """
        Send a command to its handler.

        Returns Result from handler.
        """
        command_type = type(command)
        if command_type not in self._command_handlers:
            return Result.failure(Error(
                "NO_HANDLER",
                f"No handler registered for {command_type.__name__}"
            ))

        handler = self._command_handlers[command_type]
        return handler.handle(command)

    def send_query(self, query: Query) -> Result:
        """
        Send a query to its handler.

        Returns Result from handler.
        """
        query_type = type(query)
        if query_type not in self._query_handlers:
            return Result.failure(Error(
                "NO_HANDLER",
                f"No handler registered for {query_type.__name__}"
            ))

        handler = self._query_handlers[query_type]
        return handler.handle(query)


def create_mediator(
    dataset_repository: DatasetRepository,
    model_repository: RegressionModelRepository,
    event_dispatcher: Optional[EventDispatcher] = None
) -> Mediator:
    """
    Factory function to create a fully configured Mediator.

    Registers all command and query handlers.
    """
    mediator = Mediator()

    # Command Handlers
    mediator.register_command_handler(
        CreateDatasetCommand,
        CreateDatasetHandler(dataset_repository, event_dispatcher)
    )
    mediator.register_command_handler(
        CreateRegressionModelCommand,
        CreateRegressionModelHandler(dataset_repository, model_repository, event_dispatcher)
    )
    mediator.register_command_handler(
        DeleteDatasetCommand,
        DeleteDatasetHandler(dataset_repository, event_dispatcher)
    )
    mediator.register_command_handler(
        DeleteModelCommand,
        DeleteModelHandler(model_repository, event_dispatcher)
    )

    # Query Handlers
    mediator.register_query_handler(
        GetDatasetByIdQuery,
        GetDatasetByIdHandler(dataset_repository)
    )
    mediator.register_query_handler(
        ListDatasetsQuery,
        ListDatasetsHandler(dataset_repository)
    )
    mediator.register_query_handler(
        GetModelByIdQuery,
        GetModelByIdHandler(model_repository)
    )
    mediator.register_query_handler(
        ListModelsQuery,
        ListModelsHandler(model_repository)
    )
    mediator.register_query_handler(
        GetDatasetStatisticsQuery,
        GetDatasetStatisticsHandler(dataset_repository)
    )
    mediator.register_query_handler(
        GetModelDiagnosticsQuery,
        GetModelDiagnosticsHandler(model_repository)
    )
    mediator.register_query_handler(
        CompareModelsQuery,
        CompareModelsHandler(model_repository)
    )

    return mediator
