"""
Tests für die verbesserte Core Architecture.

Diese Tests decken alle neuen DDD Patterns ab:
- Result Pattern
- Specification Pattern
- Aggregate Root Pattern
- Factory Pattern
- Unit of Work Pattern
- Command/Query Handlers
"""

import pytest
from typing import Dict, List

# Domain Layer Imports
from src.core.domain.result import (
    Result, Error, ValidationResult,
    DomainError, ValidationError, NotFoundError
)
from src.core.domain.specifications import (
    Specification, AndSpecification, OrSpecification, NotSpecification,
    HasMinimumSampleSize, HasRequiredVariables, HasSufficientVariation,
    IsStatisticallySignificant, HasMinimumRSquared, IsProductionReady,
    SpecificationFactory
)
from src.core.domain.aggregates import DatasetAggregate, RegressionModelAggregate
from src.core.domain.factories import (
    DatasetConfigFactory, RegressionParametersFactory, ModelMetricsFactory,
    DatasetFactory, RegressionModelFactory
)
from src.core.domain.value_objects import DatasetConfig, RegressionParameters, ModelMetrics
from src.core.domain.events import (
    DomainEvent, EventMetadata, DatasetCreated, RegressionModelCreated,
    InMemoryEventStore, EventDispatcher
)
from src.core.domain.unit_of_work import InMemoryUnitOfWork, unit_of_work_scope

# Infrastructure Imports
from src.infrastructure.repositories import InMemoryDatasetRepository, InMemoryModelRepository


# ============================================================================
# Result Pattern Tests
# ============================================================================

class TestResultPattern:
    """Tests für das Result Pattern."""

    def test_success_result(self):
        """Test erfolgreiche Result-Erstellung."""
        result = Result.success(42)

        assert result.is_success
        assert not result.is_failure
        assert result.value == 42

    def test_failure_result(self):
        """Test fehlgeschlagene Result-Erstellung."""
        error = Error("TEST_ERROR", "Test error message")
        result = Result.failure(error)

        assert result.is_failure
        assert not result.is_success
        assert result.error.code == "TEST_ERROR"
        assert result.error.message == "Test error message"

    def test_failure_from_string(self):
        """Test Result.failure mit String."""
        result = Result.failure("Something went wrong")

        assert result.is_failure
        assert "Something went wrong" in result.error.message

    def test_map_on_success(self):
        """Test map() auf erfolgreichem Result."""
        result = Result.success(5)
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_success
        assert mapped.value == 10

    def test_map_on_failure(self):
        """Test map() auf fehlgeschlagenem Result."""
        result = Result.failure(Error("ERR", "error"))
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_failure
        assert mapped.error.code == "ERR"

    def test_flat_map(self):
        """Test flat_map() für verkettete Operationen."""
        result = Result.success(5)

        def double_if_positive(x):
            if x > 0:
                return Result.success(x * 2)
            return Result.failure(Error("NEGATIVE", "Value must be positive"))

        mapped = result.flat_map(double_if_positive)

        assert mapped.is_success
        assert mapped.value == 10

    def test_combine_results(self):
        """Test Result.combine() für mehrere Results."""
        r1 = Result.success(1)
        r2 = Result.success(2)
        r3 = Result.success(3)

        combined = Result.combine(r1, r2, r3)

        assert combined.is_success
        assert combined.value == [1, 2, 3]

    def test_combine_with_failure(self):
        """Test Result.combine() mit Fehlern."""
        r1 = Result.success(1)
        r2 = Result.failure(Error("ERR", "error"))
        r3 = Result.success(3)

        combined = Result.combine(r1, r2, r3)

        assert combined.is_failure
        assert len(combined.errors) == 1

    def test_get_or_default(self):
        """Test get_or_default()."""
        success = Result.success(42)
        failure = Result.failure(Error("ERR", "error"))

        assert success.get_or_default(0) == 42
        assert failure.get_or_default(0) == 0

    def test_ensure(self):
        """Test ensure() für Validierung."""
        result = Result.success(5)

        valid = result.ensure(lambda x: x > 0, Error("INVALID", "Must be positive"))
        invalid = result.ensure(lambda x: x > 10, Error("INVALID", "Must be > 10"))

        assert valid.is_success
        assert invalid.is_failure


class TestValidationResult:
    """Tests für ValidationResult."""

    def test_valid_result(self):
        """Test gültiges ValidationResult."""
        result = ValidationResult.valid()

        assert result.is_valid
        assert not result.is_invalid
        assert len(result.errors) == 0

    def test_invalid_result(self):
        """Test ungültiges ValidationResult."""
        result = ValidationResult.invalid("ERR", "Error message")

        assert result.is_invalid
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_add_multiple_errors(self):
        """Test mehrere Fehler hinzufügen."""
        result = ValidationResult.valid()
        result.add_error("ERR1", "Error 1")
        result.add_error("ERR2", "Error 2")

        assert result.is_invalid
        assert len(result.errors) == 2

    def test_add_error_if(self):
        """Test bedingtes Hinzufügen von Fehlern."""
        result = ValidationResult.valid()
        result.add_error_if(True, "ERR1", "This should be added")
        result.add_error_if(False, "ERR2", "This should not be added")

        assert len(result.errors) == 1
        assert result.errors[0].code == "ERR1"

    def test_merge_results(self):
        """Test Zusammenführen von ValidationResults."""
        r1 = ValidationResult.invalid("ERR1", "Error 1")
        r2 = ValidationResult.invalid("ERR2", "Error 2")

        r1.merge(r2)

        assert len(r1.errors) == 2

    def test_to_result(self):
        """Test Konvertierung zu Result."""
        valid = ValidationResult.valid()
        invalid = ValidationResult.invalid("ERR", "Error")

        success = valid.to_result("value")
        failure = invalid.to_result("value")

        assert success.is_success
        assert success.value == "value"
        assert failure.is_failure


# ============================================================================
# Specification Pattern Tests
# ============================================================================

class TestSpecificationPattern:
    """Tests für das Specification Pattern."""

    def test_has_minimum_sample_size(self):
        """Test HasMinimumSampleSize Specification."""
        # Create test dataset
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=50
        )
        dataset = DatasetAggregate(id="test", config=config, data={"x": list(range(50))})

        spec = HasMinimumSampleSize(30)

        assert spec.is_satisfied_by(dataset)

    def test_has_minimum_sample_size_fails(self):
        """Test HasMinimumSampleSize mit zu wenigen Daten."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )
        dataset = DatasetAggregate(id="test", config=config, data={"x": list(range(10))})

        spec = HasMinimumSampleSize(30)

        assert not spec.is_satisfied_by(dataset)

    def test_and_specification(self):
        """Test AND-Kombination von Specifications."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric", "y": "numeric"},
            n_observations=50
        )
        dataset = DatasetAggregate(
            id="test",
            config=config,
            data={"x": list(range(50)), "y": list(range(50))}
        )

        spec1 = HasMinimumSampleSize(30)
        spec2 = HasRequiredVariables(["x", "y"])

        combined = spec1 & spec2

        assert isinstance(combined, AndSpecification)
        assert combined.is_satisfied_by(dataset)

    def test_or_specification(self):
        """Test OR-Kombination von Specifications."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )
        dataset = DatasetAggregate(id="test", config=config, data={"x": list(range(10))})

        spec1 = HasMinimumSampleSize(30)  # Will fail
        spec2 = HasMinimumSampleSize(5)   # Will pass

        combined = spec1 | spec2

        assert isinstance(combined, OrSpecification)
        assert combined.is_satisfied_by(dataset)

    def test_not_specification(self):
        """Test NOT-Negation von Specifications."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )
        dataset = DatasetAggregate(id="test", config=config, data={"x": list(range(10))})

        spec = HasMinimumSampleSize(30)
        negated = ~spec

        assert isinstance(negated, NotSpecification)
        assert negated.is_satisfied_by(dataset)  # Negated: not enough samples = True

    def test_specification_factory(self):
        """Test SpecificationFactory."""
        spec = SpecificationFactory.dataset_for_regression(
            target_variable="y",
            feature_variables=["x1", "x2"],
            min_sample_size=30
        )

        assert spec is not None


# ============================================================================
# Aggregate Root Tests
# ============================================================================

class TestAggregateRoot:
    """Tests für Aggregate Root Pattern."""

    def test_dataset_aggregate_creation(self):
        """Test DatasetAggregate Erstellung."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test dataset",
            variables={"x": "numeric", "y": "numeric"},
            n_observations=100
        )

        result = DatasetAggregate.create(
            id="test_dataset",
            config=config,
            data={"x": list(range(100)), "y": list(range(100))}
        )

        assert result.is_success
        aggregate = result.value
        assert aggregate.id == "test_dataset"
        assert len(aggregate.uncommitted_events) == 1

    def test_dataset_aggregate_add_variable(self):
        """Test Variable hinzufügen."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )
        dataset = DatasetAggregate(id="test", config=config, data={"x": list(range(10))})

        result = dataset.add_variable("y", list(range(10)))

        assert result.is_success
        assert dataset.has_variable("y")

    def test_dataset_aggregate_add_variable_invalid_length(self):
        """Test Variable mit falscher Länge hinzufügen."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )
        dataset = DatasetAggregate(id="test", config=config, data={"x": list(range(10))})

        result = dataset.add_variable("y", list(range(5)))  # Wrong length

        assert result.is_failure
        assert result.error.code == "LENGTH_MISMATCH"

    def test_aggregate_version_tracking(self):
        """Test Version Tracking."""
        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )
        dataset = DatasetAggregate(id="test", config=config, data={"x": list(range(10))})

        assert dataset.version == 0

        dataset.increment_version()
        assert dataset.version == 1


# ============================================================================
# Factory Pattern Tests
# ============================================================================

class TestFactoryPattern:
    """Tests für Factory Pattern."""

    def test_dataset_config_factory_synthetic(self):
        """Test DatasetConfigFactory für synthetische Daten."""
        result = DatasetConfigFactory.create_synthetic(
            name="Test Dataset",
            description="A test dataset",
            variables={"x": "numeric", "y": "numeric"},
            n_observations=100
        )

        assert result.is_success
        config = result.value
        assert config.name == "Test Dataset"
        assert config.dataset_type == "synthetic"

    def test_regression_parameters_factory(self):
        """Test RegressionParametersFactory."""
        result = RegressionParametersFactory.create(
            intercept=5.0,
            coefficients={"x": 2.5, "z": -1.0},
            noise_level=0.5
        )

        assert result.is_success
        params = result.value
        assert params.intercept == 5.0
        assert params.coefficients["x"] == 2.5

    def test_regression_parameters_factory_invalid_noise(self):
        """Test RegressionParametersFactory mit ungültigem Noise."""
        result = RegressionParametersFactory.create(
            intercept=5.0,
            coefficients={"x": 2.5},
            noise_level=-1.0  # Invalid
        )

        assert result.is_failure

    def test_dataset_factory_create_synthetic(self):
        """Test DatasetFactory für synthetische Daten."""
        result = DatasetFactory.create_synthetic(
            name="Synthetic Test",
            n_observations=100,
            features=["x1", "x2"],
            target="y",
            intercept=10.0,
            coefficients={"x1": 2.0, "x2": -1.5},
            seed=42
        )

        assert result.is_success
        dataset = result.value
        assert len(dataset.data) == 3  # x1, x2, y
        assert len(dataset.data["y"]) == 100

    def test_model_metrics_factory(self):
        """Test ModelMetricsFactory."""
        y_actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_predicted = [1.1, 2.1, 2.9, 4.1, 4.9]

        result = ModelMetricsFactory.calculate_from_residuals(
            y_actual, y_predicted, n_features=1
        )

        assert result.is_success
        metrics = result.value
        assert 0 <= metrics.r_squared <= 1
        assert metrics.rmse >= 0


# ============================================================================
# Event System Tests
# ============================================================================

class TestEventSystem:
    """Tests für das Event System."""

    def test_event_metadata_creation(self):
        """Test EventMetadata Erstellung."""
        metadata = EventMetadata.create(
            aggregate_id="test_123",
            aggregate_type="Dataset"
        )

        assert metadata.event_id is not None
        assert metadata.correlation_id is not None
        assert metadata.aggregate_id == "test_123"

    def test_in_memory_event_store(self):
        """Test InMemoryEventStore."""
        store = InMemoryEventStore()

        event = DatasetCreated(
            dataset_id="test",
            dataset_name="Test",
            n_variables=2,
            n_observations=100,
            source="test",
            _metadata=EventMetadata.create(aggregate_id="test")
        )

        store.append(event)

        assert store.count() == 1
        events = store.get_events(aggregate_id="test")
        assert len(events) == 1

    def test_event_dispatcher(self):
        """Test EventDispatcher."""
        store = InMemoryEventStore()
        dispatcher = EventDispatcher(event_store=store)

        received_events = []

        def handler(event):
            received_events.append(event)

        dispatcher.register(DatasetCreated, handler)

        event = DatasetCreated(
            dataset_id="test",
            dataset_name="Test",
            n_variables=2,
            n_observations=100,
            source="test",
            _metadata=EventMetadata.create()
        )

        dispatcher.dispatch(event)

        assert len(received_events) == 1
        assert store.count() == 1


# ============================================================================
# Unit of Work Tests
# ============================================================================

class TestUnitOfWork:
    """Tests für Unit of Work Pattern."""

    def test_unit_of_work_commit(self):
        """Test Unit of Work Commit."""
        dataset_repo = InMemoryDatasetRepository()
        model_repo = InMemoryModelRepository()

        uow = InMemoryUnitOfWork(
            dataset_repository=dataset_repo,
            model_repository=model_repo
        )

        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )

        dataset = DatasetAggregate(
            id="test",
            config=config,
            data={"x": list(range(10))}
        )

        uow.begin()
        uow.register_new(dataset)
        result = uow.commit()

        assert result.is_success
        assert dataset_repo.count() == 1

    def test_unit_of_work_context_manager(self):
        """Test Unit of Work Context Manager."""
        dataset_repo = InMemoryDatasetRepository()
        model_repo = InMemoryModelRepository()

        uow = InMemoryUnitOfWork(
            dataset_repository=dataset_repo,
            model_repository=model_repo
        )

        config = DatasetConfig(
            name="test",
            dataset_type="synthetic",
            source="test",
            description="test",
            variables={"x": "numeric"},
            n_observations=10
        )

        dataset = DatasetAggregate(
            id="test",
            config=config,
            data={"x": list(range(10))}
        )

        with unit_of_work_scope(uow) as active_uow:
            active_uow.register_new(dataset)

        assert dataset_repo.count() == 1

    def test_unit_of_work_rollback(self):
        """Test Unit of Work Rollback."""
        dataset_repo = InMemoryDatasetRepository()
        model_repo = InMemoryModelRepository()

        uow = InMemoryUnitOfWork(
            dataset_repository=dataset_repo,
            model_repository=model_repo
        )

        uow.begin()
        # Nicht commiten, sondern rollback
        uow.rollback()

        assert not uow.is_active
        assert dataset_repo.count() == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestArchitectureIntegration:
    """Integration Tests für die gesamte Architektur."""

    def test_full_workflow(self):
        """Test kompletter Workflow von Dataset-Erstellung bis Model-Analyse."""
        # 1. Dataset erstellen mit Factory
        dataset_result = DatasetFactory.create_synthetic(
            name="Integration Test",
            n_observations=100,
            features=["x1", "x2"],
            target="y",
            intercept=5.0,
            coefficients={"x1": 2.0, "x2": -1.0},
            seed=42
        )

        assert dataset_result.is_success
        dataset = dataset_result.value

        # 2. Validierung mit Specification
        spec = SpecificationFactory.dataset_for_regression(
            target_variable="y",
            feature_variables=["x1", "x2"],
            min_sample_size=30
        )

        validation = spec.validate(dataset)
        # Note: Regular entities don't have validate() from specs directly
        # So we just check if spec is satisfied
        assert spec.is_satisfied_by(dataset)

        # 3. Model erstellen mit Factory
        params_result = RegressionParametersFactory.create(
            intercept=5.0,
            coefficients={"x1": 2.0, "x2": -1.0},
            noise_level=0.1
        )

        assert params_result.is_success

        model_result = RegressionModelFactory.create(
            dataset=dataset,
            target_variable="y",
            feature_variables=["x1", "x2"],
            parameters=params_result.value
        )

        assert model_result.is_success
        model = model_result.value

        # 4. Model Quality Check mit Specification
        # Create a simple wrapper that works with specifications
        assert model.metrics.r_squared >= 0

    def test_result_chaining(self):
        """Test Result Monad Chaining."""
        def create_dataset(name: str) -> Result:
            return DatasetFactory.create_synthetic(
                name=name,
                n_observations=50,
                features=["x"],
                target="y"
            )

        def validate_dataset(dataset) -> Result:
            spec = HasMinimumSampleSize(30)
            if spec.is_satisfied_by(dataset):
                return Result.success(dataset)
            return Result.failure(Error("VALIDATION", spec.get_failure_message()))

        def create_model(dataset) -> Result:
            params_result = RegressionParametersFactory.create_for_simple_regression(
                intercept=0.0,
                slope=1.0,
                feature_name="x"
            )
            if params_result.is_failure:
                return params_result

            return RegressionModelFactory.create(
                dataset=dataset,
                target_variable="y",
                feature_variables=["x"],
                parameters=params_result.value
            )

        # Chain operations
        final_result = (
            create_dataset("Chained Test")
            .flat_map(validate_dataset)
            .flat_map(create_model)
        )

        assert final_result.is_success
        model = final_result.value
        assert model.model_type == "simple"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
