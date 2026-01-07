# Core Architecture Improvements

## Übersicht

Die Core-Architektur wurde umfassend verbessert und implementiert nun fortgeschrittene Domain-Driven Design (DDD) und Clean Architecture Patterns.

## Neue Architektur-Patterns

### 1. Result Pattern (`src/core/domain/result.py`)

Das Result Pattern ermöglicht funktionale Fehlerbehandlung ohne Exceptions:

```python
from src.core.domain.result import Result, Error

# Erfolgreiches Ergebnis
result = Result.success(42)
if result.is_success:
    value = result.value

# Fehlgeschlagenes Ergebnis
result = Result.failure(Error("CODE", "Fehlermeldung"))
if result.is_failure:
    error = result.error

# Verkettung mit map und flat_map
final = (
    Result.success(5)
    .map(lambda x: x * 2)
    .ensure(lambda x: x > 5, Error("TOO_SMALL", "Wert zu klein"))
)
```

**Vorteile:**
- Explizite Fehlerbehandlung
- Keine versteckten Exceptions
- Monadische Operationen (map, flat_map)
- Kombinieren mehrerer Results

### 2. Specification Pattern (`src/core/domain/specifications.py`)

Wiederverwendbare, kombinierbare Geschäftsregeln:

```python
from src.core.domain.specifications import (
    HasMinimumSampleSize,
    HasRequiredVariables,
    SpecificationFactory
)

# Einzelne Spezifikation
spec = HasMinimumSampleSize(30)
if spec.is_satisfied_by(dataset):
    print("Dataset erfüllt Anforderung")

# Kombinierte Spezifikationen mit &, |, ~
combined = (
    HasMinimumSampleSize(30) &
    HasRequiredVariables(["x", "y"]) &
    ~HasConstantVariables()
)

# Factory für häufige Kombinationen
spec = SpecificationFactory.dataset_for_regression(
    target_variable="y",
    feature_variables=["x1", "x2"],
    min_sample_size=30
)
```

**Verfügbare Specifications:**
- `HasMinimumSampleSize`: Prüft Mindestgröße
- `HasRequiredVariables`: Prüft Vorhandensein von Variablen
- `HasNoMissingValues`: Prüft auf fehlende Werte
- `HasSufficientVariation`: Prüft auf Variation
- `IsStatisticallySignificant`: Prüft F-Test Signifikanz
- `HasMinimumRSquared`: Prüft R²-Wert
- `IsProductionReady`: Kombinierte Produktions-Prüfung

### 3. Aggregate Root Pattern (`src/core/domain/aggregates.py`)

Aggregates als Konsistenzgrenzen mit Event-Sammlung:

```python
from src.core.domain.aggregates import DatasetAggregate, RegressionModelAggregate

# Aggregate erstellen
result = DatasetAggregate.create(
    id="dataset_1",
    config=config,
    data={"x": [1,2,3], "y": [2,4,6]}
)

if result.is_success:
    aggregate = result.value
    
    # Operationen auf dem Aggregate
    aggregate.add_variable("z", [1,2,3])
    
    # Events sammeln sich automatisch
    events = aggregate.uncommitted_events
    
    # Version für Optimistic Locking
    version = aggregate.version
```

**Features:**
- Automatische Validierung bei Erstellung
- Event-Sammlung für Domain Events
- Version Tracking für Optimistic Locking
- Factory-Methoden für konsistente Erstellung

### 4. Domain Events mit Event Store (`src/core/domain/events.py`)

Verbesserte Event-Architektur mit Metadata und Persistenz:

```python
from src.core.domain.events import (
    DatasetCreated,
    EventDispatcher,
    InMemoryEventStore
)

# Event Store für Persistenz
store = InMemoryEventStore()

# Event Dispatcher mit automatischer Speicherung
dispatcher = EventDispatcher(event_store=store)

# Handler registrieren
def on_dataset_created(event):
    print(f"Dataset erstellt: {event.dataset_id}")

dispatcher.register(DatasetCreated, on_dataset_created)

# Events haben automatische Metadata
event = DatasetCreated(
    dataset_id="ds_1",
    dataset_name="Test",
    n_variables=2,
    n_observations=100,
    source="generated"
)

print(event.event_id)  # Unique ID
print(event.metadata.correlation_id)  # Für Tracing
```

**Event-Typen:**
- `DatasetCreated`, `DatasetUpdated`, `DatasetDeleted`
- `RegressionModelCreated`, `RegressionModelValidated`
- `ModelsCompared`, `ModelPredictionMade`

### 5. Unit of Work Pattern (`src/core/domain/unit_of_work.py`)

Transaktionale Konsistenz über mehrere Aggregates:

```python
from src.core.domain.unit_of_work import InMemoryUnitOfWork, unit_of_work_scope

# Mit Context Manager
with unit_of_work_scope(uow) as active_uow:
    active_uow.register_new(dataset_aggregate)
    active_uow.register_new(model_aggregate)
    # Commit erfolgt automatisch bei Erfolg

# Manuell
uow.begin()
uow.register_new(aggregate)
uow.register_dirty(modified_aggregate)
result = uow.commit()

if result.is_failure:
    print(f"Commit fehlgeschlagen: {result.error}")
```

**Features:**
- Atomare Operationen über mehrere Aggregates
- Automatischer Rollback bei Fehlern
- Event-Sammlung und -Dispatch beim Commit
- Version-Inkrement bei erfolgreichem Commit

### 6. CQRS Handler (`src/core/application/handlers.py`)

Getrennte Command- und Query-Handler:

```python
from src.core.application.handlers import (
    CreateRegressionModelHandler,
    GetDatasetByIdHandler,
    Mediator,
    create_mediator
)

# Mediator erstellen
mediator = create_mediator(
    dataset_repository=repo,
    model_repository=model_repo,
    event_dispatcher=dispatcher
)

# Command senden
result = mediator.send_command(CreateRegressionModelCommand(
    dataset_id="ds_1",
    target_variable="y",
    feature_variables=["x"],
    parameters=params
))

# Query senden
result = mediator.send_query(GetDatasetByIdQuery(dataset_id="ds_1"))
```

### 7. Factory Pattern (`src/core/domain/factories.py`)

Konsistente Objekt-Erstellung mit Validierung:

```python
from src.core.domain.factories import (
    DatasetFactory,
    RegressionModelFactory,
    RegressionParametersFactory
)

# Synthetische Daten generieren
result = DatasetFactory.create_synthetic(
    name="Test Dataset",
    n_observations=100,
    features=["x1", "x2"],
    target="y",
    intercept=5.0,
    coefficients={"x1": 2.0, "x2": -1.0}
)

# Parameter erstellen
params = RegressionParametersFactory.create(
    intercept=5.0,
    coefficients={"x": 2.5},
    noise_level=0.1
)

# Model aus Dataset erstellen
model = RegressionModelFactory.create(
    dataset=dataset,
    target_variable="y",
    feature_variables=["x"],
    parameters=params.value
)
```

## Architektur-Struktur

```
src/core/
├── domain/                    # Reine Geschäftslogik
│   ├── __init__.py           # Exports
│   ├── aggregates.py         # Aggregate Roots
│   ├── entities.py           # Domain Entities
│   ├── events.py             # Domain Events + Event Store
│   ├── factories.py          # Object Factories
│   ├── repositories.py       # Repository Interfaces
│   ├── result.py             # Result Monad
│   ├── services.py           # Domain Services
│   ├── specifications.py     # Specification Pattern
│   ├── unit_of_work.py       # Unit of Work
│   └── value_objects.py      # Value Objects
│
└── application/               # Anwendungslogik
    ├── __init__.py           # Exports
    ├── application_services.py
    ├── commands.py           # CQRS Commands
    ├── event_handlers.py     # Event Handler
    ├── handlers.py           # Command/Query Handlers + Mediator
    ├── queries.py            # CQRS Queries
    └── use_cases.py          # Business Use Cases
```

## Best Practices

### 1. Immer Result verwenden
```python
# ❌ Schlecht: Exception werfen
def create_dataset(data):
    if not data:
        raise ValueError("No data")
    return Dataset(data)

# ✅ Gut: Result zurückgeben
def create_dataset(data) -> Result[Dataset]:
    if not data:
        return Result.failure(Error("NO_DATA", "No data provided"))
    return Result.success(Dataset(data))
```

### 2. Specifications für Validierung
```python
# ❌ Schlecht: Validierung im Code verstreut
if len(dataset) < 30:
    raise ValueError("Not enough data")
if "y" not in dataset:
    raise ValueError("Missing target")

# ✅ Gut: Specifications verwenden
spec = HasMinimumSampleSize(30) & HasRequiredVariables(["y"])
validation = spec.validate(dataset)
if validation.is_invalid:
    return validation.to_result(None)
```

### 3. Aggregates für Konsistenz
```python
# ❌ Schlecht: Direkte Entity-Manipulation
dataset.data["x"] = values
dataset.data["y"] = values

# ✅ Gut: Über Aggregate-Methoden
result = aggregate.add_variable("x", values)
if result.is_failure:
    handle_error(result.error)
```

### 4. Unit of Work für Transaktionen
```python
# ❌ Schlecht: Einzelne Saves
repo.save(dataset)
repo.save(model)  # Was wenn das fehlschlägt?

# ✅ Gut: Unit of Work
with unit_of_work_scope(uow) as active:
    active.register_new(dataset)
    active.register_new(model)
    # Alles oder nichts
```

## Tests

Die neue Architektur wird durch 45 Unit Tests abgedeckt:

```bash
pytest tests/unit/test_architecture.py -v
```

Testkategorien:
- Result Pattern Tests (10)
- ValidationResult Tests (6)
- Specification Pattern Tests (6)
- Aggregate Root Tests (4)
- Factory Pattern Tests (5)
- Event System Tests (3)
- Unit of Work Tests (3)
- Integration Tests (2)

## Migration

Für bestehenden Code:

1. **Ersetze Exceptions durch Results**
2. **Verwende Specifications für Validierung**
3. **Nutze Factories für Objekt-Erstellung**
4. **Verwende Aggregates statt direkte Entity-Manipulation**
5. **Nutze Unit of Work für zusammengehörige Operationen**
