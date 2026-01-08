# 🏗️ Architektur-Dokumentation

**Linear Regression Guide - Clean Architecture**

Diese Dokumentation beschreibt die Architektur der Anwendung nach der Migration zu Clean Architecture.

---

## 📋 Inhaltsverzeichnis

1. [Architektur-Übersicht](#architektur-übersicht)
2. [Layer-Struktur](#layer-struktur)
3. [Datenfluss](#datenfluss)
4. [Module im Detail](#module-im-detail)
5. [Design-Prinzipien](#design-prinzipien)
6. [Code-Beispiele](#code-beispiele)

---

## 🏛️ Architektur-Übersicht

Die Anwendung folgt strikt der **Clean Architecture** mit vier Schichten:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           src/                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CORE (Pure Python)                                │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────┐   │   │
│  │  │  core/domain/           │  │  core/application/              │   │   │
│  │  │  • entities.py          │  │  • use_cases.py                 │   │   │
│  │  │  • value_objects.py     │  │  • dtos.py                      │   │   │
│  │  │  • interfaces.py        │  │                                 │   │   │
│  │  │  (No numpy/pandas!)     │  │  (Orchestration only)           │   │   │
│  │  └─────────────────────────┘  └─────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│                            (implements)                                     │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │               INFRASTRUCTURE (External Dependencies)                 │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌────────────────┐   │   │
│  │  │  data/            │  │  services/        │  │  content/      │   │   │
│  │  │  • generators.py  │  │  • calculate.py   │  │  • builder.py  │   │   │
│  │  │  • provider.py    │  │  • plot.py        │  │                │   │   │
│  │  │  (numpy, pandas)  │  │  (scipy)          │  │                │   │   │
│  │  └───────────────────┘  └───────────────────┘  └────────────────┘   │   │
│  │  ┌───────────────────┐  ┌───────────────────┐                       │   │
│  │  │  ai/              │  │  regression_      │                       │   │
│  │  │  perplexity.py    │  │  pipeline.py      │                       │   │
│  │  └───────────────────┘  └───────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│                               (uses)                                        │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    INTERFACE ADAPTERS                                │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌────────────────┐   │   │
│  │  │  api/             │  │  adapters/        │  │  container.py  │   │   │
│  │  │  • endpoints.py   │  │  • flask_app.py   │  │  (DI Wiring)   │   │   │
│  │  │  • serializers.py │  │  • streamlit/     │  │                │   │   │
│  │  └───────────────────┘  └───────────────────┘  └────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Layer-Struktur

### Layer 1: Core Domain (`src/core/domain/`)

**PURE PYTHON - Keine externen Abhängigkeiten!**

| Datei | Zweck |
|-------|-------|
| `entities.py` | `RegressionModel` - Entität mit Identität |
| `value_objects.py` | `RegressionParameters`, `RegressionMetrics`, `DatasetMetadata` |
| `interfaces.py` | `IDataProvider`, `IRegressionService` (Protocol) |

**Regeln:**
- ✅ Nur Python Standard Library
- ❌ Kein `numpy`, `pandas`, `scipy`, `datetime`
- ❌ Keine Framework-Abhängigkeiten

### Layer 2: Core Application (`src/core/application/`)

**Use Cases & DTOs**

| Datei | Zweck |
|-------|-------|
| `use_cases.py` | `RunRegressionUseCase` - Orchestrierung |
| `dtos.py` | `RegressionRequestDTO`, `RegressionResponseDTO` |

**Regeln:**
- ✅ Importiert nur aus `core/domain`
- ✅ Orchestriert, implementiert keine Business-Logik
- ❌ Keine direkten Abhängigkeiten zu Infrastructure

### Layer 3: Infrastructure (`src/infrastructure/`)

**Konkrete Implementierungen**

| Modul | Zweck |
|-------|-------|
| `data/generators.py` | Datengenerierung (numpy) |
| `data/provider.py` | `DataProviderImpl` implementiert `IDataProvider` |
| `services/calculate.py` | `StatisticsCalculator` - OLS, R², t-Tests |
| `services/plot.py` | `PlotBuilder` - Plotly Visualisierungen |
| `services/regression.py` | `RegressionServiceImpl` implementiert `IRegressionService` |
| `content/` | Edukativer Content Builder |
| `ai/` | Perplexity AI Client |
| `regression_pipeline.py` | 4-Step Pipeline Orchestrierung |

**Regeln:**
- ✅ Implementiert Interfaces aus `core/domain`
- ✅ Darf externe Libraries nutzen (numpy, scipy, plotly)
- ❌ Keine Framework-spezifische UI-Logik

### Layer 4: Interface Adapters

**Framework-spezifischer Code**

| Modul | Framework |
|-------|-----------|
| `api/endpoints.py` | REST API (Framework-agnostisch) |
| `api/serializers.py` | JSON Serialisierung |
| `adapters/flask_app.py` | Flask HTML App |
| `adapters/streamlit/` | Streamlit Interactive App |
| `container.py` | Dependency Injection Container |

---

## 🔄 Datenfluss

### Clean Architecture Flow (Use Case)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. API/Controller                                                            │
│    RegressionRequestDTO { dataset_id="electronics", n=50, ... }             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Container                                                                 │
│    container.run_regression_use_case                                        │
│    (injects: DataProviderImpl, RegressionServiceImpl)                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Use Case (Orchestration)                                                  │
│    RunRegressionUseCase.execute(request)                                    │
│    ├─ data_provider.get_dataset() → raw data                                │
│    ├─ regression_service.train_simple() → RegressionModel                   │
│    └─ _build_response() → RegressionResponseDTO                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Response DTO                                                              │
│    RegressionResponseDTO { r_squared=0.91, slope=0.51, predictions=[...] }  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Legacy Pipeline Flow (Still Supported)

```
RegressionPipeline.run_simple()
    ├─ DataFetcher.get_simple() → DataResult
    ├─ StatisticsCalculator.simple_regression() → RegressionResult
    └─ PlotBuilder.simple_regression_plots() → PlotCollection
```

---

## 📦 Module im Detail

### Domain Value Objects

```python
@dataclass(frozen=True)
class RegressionParameters:
    intercept: float
    coefficients: Dict[str, float]

@dataclass(frozen=True)
class RegressionMetrics:
    r_squared: float
    r_squared_adj: float
    mse: float
    rmse: float
```

### Domain Entity

```python
@dataclass
class RegressionModel:
    id: str
    parameters: Optional[RegressionParameters]
    metrics: Optional[RegressionMetrics]
    
    def is_trained(self) -> bool:
        return self.parameters is not None
    
    def get_equation_string(self) -> str:
        # Pure Python business logic
```

### Use Case

```python
class RunRegressionUseCase:
    def __init__(self, data_provider: IDataProvider, regression_service: IRegressionService):
        self.data_provider = data_provider
        self.regression_service = regression_service
    
    def execute(self, request: RegressionRequestDTO) -> RegressionResponseDTO:
        # Orchestrate only - no calculations here
```

### DI Container

```python
class Container:
    def __init__(self):
        self._data_provider = DataProviderImpl()
        self._regression_service = RegressionServiceImpl()
    
    @property
    def run_regression_use_case(self) -> RunRegressionUseCase:
        return RunRegressionUseCase(
            data_provider=self._data_provider,
            regression_service=self._regression_service
        )
```

---

## 🎯 Design-Prinzipien

### 1. Dependency Inversion

Domain definiert Interfaces, Infrastructure implementiert sie:

```python
# Domain (interfaces.py)
class IDataProvider(Protocol):
    def get_dataset(self, dataset_id: str, n: int, **kwargs) -> Dict[str, Any]: ...

# Infrastructure (provider.py)
class DataProviderImpl(IDataProvider):
    def get_dataset(self, dataset_id: str, n: int, **kwargs) -> Dict[str, Any]:
        # Concrete implementation with numpy
```

### 2. Layer Isolation

```
Adapters → API → Application → Domain ← Infrastructure
```

- Domain kennt niemanden
- Application kennt nur Domain
- Infrastructure implementiert Domain-Interfaces
- Adapters kann alles importieren

### 3. Pure Domain

```python
# ❌ VERBOTEN in core/domain:
import numpy as np
from datetime import datetime

# ✅ ERLAUBT in core/domain:
from dataclasses import dataclass
from typing import Dict, List, Optional
```

---

## 🧪 Testing

```bash
# Unit Tests (alle Layer)
pytest tests/unit/ -v

# Use Case Test
pytest tests/unit/test_pipeline.py::TestCleanArchitectureUseCase -v

# Validation: No external deps in domain
grep -r "import numpy\|import pandas" src/core/
# Should return nothing!
```

---

## 📚 Weiterführende Dokumentation

- **[API.md](API.md)** - REST API Dokumentation
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Frontend-Integration
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment-Anleitung
