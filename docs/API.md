# 📚 Regression Analysis API Documentation

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Content-Type:** `application/json`

Diese API ermöglicht es Entwicklern, die Regression Analysis Plattform in ihre eigenen Anwendungen zu integrieren - egal ob Next.js, Vue, React, Angular, Mobile Apps oder jedes andere Frontend.

---

## 🚀 Quick Start

### 1. Server starten

```bash
# API-Server starten
python3 run.py --api --port 8000

# Alternativ: Flask mit API-Endpunkten
python3 run.py --flask --port 5000
```

### 2. Erste Anfrage

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "framework": "flask",
  "api_powered": true
}
```

### 3. Regression ausführen

```bash
curl -X POST http://localhost:8000/api/regression/simple \
  -H "Content-Type: application/json" \
  -d '{"dataset": "electronics", "n": 50}'
```

---

## 📋 Inhaltsverzeichnis

1. [Authentifizierung](#authentifizierung)
2. [Basis-Endpunkte](#basis-endpunkte)
3. [Regression Endpunkte](#regression-endpunkte)
4. [Content Endpunkte](#content-endpunkte)
5. [AI Endpunkte](#ai-endpunkte)
6. [Datenstrukturen](#datenstrukturen)
7. [Fehlerbehandlung](#fehlerbehandlung)
8. [Integration Beispiele](#integration-beispiele)
9. [Best Practices](#best-practices)

---

## 🔐 Authentifizierung

Die API benötigt **keine Authentifizierung** für den lokalen Betrieb.

Für Produktionsumgebungen empfehlen wir:
- API-Key Header: `X-API-Key: your-api-key`
- Rate Limiting implementieren
- CORS entsprechend konfigurieren

### CORS

Die API erlaubt standardmäßig alle Origins (`*`). Für Produktion:

```python
from src.api import create_api_server
app = create_api_server(cors_origins=[
    "https://your-domain.com",
    "http://localhost:3000"
])
```

---

## 🔧 Basis-Endpunkte

### Health Check

Prüft ob der Server läuft.

```
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "framework": "flask",
  "api_powered": true
}
```

---

### OpenAPI Specification

Maschinenlesbare API-Spezifikation.

```
GET /api/openapi.json
```

**Response:** OpenAPI 3.0 JSON Specification

---

### Verfügbare Datasets

Listet alle verfügbaren Datensätze.

```
GET /api/datasets
```

**Response:**
```json
{
  "success": true,
  "data": {
    "simple": [
      {
        "id": "electronics",
        "name": "Elektronikmarkt",
        "description": "Verkaufsfläche vs Umsatz",
        "icon": "🏪"
      },
      {
        "id": "advertising",
        "name": "Werbestudie",
        "description": "Werbeausgaben vs Umsatz",
        "icon": "📢"
      },
      {
        "id": "temperature",
        "name": "Eisverkauf",
        "description": "Temperatur vs Verkauf",
        "icon": "🍦"
      }
    ],
    "multiple": [
      {
        "id": "cities",
        "name": "Städtestudie",
        "description": "Preis & Werbung → Umsatz",
        "icon": "🏙️"
      },
      {
        "id": "houses",
        "name": "Hauspreise",
        "description": "Fläche & Pool → Preis",
        "icon": "🏠"
      }
    ]
  }
}
```

### Datensatz Rohdaten (Data Explorer)
Hole Rohdaten für einen spezifischen Datensatz.

```
GET /api/datasets/{dataset_id}/raw
```

**Response:**
```json
{
  "success": true,
  "data": {
    "columns": ["Feature 1", "Feature 2", "Target"],
    "data": [
      [1.2, 3.4, 0],
      [2.3, 4.5, 1]
    ],
    "metadata": { ... }
  }
}
```

---

## 📈 Regression Endpunkte

### Simple Regression

Führt eine einfache lineare Regression durch.

```
POST /api/regression/simple
```

**Request Body:**
| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `dataset` | string | `"electronics"` | Dataset ID |
| `n` | integer | `50` | Anzahl Datenpunkte |
| `noise` | float | `0.4` | Rausch-Level |
| `seed` | integer | `42` | Random Seed |
| `include_predictions` | boolean | `true` | Vorhersagen inkludieren |

**Beispiel:**
```bash
curl -X POST http://localhost:8000/api/regression/simple \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "electronics",
    "n": 100,
    "noise": 0.5,
    "seed": 123
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "type": "simple",
    "data": {
      "type": "simple_regression_data",
      "x": [2.1, 3.5, 4.2, ...],
      "y": [1.2, 2.1, 2.8, ...],
      "n": 100,
      "x_label": "Verkaufsfläche (100 qm)",
      "y_label": "Umsatz (Mio. €)",
      "context": {
        "title": "🏪 Elektronikmarkt",
        "description": "Analyse des Zusammenhangs..."
      }
    },
    "stats": {
      "type": "simple_regression_stats",
      "coefficients": {
        "intercept": 0.6234,
        "slope": 0.4891
      },
      "model_fit": {
        "r_squared": 0.8923,
        "r_squared_adj": 0.8912
      },
      "standard_errors": {
        "intercept": 0.1234,
        "slope": 0.0456
      },
      "t_tests": {
        "intercept": {
          "t_value": 5.052,
          "p_value": 0.00001
        },
        "slope": {
          "t_value": 10.723,
          "p_value": 0.00000
        }
      },
      "sum_of_squares": {
        "sse": 12.34,
        "sst": 115.67,
        "ssr": 103.33,
        "mse": 0.126
      },
      "sample": {
        "n": 100,
        "df": 98
      },
      "predictions": [1.15, 2.03, 2.67, ...],
      "residuals": [0.05, 0.07, 0.13, ...]
    },
    "plots": {
      "scatter": { /* Plotly JSON */ },
      "residuals": { /* Plotly JSON */ },
      "diagnostics": { /* Plotly JSON */ }
    },
    "params": {
      "dataset": "electronics",
      "n": 100,
      "noise": 0.5,
      "seed": 123
    }
  }
}
```

---

### Multiple Regression

Führt eine multiple lineare Regression durch.

```
POST /api/regression/multiple
```

**Request Body:**
| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `dataset` | string | `"cities"` | Dataset ID |
| `n` | integer | `75` | Anzahl Datenpunkte |
| `noise` | float | `3.5` | Rausch-Level |
| `seed` | integer | `42` | Random Seed |

**Beispiel:**
```bash
curl -X POST http://localhost:8000/api/regression/multiple \
  -H "Content-Type: application/json" \
  -d '{"dataset": "houses", "n": 100}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "type": "multiple",
    "data": {
      "type": "multiple_regression_data",
      "x1": [...],
      "x2": [...],
      "y": [...],
      "n": 100,
      "x1_label": "Wohnfläche (sqft/10)",
      "x2_label": "Pool (0/1)",
      "y_label": "Preis ($1000)"
    },
    "stats": {
      "type": "multiple_regression_stats",
      "coefficients": {
        "intercept": 52.34,
        "slopes": [7.89, 28.45]
      },
      "model_fit": {
        "r_squared": 0.7234,
        "r_squared_adj": 0.7178,
        "f_statistic": 126.78,
        "f_p_value": 0.00000
      },
      "standard_errors": [5.23, 0.34, 4.56],
      "t_tests": {
        "t_values": [10.01, 23.21, 6.24],
        "p_values": [0.0001, 0.0000, 0.0001]
      },
      "sample": {
        "n": 100,
        "k": 2
      }
    },
    "plots": {
      "scatter": { /* 3D Plotly JSON */ },
      "residuals": { /* Plotly JSON */ },
      "diagnostics": { /* Plotly JSON */ }
    }
  }
}
```

---

---

## 🤖 Classification Endpunkte (ML Bridge)

### Classification Analysis
Führt eine Klassifikationsanalyse (Logistic Regression oder KNN) durch.

```
POST /api/classification
```

**Request Body:**
| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `dataset` | string | `"fruits"` | Dataset ID (`fruits`, `digits`, `binary_electronics`) |
| `n` | integer | `100` | Anzahl Samples (max 500) |
| `noise` | float | `0.2` | Rausch-Level |
| `seed` | integer | `42` | Random Seed |
| `method` | string | `"logistic"` | Methode: `"logistic"` oder `"knn"` |
| `k` | integer | `3` | Anzahl Nachbarn (nur für KNN) |

**Beispiel:**
```bash
curl -X POST http://localhost:8000/api/classification \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "fruits",
    "method": "knn",
    "k": 5
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "X": [[7.5, 7.3, 175.0, 0.75], ...],
    "y": [0, 0, 1, 2, ...],
    "feature_names": ["height", "width", "mass", "color"],
    "target_names": ["apple", "mandarin", "orange", "lemon"]
  },
  "results": {
    "metrics": {
      "accuracy": 0.92,
      "precision": 0.91,
      "recall": 0.93,
      "f1": 0.92,
      "confusion_matrix": [[10, 1, 0, 0], ...],
      "auc": null
    },
    "params": {
      "k": 5,
      "method": "knn"
    }
  },
  "plots": {
    "scatter": { /* 3D Classification Surface & Points */ },
    "residuals": { /* 3D Confusion Matrix */ },
    "diagnostics": { /* 3D ROC Curve */ }
  },
  "metadata": {
    "dataset": "🍎 Fruit Classification",
    "description": "KNN Case Study..."
  }
}
```

---

## 📖 Content Endpunkte

### Educational Content (Simple)

Holt vollständigen edukativen Content für einfache Regression.

```
POST /api/content/simple
```

**Request Body:**
```json
{
  "dataset": "electronics",
  "n": 50,
  "noise": 0.4,
  "seed": 42
}
```

**Response:**
```json
{
  "success": true,
  "content": {
    "title": "📊 Einfache Lineare Regression",
    "subtitle": "Statistisches Lernen mit interaktiven Visualisierungen",
    "chapters": [
      {
        "type": "chapter",
        "number": "1.0",
        "title": "Einleitung - Die Analyse von Zusammenhängen",
        "icon": "📖",
        "sections": [
          {
            "type": "info_box",
            "content": "**Kontext:** 🏪 Elektronikmarkt\n\n..."
          },
          {
            "type": "markdown",
            "text": "### 🎯 Lernziele dieses Moduls\n\n..."
          },
          {
            "type": "metric_row",
            "metrics": [
              {"type": "metric", "label": "R²", "value": "0.9114", "help_text": "Erklärte Varianz"},
              {"type": "metric", "label": "β₀", "value": "0.6610", "help_text": "Y-Achsenabschnitt"},
              {"type": "metric", "label": "β₁", "value": "0.5088", "help_text": "Steigung"}
            ]
          }
        ]
      },
      // ... 10 weitere Kapitel
    ]
  },
  "plots": {
    "scatter": { /* Plotly JSON */ },
    "residuals": { /* Plotly JSON */ },
    "diagnostics": { /* Plotly JSON */ },
    "extra": {
      "histogram": { /* Plotly JSON */ }
    }
  },
  "stats": { /* Statistiken (siehe Regression API) */ },
  "data": { /* Rohdaten */ }
}
```

---

### Educational Content (Multiple)

Holt vollständigen edukativen Content für multiple Regression.

```
POST /api/content/multiple
```

**Request/Response:** Analog zu `/api/content/simple`, aber mit 9 Kapiteln für multiple Regression.

---

### Content Schema

Dokumentiert alle verfügbaren Content-Element-Typen.

```
GET /api/content/schema
```

**Response:**
```json
{
  "success": true,
  "schema": {
    "element_types": [
      "markdown", "metric", "metric_row", "formula", "plot",
      "table", "columns", "expander", "info_box", "warning_box",
      "success_box", "code_block", "divider", "chapter", "section"
    ],
    "structure": {
      "EducationalContent": {
        "title": "string",
        "subtitle": "string",
        "chapters": "Chapter[]"
      },
      "Chapter": {
        "number": "string",
        "title": "string",
        "icon": "string (emoji)",
        "sections": "Section[] | ContentElement[]"
      },
      // ... weitere Strukturen
    }
  }
}
```

---

## 🤖 AI Endpunkte

### AI Status

Prüft ob die AI-Integration konfiguriert ist.

```
GET /api/ai/status
```

**Response:**
```json
{
  "success": true,
  "status": {
    "configured": true,
    "model": "llama-3.1-sonar-small-128k-online",
    "cache_size": 3
  }
}
```

---

### AI Interpretation

Interpretiert Regressionsergebnisse mit Perplexity AI.

```
POST /api/ai/interpret
```

**Request Body:**
```json
{
  "stats": {
    "intercept": 0.6610,
    "slope": 0.5088,
    "r_squared": 0.9114,
    "r_squared_adj": 0.9096,
    "se_intercept": 0.1378,
    "se_slope": 0.0245,
    "t_intercept": 4.797,
    "t_slope": 20.77,
    "p_intercept": 0.00002,
    "p_slope": 0.00000,
    "n": 50,
    "df": 48,
    "mse": 0.0634,
    "x_label": "Verkaufsfläche",
    "y_label": "Umsatz"
  },
  "use_cache": true
}
```

**Response:**
```json
{
  "success": true,
  "interpretation": {
    "content": "## 📊 Interpretation der Regressionsanalyse\n\n### 1. Zusammenfassung\nDas Modell zeigt einen **starken positiven** Zusammenhang...",
    "model": "llama-3.1-sonar-small-128k-online",
    "cached": false,
    "latency_ms": 2341.5
  },
  "usage": {
    "prompt_tokens": 456,
    "completion_tokens": 892,
    "total_tokens": 1348
  },
  "citations": [
    "https://en.wikipedia.org/wiki/Linear_regression"
  ]
}
```

**Ohne API-Key:**
Die API gibt eine Fallback-Interpretation zurück, ohne Perplexity AI zu nutzen.

---

### R-Output generieren

Generiert einen R-Style Output aus Statistiken.

```
POST /api/ai/r-output
```

**Request Body:**
```json
{
  "stats": {
    "intercept": 0.6610,
    "slope": 0.5088,
    "x_label": "Verkaufsfläche",
    "y_label": "Umsatz",
    "n": 50,
    "df": 48,
    // ... weitere Stats
  }
}
```

**Response:**
```json
{
  "success": true,
  "r_output": "Call:\nlm(formula = Umsatz ~ Verkaufsfläche)\n\nResiduals:\n     Min       1Q   Median       3Q      Max \n -0.6161  -0.1056  -0.0195   0.1148   0.2902\n\nCoefficients:\n              Estimate Std. Error t value Pr(>|t|)    \n(Intercept)     0.6610     0.1378   4.797   1.61e-05 ***\nVerkaufsfläche  0.5088     0.0245  20.770   < 2e-16 ***\n---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n\nResidual standard error: 0.2519 on 48 degrees of freedom\nMultiple R-squared:  0.9114,    Adjusted R-squared:  0.9096\nF-statistic: 431.4 on 1 and 48 DF,  p-value: < 2.2e-16"
}
```

---

### Cache leeren

Löscht den Interpretation-Cache.

```
POST /api/ai/cache/clear
```

**Response:**
```json
{
  "success": true,
  "cleared": 5,
  "cache_size": 0
}
```

---

## 📦 Datenstrukturen

### Content Element Types

Alle Content-Elemente folgen diesem Schema:

```typescript
// TypeScript Interface Definitionen

interface EducationalContent {
  title: string;
  subtitle: string;
  chapters: Chapter[];
}

interface Chapter {
  type: "chapter";
  number: string;
  title: string;
  icon: string;
  sections: (Section | ContentElement)[];
}

interface Section {
  type: "section";
  title: string;
  icon: string;
  content: ContentElement[];
}

// Content Element Union Type
type ContentElement = 
  | Markdown 
  | Metric 
  | MetricRow 
  | Formula 
  | Plot 
  | Table 
  | Columns 
  | Expander 
  | InfoBox 
  | WarningBox 
  | SuccessBox 
  | CodeBlock 
  | Divider;

interface Markdown {
  type: "markdown";
  text: string;  // Markdown-formatierter Text
}

interface Metric {
  type: "metric";
  label: string;
  value: string;
  help_text?: string;
  delta?: string;
}

interface MetricRow {
  type: "metric_row";
  metrics: Metric[];
}

interface Formula {
  type: "formula";
  latex: string;   // LaTeX-Formel
  inline?: boolean;
}

interface Plot {
  type: "plot";
  plot_key: string;  // Key in plots-Object
  title?: string;
  description?: string;
  height?: number;
}

interface Table {
  type: "table";
  headers: string[];
  rows: string[][];
  caption?: string;
}

interface Columns {
  type: "columns";
  columns: ContentElement[][];
  widths?: number[];  // Relative Breiten
}

interface Expander {
  type: "expander";
  title: string;
  content: ContentElement[];
  expanded?: boolean;
}

interface InfoBox {
  type: "info_box";
  content: string;
}

interface WarningBox {
  type: "warning_box";
  content: string;
}

interface SuccessBox {
  type: "success_box";
  content: string;
}

interface CodeBlock {
  type: "code_block";
  code: string;
  language?: string;
}

interface Divider {
  type: "divider";
}
```

---

### Plotly Figure Format

Alle Plots werden im Plotly JSON-Format zurückgegeben:

```json
{
  "data": [
    {
      "type": "scatter",
      "x": [1, 2, 3, 4, 5],
      "y": [1.2, 2.1, 2.8, 3.9, 5.1],
      "mode": "markers",
      "name": "Datenpunkte",
      "marker": {
        "size": 10,
        "color": "#3498db"
      }
    },
    {
      "type": "scatter",
      "x": [1, 5],
      "y": [1.0, 5.0],
      "mode": "lines",
      "name": "Regression",
      "line": {
        "color": "#e74c3c",
        "width": 2
      }
    }
  ],
  "layout": {
    "title": "Regression: Umsatz vs Verkaufsfläche",
    "xaxis": {"title": "Verkaufsfläche (100 qm)"},
    "yaxis": {"title": "Umsatz (Mio. €)"},
    "template": "plotly_white"
  }
}
```

**Rendering in JavaScript:**
```javascript
// Mit Plotly.js
const plotData = response.plots.scatter;
Plotly.newPlot('chart', plotData.data, plotData.layout);
```

---

## ⚠️ Fehlerbehandlung

### Error Response Format

Alle Fehler folgen diesem Format:

```json
{
  "success": false,
  "error": "Beschreibung des Fehlers"
}
```

### HTTP Status Codes

| Code | Bedeutung |
|------|-----------|
| `200` | Erfolg |
| `400` | Ungültige Anfrage |
| `404` | Ressource nicht gefunden |
| `429` | Rate Limit erreicht |
| `500` | Server-Fehler |

### Beispiel Fehler-Handling

```javascript
async function fetchRegression(params) {
  const response = await fetch('/api/regression/simple', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  
  const data = await response.json();
  
  if (!data.success) {
    console.error('API Error:', data.error);
    throw new Error(data.error);
  }
  
  return data.data;
}
```

---

## 🛠️ Integration Beispiele

### Next.js / React

```typescript
// lib/regression-api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface RegressionParams {
  dataset?: string;
  n?: number;
  noise?: number;
  seed?: number;
}

export async function runSimpleRegression(params: RegressionParams) {
  const response = await fetch(`${API_URL}/api/regression/simple`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  
  const data = await response.json();
  if (!data.success) throw new Error(data.error);
  return data.data;
}

export async function getEducationalContent(params: RegressionParams) {
  const response = await fetch(`${API_URL}/api/content/simple`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  
  const data = await response.json();
  if (!data.success) throw new Error(data.error);
  return data;
}

export async function getAIInterpretation(stats: Record<string, any>) {
  const response = await fetch(`${API_URL}/api/ai/interpret`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stats }),
  });
  
  return response.json();
}
```

```tsx
// components/RegressionChart.tsx
'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Props {
  dataset: string;
  n: number;
}

export function RegressionChart({ dataset, n }: Props) {
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    async function load() {
      const data = await runSimpleRegression({ dataset, n });
      setPlotData(data.plots.scatter);
      setLoading(false);
    }
    load();
  }, [dataset, n]);
  
  if (loading) return <div>Loading...</div>;
  
  return (
    <Plot
      data={plotData.data}
      layout={plotData.layout}
      style={{ width: '100%', height: '400px' }}
    />
  );
}
```

---

### Vue.js / Nuxt

```typescript
// composables/useRegression.ts
import { ref, computed } from 'vue';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function useRegression() {
  const loading = ref(false);
  const error = ref<string | null>(null);
  const result = ref<any>(null);
  const content = ref<any>(null);
  
  async function runAnalysis(params: {
    dataset?: string;
    n?: number;
    type?: 'simple' | 'multiple';
  }) {
    loading.value = true;
    error.value = null;
    
    try {
      const endpoint = params.type === 'multiple' 
        ? '/api/content/multiple' 
        : '/api/content/simple';
      
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error);
      }
      
      result.value = data.stats;
      content.value = data.content;
      
      return data;
    } catch (e) {
      error.value = e.message;
      throw e;
    } finally {
      loading.value = false;
    }
  }
  
  const r2 = computed(() => result.value?.model_fit?.r_squared ?? 0);
  const isSignificant = computed(() => {
    const p = result.value?.t_tests?.slope?.p_value ?? 1;
    return p < 0.05;
  });
  
  return {
    loading,
    error,
    result,
    content,
    r2,
    isSignificant,
    runAnalysis,
  };
}
```

```vue
<!-- components/RegressionDashboard.vue -->
<template>
  <div class="regression-dashboard">
    <div class="controls">
      <select v-model="dataset">
        <option v-for="ds in datasets" :value="ds.id">
          {{ ds.icon }} {{ ds.name }}
        </option>
      </select>
      
      <input type="range" v-model="n" min="20" max="200" step="10" />
      <span>n = {{ n }}</span>
      
      <button @click="analyze" :disabled="loading">
        {{ loading ? 'Lädt...' : 'Analysieren' }}
      </button>
    </div>
    
    <div v-if="error" class="error">{{ error }}</div>
    
    <div v-if="content" class="results">
      <h2>{{ content.title }}</h2>
      <p>R² = {{ r2.toFixed(4) }}</p>
      <span :class="isSignificant ? 'significant' : 'not-significant'">
        {{ isSignificant ? '✅ Signifikant' : '⚠️ Nicht signifikant' }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRegression } from '~/composables/useRegression';

const { loading, error, content, r2, isSignificant, runAnalysis } = useRegression();

const dataset = ref('electronics');
const n = ref(50);
const datasets = ref([]);

async function analyze() {
  await runAnalysis({ dataset: dataset.value, n: n.value });
}

onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/datasets');
  const data = await response.json();
  datasets.value = data.data.simple;
});
</script>
```

---

### Python Client

```python
# regression_client.py
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class RegressionResult:
    intercept: float
    slope: float
    r_squared: float
    p_value: float
    n: int
    
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> 'RegressionResult':
        stats = data['stats']
        return cls(
            intercept=stats['coefficients']['intercept'],
            slope=stats['coefficients']['slope'],
            r_squared=stats['model_fit']['r_squared'],
            p_value=stats['t_tests']['slope']['p_value'],
            n=stats['sample']['n']
        )

class RegressionAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health(self) -> Dict[str, Any]:
        return self._get("/api/health")
    
    def datasets(self) -> Dict[str, List[Dict]]:
        response = self._get("/api/datasets")
        return response['data']
    
    def simple_regression(
        self,
        dataset: str = "electronics",
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42
    ) -> RegressionResult:
        response = self._post("/api/regression/simple", {
            "dataset": dataset,
            "n": n,
            "noise": noise,
            "seed": seed
        })
        return RegressionResult.from_api(response['data'])
    
    def get_content(
        self,
        regression_type: str = "simple",
        **kwargs
    ) -> Dict[str, Any]:
        endpoint = f"/api/content/{regression_type}"
        return self._post(endpoint, kwargs)
    
    def interpret(
        self,
        stats: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        return self._post("/api/ai/interpret", {
            "stats": stats,
            "use_cache": use_cache
        })
    
    def _get(self, endpoint: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}{endpoint}")
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        result = response.json()
        if not result.get('success', True):
            raise Exception(result.get('error', 'Unknown error'))
        return result

# Verwendung
if __name__ == "__main__":
    api = RegressionAPI()
    
    # Health check
    print(api.health())
    
    # Datasets
    datasets = api.datasets()
    print(f"Available: {[d['name'] for d in datasets['simple']]}")
    
    # Regression
    result = api.simple_regression(n=100)
    print(f"R² = {result.r_squared:.4f}")
    print(f"β₁ = {result.slope:.4f}")
    print(f"Signifikant: {result.p_value < 0.05}")
```

---

### Mobile (React Native / Flutter)

```javascript
// React Native - services/api.js
const API_URL = 'http://your-server:8000';

export const regressionApi = {
  async getDatasets() {
    const response = await fetch(`${API_URL}/api/datasets`);
    const data = await response.json();
    return data.data;
  },
  
  async runRegression(type, params) {
    const response = await fetch(`${API_URL}/api/regression/${type}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return response.json();
  },
  
  async getInterpretation(stats) {
    const response = await fetch(`${API_URL}/api/ai/interpret`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stats }),
    });
    return response.json();
  },
};
```

```dart
// Flutter - lib/services/regression_api.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class RegressionAPI {
  final String baseUrl;
  
  RegressionAPI({this.baseUrl = 'http://localhost:8000'});
  
  Future<Map<String, dynamic>> runSimpleRegression({
    String dataset = 'electronics',
    int n = 50,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/regression/simple'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'dataset': dataset, 'n': n}),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to run regression');
    }
  }
  
  Future<Map<String, dynamic>> getInterpretation(
    Map<String, dynamic> stats,
  ) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/ai/interpret'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'stats': stats}),
    );
    
    return jsonDecode(response.body);
  }
}
```

---

## 💡 Best Practices

### 1. Caching implementieren

```javascript
// Cache für wiederholte Anfragen
const cache = new Map();

async function getCachedContent(params) {
  const key = JSON.stringify(params);
  
  if (cache.has(key)) {
    return cache.get(key);
  }
  
  const data = await fetchContent(params);
  cache.set(key, data);
  
  return data;
}
```

### 2. Fehler graceful handeln

```javascript
async function safeInterpret(stats) {
  try {
    const result = await api.interpret(stats);
    return result.interpretation.content;
  } catch (error) {
    // Fallback: Lokale Interpretation
    return generateLocalInterpretation(stats);
  }
}
```

### 3. Loading States

```jsx
function RegressionView() {
  const [state, setState] = useState('idle'); // idle, loading, success, error
  
  return (
    <>
      {state === 'loading' && <Spinner />}
      {state === 'error' && <ErrorMessage />}
      {state === 'success' && <Results />}
    </>
  );
}
```

### 4. Optimistic Updates

```javascript
// UI sofort aktualisieren, dann API-Ergebnis verwenden
function updateAnalysis(newParams) {
  // Sofort UI aktualisieren (optimistisch)
  setParams(newParams);
  setLoading(true);
  
  // API im Hintergrund aufrufen
  api.getContent(newParams)
    .then(setContent)
    .finally(() => setLoading(false));
}
```

### 5. Debouncing für Slider

```javascript
import { useDebouncedCallback } from 'use-debounce';

function DataPointSlider() {
  const [n, setN] = useState(50);
  
  const debouncedFetch = useDebouncedCallback(
    (value) => fetchData({ n: value }),
    300
  );
  
  return (
    <input
      type="range"
      value={n}
      onChange={(e) => {
        setN(e.target.value);
        debouncedFetch(e.target.value);
      }}
    />
  );
}
```

---

## 📞 Support

- **GitHub Issues:** [Repository Issues](https://github.com/your-repo/issues)
- **API Status:** `/api/health`
- **OpenAPI Spec:** `/api/openapi.json`

---

## 📝 Changelog

### v1.0.0 (2026-01-07)
- Initial API release
- Simple & Multiple Regression
- Educational Content API
- Perplexity AI Integration
- Full TypeScript support
