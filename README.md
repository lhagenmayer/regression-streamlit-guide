# 📊 Linear Regression Guide & ML Bridge

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Clean-Architecture-2ecc71?style=flat-square&logo=architecture&logoColor=white)](docs/ARCHITECTURE.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

An interactive, educational platform bridging the gap between **Statistical Analysis** and **Machine Learning**. Built with **Clean Architecture** principles to demonstrate enterprise-grade software design in Python.

## ✨ Key Features

### 📉 Statistical Analysis
- **Simple Linear Regression**: Interactive exploration of slope, intercept, and R².
- **Multiple Regression**: 3D visualization of regression planes (e.g., House Prices).
- **Diagnostics**: Residual plots, Q-Q plots, and heteroskedasticity checks.
- **Hypothesis Testing**: T-tests, F-statistics, and p-values explained.

### 🤖 Machine Learning Bridge (NEW)
- **Classification Algorithms**: Logistic Regression & K-Nearest Neighbors (KNN) built from scratch (NumPy-only).
- **3D Visualizations**:
    - **Loss Landscapes**: Visualize Gradient Descent on 3D error surfaces.
    - **Classification Surfaces**: 3D probability boundaries with colored error analysis.
    - **3D Confusion Matrices**: "Lego-style" visualization of model performance.
    - **3D ROC Curves**: Dynamic trade-off visualization (FPR vs TPR vs Threshold).
- **Educational Content**: Interactive chapters explaining the transition from OLS to Gradient Descent.
- **Data Explorer**: Inspect and download raw datasets (CSV) directly within the app.

### 🏗️ State-of-the-Art Architecture
- **Clean Architecture**: Strict separation of Domain, Application, and Infrastructure layers.
- **Pure Domain**: Business logic has ZERO external dependencies (no numpy/pandas in core).
- **Dependency Injection**: Decoupled components wired via a DI container.
- **Type Safety**: Strictly typed with Python 3.9+ type hints and Pydantic validation.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Pip / Venv

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lhagenmayer/linear-regression-guide.git
   cd linear-regression-guide
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Flask Web App (The Full Experience)**
```bash
python run.py --flask
# Opens http://localhost:5000
```

**Option 2: Streamlit Dashboard (Interactive Exploration)**
```bash
python run.py --streamlit
# Opens http://localhost:8501
```

**Option 3: REST API Server**
```bash
python run.py --api
# API available at http://localhost:8000
```

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Deep dive into the Clean Architecture design.
- **[API Reference](docs/API.md)**: REST API endpoints and data schemas.
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)**: How to integrate the frontend with the API.

## 🛠️ Technology Stack

- **Core**: Python 3.11+, NumPy, SciPy
- **Architecture**: Domain-Driven Design (DDD), CQRS Patterns
- **API**: Flask, Pydantic (Validation)
- **Visualization**: Plotly (Interactive 3D Plots)
- **Frontend**: Flask (Jinja2 + Tailwind) & Streamlit

## 🤝 Contributing

Contributions are welcome! Please ensure you follow the **Clean Architecture** rules defined in `docs/ARCHITECTURE.md`.

1. **Domain Layer**: No external imports.
2. **Infrastructure Layer**: Implement interfaces defined in Domain.
3. **Tests**: Add unit tests for new logic.

---

*Created for educational purposes by [Luca Hagenmayer].*
