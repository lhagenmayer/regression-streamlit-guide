"""
Simple Regression Educational Content - Framework Agnostic.

This module defines ALL educational content as DATA STRUCTURES.
NO UI imports, NO framework dependencies.

The content is then interpreted by Renderers (Streamlit, Flask, etc.)
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .structure import (
    ContentElement, Chapter, Section, EducationalContent,
    Markdown, Metric, MetricRow, Formula, Plot, Table,
    Columns, Expander, InfoBox, WarningBox, SuccessBox,
    CodeBlock, Divider
)
from .builder import ContentBuilder


class SimpleRegressionContent(ContentBuilder):
    """
    Builds complete educational content for simple linear regression.
    
    All content is returned as DATA - no UI code here.
    """
    
    def build(self) -> EducationalContent:
        """Build all chapters of simple regression content."""
        return EducationalContent(
            title="📊 Einfache Lineare Regression",
            subtitle="Statistisches Lernen mit interaktiven Visualisierungen",
            chapters=[
                self._chapter_1_introduction(),
                self._chapter_1_5_multivariate_distributions(),
                self._chapter_2_regression_model(),
                self._chapter_2_5_covariance_correlation(),
                self._chapter_3_ols_estimation(),
                self._chapter_3_1_model_anatomy(),
                self._chapter_4_model_validation(),
                self._chapter_5_significance(),
                self._chapter_5_5_anova(),
                self._chapter_5_6_heteroskedasticity(),
                self._chapter_6_conclusion(),
            ]
        )
    
    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    def _chapter_1_introduction(self) -> Chapter:
        """Chapter 1: Introduction to regression analysis."""
        s = self.stats
        
        return Chapter(
            number="1.0",
            title="Einleitung - Die Analyse von Zusammenhängen",
            icon="📖",
            sections=[
                # Context info box
                InfoBox(f"""
**Kontext:** {s.get('context_title', 'Regressionsanalyse')}

{s.get('context_description', 'Analyse des linearen Zusammenhangs zwischen zwei Variablen.')}

**Zentrale Fragestellung:** Gibt es einen linearen Zusammenhang zwischen **{s.get('x_label', 'X')}** und **{s.get('y_label', 'Y')}**?
Und wenn ja: Wie stark ist dieser Zusammenhang?
"""),
                
                # Learning objectives
                Markdown("""
### 🎯 Lernziele dieses Moduls

Nach Abschluss dieses Moduls werden Sie verstehen:

1. **Konzepte**: Mehrdimensionale Verteilungen, Kovarianz, Korrelation
2. **Methodik**: OLS-Schätzung, Residuenanalyse
3. **Modellgüte**: R², Standardfehler, Bestimmtheitsmass
4. **Inferenz**: t-Tests, F-Tests, Signifikanz
5. **Probleme**: Heteroskedastizität, robuste Standardfehler
"""),
                
                # Key metrics overview
                Markdown("### 📊 Übersicht der Ergebnisse"),
                MetricRow([
                    Metric("R²", self.fmt(s.get('r_squared', 0)), "Erklärte Varianz"),
                    Metric("β₀", self.fmt(s.get('intercept', 0)), "Y-Achsenabschnitt"),
                    Metric("β₁", self.fmt(s.get('slope', 0)), "Steigung"),
                    Metric("p-Wert", self.fmt(s.get('p_slope', 0)), "Signifikanz"),
                    Metric("n", str(s.get('n', 0)), "Stichprobengrösse"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 1.5: MULTIVARIATE DISTRIBUTIONS
    # =========================================================================
    def _chapter_1_5_multivariate_distributions(self) -> Chapter:
        """Chapter 1.5: Multivariate distributions."""
        return Chapter(
            number="1.5",
            title="Mehrdimensionale Verteilungen",
            icon="📖",
            sections=[
                Markdown("""
**Das Fundament für Zusammenhänge**

Bevor wir Zusammenhänge analysieren können, müssen wir verstehen wie zwei 
Zufallsvariablen X und Y **gemeinsam** verteilt sein können.
"""),
                
                # Joint distribution expander
                Expander("🎲 Gemeinsame Verteilung f(X,Y)", [
                    Markdown("""
Die **gemeinsame Dichtefunktion** f(x,y) beschreibt die Wahrscheinlichkeitsverteilung
zweier Zufallsvariablen X und Y.

**Wichtige Konzepte:**
"""),
                    Columns([
                        # Left column
                        [
                            Markdown("""
**Randverteilungen:**
- f_X(x) = ∫ f(x,y) dy
- f_Y(y) = ∫ f(x,y) dx

**Bedingte Verteilung:**
"""),
                            Formula(r"f(y|x) = \frac{f(x,y)}{f_X(x)}"),
                        ],
                        # Right column  
                        [
                            Markdown("**Für die bivariate Normalverteilung:**"),
                            Formula(r"f(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{z}{2(1-\rho^2)}\right)"),
                        ],
                    ]),
                    Markdown("### 🎛️ Interaktive Bivariate Normalverteilung"),
                    # This will be rendered as an interactive element
                    Plot("bivariate_normal_3d", "Bivariate Normalverteilung", 
                         "Slider für Korrelation ρ", height=500),
                ], expanded=True),
                
                # Independence expander
                Expander("🔗 Stochastische Unabhängigkeit", [
                    Markdown("X und Y sind **stochastisch unabhängig** wenn:"),
                    Formula(r"f(x,y) = f_X(x) \cdot f_Y(y)"),
                    Formula(r"\text{oder äquivalent: } \rho_{XY} = 0 \text{ (für Normalverteilungen)}"),
                    WarningBox("""
⚠️ **Wichtig:** Unabhängigkeit impliziert keine Korrelation, aber keine Korrelation
impliziert **nicht** unbedingt Unabhängigkeit (es sei denn bei Normalverteilung)!
"""),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 2: REGRESSION MODEL
    # =========================================================================
    def _chapter_2_regression_model(self) -> Chapter:
        """Chapter 2: The simple linear regression model."""
        s = self.stats
        
        return Chapter(
            number="2.0",
            title="Das Fundament - Das einfache lineare Regressionsmodell",
            icon="📖",
            sections=[
                Columns([
                    # Left column - Model explanation
                    [
                        Markdown("""
Das **einfache lineare Regressionsmodell** beschreibt den Zusammenhang 
zwischen einer unabhängigen Variable X und einer abhängigen Variable Y:
"""),
                        Formula(r"Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i"),
                        Table(
                            headers=["Symbol", "Bedeutung", "Eigenschaften"],
                            rows=[
                                ["Yᵢ", "Abhängige Variable", "Beobachtete Werte"],
                                ["Xᵢ", "Unabhängige Variable", "Erklärende Variable"],
                                ["β₀", "Intercept", "Y-Achsenabschnitt"],
                                ["β₁", "Steigung", "Effekt von X auf Y"],
                                ["εᵢ", "Störterm", "E(ε)=0, Var(ε)=σ²"],
                            ]
                        ),
                    ],
                    # Right column - Context
                    [
                        InfoBox(f"""
### 💡 Praxisbeispiel: {s.get('context_title', 'Datenanalyse')}

**Unabhängige Variable (X):** {s.get('x_label', 'X')}  
**Abhängige Variable (Y):** {s.get('y_label', 'Y')}

**Erwartung:** {s.get('context_description', 'Linearer Zusammenhang')}
"""),
                    ],
                ], widths=[1.5, 1.0]),
                
                # Raw data visualization
                Markdown("### 📊 Die Rohdaten visualisieren"),
                Columns([
                    [
                        Markdown(f"""
**Was zeigt dieser Plot?**

Ein Streudiagramm (Scatter Plot) der {s.get('n', 0)} Beobachtungen. Jeder Punkt repräsentiert 
eine Messung mit Werten für {s.get('x_label', 'X')} (x-Achse) und {s.get('y_label', 'Y')} (y-Achse).

**Worauf achten?**
- Gibt es einen **Trend**? (aufsteigend/absteigend)
- Wie **eng** liegen die Punkte beieinander?
- Gibt es **Ausreisser**?
"""),
                        Plot("raw_scatter", "Streudiagramm der Rohdaten"),
                    ],
                    [
                        Markdown("### 📊 Deskriptive Statistik"),
                        Table(
                            headers=["Statistik", s.get('x_label', 'X'), s.get('y_label', 'Y')],
                            rows=[
                                ["Mittelwert", self.fmt(s.get('x_mean', 0), 2), self.fmt(s.get('y_mean', 0), 2)],
                                ["Std.Abw.", self.fmt(s.get('x_std', 0), 2), self.fmt(s.get('y_std', 0), 2)],
                                ["Min", self.fmt(s.get('x_min', 0), 2), self.fmt(s.get('y_min', 0), 2)],
                                ["Max", self.fmt(s.get('x_max', 0), 2), self.fmt(s.get('y_max', 0), 2)],
                            ]
                        ),
                        Metric("Korrelation r", self.fmt(s.get('correlation', 0))),
                    ],
                ], widths=[2.0, 1.0]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 2.5: COVARIANCE & CORRELATION
    # =========================================================================
    def _chapter_2_5_covariance_correlation(self) -> Chapter:
        """Chapter 2.5: Covariance and correlation."""
        s = self.stats
        corr = s.get('correlation', 0)
        cov = s.get('covariance', 0)
        
        return Chapter(
            number="2.5",
            title="Kovarianz & Korrelation - Die Bausteine der Regression",
            icon="📖",
            sections=[
                # Covariance section
                Expander("📐 Die Kovarianz", [
                    Markdown("Die **Kovarianz** misst den linearen Zusammenhang zwischen zwei Variablen."),
                    Columns([
                        [
                            Formula(r"\text{Cov}(X,Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})"),
                            Metric("Cov(X,Y)", self.fmt(cov)),
                            Markdown("""
**Interpretation:**
- Cov > 0: Positive Beziehung
- Cov < 0: Negative Beziehung  
- Cov = 0: Keine lineare Beziehung
"""),
                        ],
                        [
                            Plot("covariance_3d", "3D Kovarianz-Visualisierung", height=400),
                        ],
                    ]),
                ], expanded=True),
                
                # Correlation section
                Expander("📊 Der Korrelationskoeffizient (Pearson)", [
                    Markdown("Der **Korrelationskoeffizient** ist die standardisierte Kovarianz:"),
                    Formula(r"r = \frac{\text{Cov}(X,Y)}{s_X \cdot s_Y} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}"),
                    Columns([
                        [
                            Metric("Korrelation r", self.fmt(corr)),
                            Metric("r²", self.fmt(corr**2 if corr else 0)),
                            self._interpret_correlation_box(corr),
                        ],
                        [
                            Plot("correlation_examples", "6-Panel Korrelations-Beispiele", height=400),
                        ],
                    ]),
                ]),
                
                # Significance test
                Expander("🔬 Signifikanztest für die Korrelation", [
                    Markdown("""
**Hypothesen:**
- H₀: ρ = 0 (keine Korrelation in der Population)
- H₁: ρ ≠ 0 (es gibt eine Korrelation)

**Teststatistik:**
"""),
                    Formula(r"t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}"),
                    MetricRow([
                        Metric("t-Wert", self.fmt(s.get('t_correlation', 0), 3)),
                        Metric("p-Wert", self.fmt(s.get('p_correlation', 0))),
                    ]),
                    self._interpret_p_value_box(s.get('p_correlation', 1), "Korrelation"),
                ]),
                
                # Spearman correlation
                Expander("📊 Bonus: Spearman Rangkorrelation", [
                    Markdown("""
Die **Spearman Rangkorrelation** ist robust gegen Ausreisser und misst
monotone (nicht nur lineare) Zusammenhänge.
"""),
                    Formula(r"r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}"),
                    MetricRow([
                        Metric("Spearman ρ", self.fmt(s.get('spearman_r', 0))),
                        Metric("p-Wert", self.fmt(s.get('spearman_p', 0))),
                    ]),
                    Table(
                        headers=["Methode", "Korrelation"],
                        rows=[
                            ["Pearson", self.fmt(corr)],
                            ["Spearman", self.fmt(s.get('spearman_r', 0))],
                            ["Differenz", self.fmt(abs(corr - s.get('spearman_r', 0)) if corr and s.get('spearman_r') else 0)],
                        ]
                    ),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 3: OLS ESTIMATION
    # =========================================================================
    def _chapter_3_ols_estimation(self) -> Chapter:
        """Chapter 3: OLS Estimation method."""
        s = self.stats
        slope = s.get('slope', 0)
        intercept = s.get('intercept', 0)
        sign = "+" if slope >= 0 else ""
        
        return Chapter(
            number="3.0",
            title="Die Methode - Schätzung mittels OLS",
            icon="📖",
            sections=[
                Markdown("""
**Ordinary Least Squares (OLS)** minimiert die Summe der quadrierten Residuen:
"""),
                Formula(r"\min_{b_0, b_1} \sum_{i=1}^{n}(y_i - b_0 - b_1 x_i)^2 = \min SSE"),
                
                Columns([
                    [
                        Markdown("### 📊 OLS Visualisierung"),
                        Markdown("""
Die **roten Linien** zeigen die Residuen - die vertikalen Abstände 
zwischen den Datenpunkten und der Regressionsgerade.

**OLS minimiert die Summe der QUADRATE dieser Abstände.**
"""),
                        Plot("ols_regression", "Regression mit Residuen", height=400),
                    ],
                    [
                        Markdown("### 📐 Die OLS-Schätzer"),
                        Formula(r"b_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}"),
                        Formula(r"b_0 = \bar{y} - b_1\bar{x}"),
                        Table(
                            headers=["Schätzer", "Formel", "Wert"],
                            rows=[
                                ["b₁", "Cov(X,Y)/Var(X)", self.fmt(slope)],
                                ["b₀", "ȳ - b₁x̄", self.fmt(intercept)],
                            ]
                        ),
                        SuccessBox(f"**ŷ = {self.fmt(intercept)} {sign} {self.fmt(slope)} · x**"),
                    ],
                ], widths=[2.0, 1.0]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 3.1: MODEL ANATOMY
    # =========================================================================
    def _chapter_3_1_model_anatomy(self) -> Chapter:
        """Chapter 3.1: Model anatomy and uncertainty."""
        s = self.stats
        
        return Chapter(
            number="3.1",
            title="Das Regressionsmodell im Detail - Anatomie & Unsicherheit",
            icon="📖",
            sections=[
                Expander("🔍 Die Anatomie des Modells", [
                    Markdown("Jede Beobachtung lässt sich zerlegen in:"),
                    Formula(r"y_i = \underbrace{\hat{y}_i}_{\text{Fitted}} + \underbrace{e_i}_{\text{Residuum}}"),
                    Formula(r"y_i = \underbrace{(\bar{y})}_{\text{Mittelwert}} + \underbrace{(\hat{y}_i - \bar{y})}_{\text{Erklärt}} + \underbrace{(y_i - \hat{y}_i)}_{\text{Unerklärt}}"),
                    Plot("decomposition", "Zerlegung einer Beobachtung", height=400),
                ], expanded=True),
                
                Expander("📏 3D Konfidenz-Trichter", [
                    Markdown("""
Die Unsicherheit unserer Schätzung wird visualisiert durch:
- **Konfidenzband**: Unsicherheit der LINIE
- **Prognoseband**: Unsicherheit einzelner PUNKTE
"""),
                    Plot("confidence_funnel_3d", "3D Konfidenz-Trichter", height=500),
                ]),
                
                Expander("📖 Interpretation der Ergebnisse", [
                    self._build_interpretation_section(s),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 4: MODEL VALIDATION
    # =========================================================================
    def _chapter_4_model_validation(self) -> Chapter:
        """Chapter 4: Model validation and goodness of fit."""
        s = self.stats
        se_regression = np.sqrt(s.get('mse', 0)) if s.get('mse') else 0
        
        return Chapter(
            number="4.0",
            title="Die Güteprüfung - Validierung des Regressionsmodells",
            icon="📖",
            sections=[
                # Standard error of regression
                Expander("📐 4.1 Standardfehler der Regression (sₑ)", [
                    Markdown("""
Der **Standardfehler der Regression** (auch: Root Mean Square Error) misst
die durchschnittliche Abweichung der Beobachtungen von der Regressionsgerade.
"""),
                    Formula(r"s_e = \sqrt{\frac{SSE}{n-2}} = \sqrt{\frac{\sum e_i^2}{n-2}} = \sqrt{MSE}"),
                    Columns([
                        [
                            Metric("sₑ (RMSE)", self.fmt(se_regression)),
                            Metric("MSE", self.fmt(s.get('mse', 0))),
                            Metric("SSE", self.fmt(s.get('sse', 0))),
                        ],
                        [
                            InfoBox(f"""
**Interpretation:**

Die typische Abweichung vom vorhergesagten Wert 
beträgt etwa **±{self.fmt(se_regression, 2)}** {s.get('y_unit', 'Einheiten')}.

Ca. 68% der Beobachtungen liegen innerhalb von ±sₑ.
Ca. 95% liegen innerhalb von ±2·sₑ = ±{self.fmt(2*se_regression, 2)}.
"""),
                        ],
                    ]),
                ], expanded=True),
                
                # Standard error of coefficients
                Expander("📐 4.1b Standardfehler der Koeffizienten", [
                    Markdown("Die Standardfehler der Koeffizienten messen die **Unsicherheit** unserer Schätzer."),
                    Formula(r"SE(b_0) = s_e \sqrt{\frac{1}{n} + \frac{\bar{x}^2}{\sum(x_i-\bar{x})^2}}"),
                    Formula(r"SE(b_1) = \frac{s_e}{\sqrt{\sum(x_i-\bar{x})^2}}"),
                    Columns([
                        [
                            Metric("SE(b₀)", self.fmt(s.get('se_intercept', 0))),
                            Metric("SE(b₁)", self.fmt(s.get('se_slope', 0))),
                        ],
                        [
                            self._build_confidence_intervals_table(s),
                        ],
                    ]),
                    Plot("se_visualization", "Konfidenzband-Visualisierung", height=350),
                ]),
                
                # R-squared
                Expander("📊 4.2 Bestimmtheitsmass (R²)", [
                    Markdown("""
Das **Bestimmtheitsmass R²** gibt an, welcher Anteil der Varianz in Y
durch das Modell erklärt wird.
"""),
                    Formula(r"R^2 = 1 - \frac{SSE}{SST} = \frac{SSR}{SST}"),
                    Columns([
                        [
                            Plot("variance_decomposition", f"Varianzzerlegung: R² = {self.fmt(s.get('r_squared', 0))}", height=350),
                        ],
                        [
                            Markdown("### 📊 Interpretation"),
                            Metric("R²", self.fmt(s.get('r_squared', 0))),
                            Metric("R² adj.", self.fmt(s.get('r_squared_adj', 0))),
                            Markdown(f"""
**{s.get('r_squared', 0) * 100:.1f}%** der Varianz in {s.get('y_label', 'Y')} 
wird durch {s.get('x_label', 'X')} erklärt.

**{(1-s.get('r_squared', 0)) * 100:.1f}%** bleiben unerklärt.
"""),
                            self._interpret_r2_box(s.get('r_squared', 0)),
                        ],
                    ], widths=[1.5, 1.0]),
                ], expanded=True),
            ]
        )
    
    # =========================================================================
    # CHAPTER 5: SIGNIFICANCE
    # =========================================================================
    def _chapter_5_significance(self) -> Chapter:
        """Chapter 5: Statistical significance and inference."""
        s = self.stats
        
        return Chapter(
            number="5.0",
            title="Die Signifikanz - Statistische Inferenz und Hypothesentests",
            icon="📖",
            sections=[
                # Gauss-Markov assumptions
                Expander("📋 Voraussetzungen: Die Gauss-Markov Annahmen", [
                    Markdown("Damit OLS **BLUE** ist (Best Linear Unbiased Estimator), müssen folgende Annahmen erfüllt sein:"),
                    Columns([
                        [
                            Markdown("""
**1. Linearität**
- E(Y|X) = β₀ + β₁X
- Der Zusammenhang ist linear in den Parametern

**2. Strikt exogene Regressoren**
- E(ε|X) = 0
- Die Fehler sind unkorreliert mit X

**3. Keine perfekte Multikollinearität**
- Var(X) > 0
- X hat Variation
"""),
                        ],
                        [
                            Markdown("""
**4. Homoskedastizität**
- Var(ε|X) = σ² (konstant)
- Die Varianz der Fehler ist konstant

**5. Keine Autokorrelation**
- Cov(εᵢ, εⱼ) = 0 für i ≠ j
- Fehler sind unabhängig

**6. Normalverteilung** (für Inferenz)
- ε ~ N(0, σ²)
"""),
                        ],
                    ]),
                    Plot("assumptions_4panel", "Diagnose-Plots: Gauss-Markov Annahmen", height=600),
                ], expanded=True),
                
                # Interactive assumption violation demo
                Expander("🎛️ Interaktiv: Was passiert bei Annahmenverletzung?", [
                    Plot("assumption_violation_demo", "Interaktive Annahmenverletzung", height=350),
                ]),
                
                # t-Test
                Expander("🔬 Der t-Test für die Koeffizienten", [
                    Markdown(f"""
### t-Test für die Steigung β₁

**H₀:** β₁ = 0 (kein Effekt)  
**H₁:** β₁ ≠ 0 (es gibt einen Effekt)

**Teststatistik:**
"""),
                    Formula(rf"t = \frac{{b_1 - 0}}{{SE(b_1)}} = \frac{{{self.fmt(s.get('slope', 0))}}}{{{self.fmt(s.get('se_slope', 0))}}} = {self.fmt(s.get('t_slope', 0), 3)}"),
                    Columns([
                        [
                            Plot("t_test_plot", f"t-Test (df={s.get('df', 0)})", height=350),
                        ],
                        [
                            Markdown("### 📋 Koeffizienten-Tabelle"),
                            self._build_coefficient_table(s),
                            Markdown("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1"),
                        ],
                    ], widths=[1.5, 1.0]),
                ], expanded=True),
                
                # F-Test
                Expander("⚖️ Der F-Test", [
                    Markdown("""
Der **F-Test** testet die Gesamtsignifikanz des Modells.

**H₀:** β₁ = 0 (Modell erklärt nichts)  
**H₁:** β₁ ≠ 0 (Modell erklärt etwas)
"""),
                    Formula(r"F = \frac{MSR}{MSE} = \frac{SSR/k}{SSE/(n-k-1)}"),
                    Columns([
                        [
                            Metric("F-Statistik", self.fmt(s.get('f_statistic', 0), 3)),
                            Metric("p-Wert", self.fmt(s.get('p_f', 0))),
                            Metric("df1, df2", f"1, {s.get('df', 0)}"),
                        ],
                        [
                            InfoBox(f"""
**Hinweis:** Bei einfacher Regression gilt:

F = t² = {self.fmt(s.get('t_slope', 0)**2 if s.get('t_slope') else 0, 3)}

Der F-Test und t-Test führen zum selben Ergebnis!
"""),
                            self._interpret_p_value_box(s.get('p_f', 1), "Modell"),
                        ],
                    ]),
                ]),
                
                # ANOVA Table
                Expander("📊 Die vollständige ANOVA-Tabelle", [
                    self._build_anova_table(s),
                ]),
                
                # R-Style Output
                Expander("💻 Der komplette R-Style Output", [
                    CodeBlock(self._build_r_output(s), language="text"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 5.5: ANOVA
    # =========================================================================
    def _chapter_5_5_anova(self) -> Chapter:
        """Chapter 5.5: ANOVA for group comparisons."""
        return Chapter(
            number="5.5",
            title="ANOVA für Gruppenvergleiche",
            icon="📖",
            sections=[
                Markdown("""
Die **Analysis of Variance (ANOVA)** erweitert den t-Test auf mehr als zwei Gruppen.

**Frage:** Unterscheiden sich die Mittelwerte von k Gruppen?
"""),
                Expander("🔬 Interaktives ANOVA-Beispiel", [
                    Plot("anova_interactive", "Interaktives ANOVA-Beispiel", height=400),
                ], expanded=True),
                
                Expander("📊 3D Verteilungslandschaft", [
                    Plot("anova_3d_landscape", "3D ANOVA Verteilungslandschaft", height=450),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 5.6: HETEROSKEDASTICITY
    # =========================================================================
    def _chapter_5_6_heteroskedasticity(self) -> Chapter:
        """Chapter 5.6: Heteroskedasticity problem."""
        s = self.stats
        
        return Chapter(
            number="5.6",
            title="Das grosse Problem - Heteroskedastizität",
            icon="📖",
            sections=[
                WarningBox("""
**Heteroskedastizität** liegt vor, wenn die Varianz der Fehler nicht konstant ist.
Dies verletzt die Gauss-Markov Annahme 4 und führt zu:
- Ineffizienten Schätzern
- Falschen Standardfehlern
- Invaliden t- und F-Tests
"""),
                
                Expander("📊 Trichter-Effekt visualisieren", [
                    Plot("heteroskedasticity_demo", "Heteroskedastizität Demo", height=350),
                ], expanded=True),
                
                Expander("🔧 Robuste Standardfehler (HC3)", [
                    Markdown("""
### Robuste Standardfehler (HC3)

**HC3** (Heteroskedasticity-Consistent) Standardfehler sind robust gegen Heteroskedastizität:
"""),
                    Formula(r"SE_{HC3}(b_1) = \sqrt{\frac{\sum e_i^2 / (1-h_{ii})^2 \cdot (x_i - \bar{x})^2}{(\sum(x_i-\bar{x})^2)^2}}"),
                    Columns([
                        [
                            Markdown("### Normale SE (OLS)"),
                            Metric("SE(b₁)", self.fmt(s.get('se_slope', 0))),
                            Metric("t-Wert", self.fmt(s.get('t_slope', 0), 3)),
                            Metric("p-Wert", self.fmt(s.get('p_slope', 0))),
                        ],
                        [
                            Markdown("### Robuste SE (HC3)"),
                            Metric("SE(b₁)", self.fmt(s.get('se_slope_hc3', 0))),
                            Metric("t-Wert", self.fmt(s.get('t_slope_hc3', 0), 3)),
                            Metric("p-Wert", self.fmt(s.get('p_slope_hc3', 0))),
                        ],
                    ]),
                    self._interpret_robust_se(s),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 6: CONCLUSION
    # =========================================================================
    def _chapter_6_conclusion(self) -> Chapter:
        """Chapter 6: Conclusion and outlook."""
        s = self.stats
        sig_text = "✅ signifikant" if s.get('p_slope', 1) < 0.05 else "⚠️ nicht signifikant"
        
        return Chapter(
            number="6.0",
            title="Fazit und Ausblick",
            icon="📖",
            sections=[
                Columns([
                    [
                        SuccessBox(f"""
### 📝 Zusammenfassung der Analyse

**Regressionsgleichung:**  
{s.get('y_label', 'Y')} = {self.fmt(s.get('intercept', 0))} + {self.fmt(s.get('slope', 0))} × {s.get('x_label', 'X')}

**Interpretation:**
- R² = {self.fmt(s.get('r_squared', 0))} → {s.get('r_squared', 0)*100:.1f}% der Varianz erklärt
- Pro Einheit {s.get('x_label', 'X')} ändert sich {s.get('y_label', 'Y')} um {self.fmt(s.get('slope', 0))}
- Die Steigung ist {sig_text} (p = {self.fmt(s.get('p_slope', 0))})

**Stichprobe:** n = {s.get('n', 0)} Beobachtungen
"""),
                        Markdown("""
### ✅ Checkliste für gute Regression

- [ ] Linearität überprüft (Residuen vs. Fitted)
- [ ] Normalität der Residuen (Q-Q Plot)
- [ ] Homoskedastizität (keine Trichterform)
- [ ] Keine Ausreisser/Einflussreiche Punkte
- [ ] Unabhängigkeit der Beobachtungen
- [ ] R² interpretiert
- [ ] Koeffizienten interpretiert
- [ ] Signifikanz geprüft
"""),
                    ],
                    [
                        Markdown("### 📊 Daten"),
                        Expander("Datentabelle anzeigen", [
                            Plot("data_table", "Datentabelle"),
                        ]),
                    ],
                ], widths=[2.0, 1.0]),
                
                Expander("🌊 Bonusgrafik: Die bedingte Verteilung f(y|x)", [
                    Markdown("""
**Die bedingte Verteilung f(y|x)** zeigt die Verteilung von Y für jeden Wert von X.

Bei homoskedastischen Fehlern haben alle bedingten Verteilungen dieselbe Varianz.
"""),
                    Plot("conditional_distribution_3d", "3D Bedingte Verteilung f(y|x)", height=500),
                ]),
            ]
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _get_stars(self, p: float) -> str:
        """Get significance stars."""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        if p < 0.1: return "."
        return ""
    
    def _interpret_correlation_box(self, r: float) -> ContentElement:
        """Create interpretation box for correlation."""
        abs_r = abs(r) if r else 0
        if abs_r > 0.7:
            return SuccessBox("✅ Starke Korrelation")
        elif abs_r > 0.4:
            return InfoBox("ℹ️ Mittlere Korrelation")
        else:
            return WarningBox("⚠️ Schwache Korrelation")
    
    def _interpret_r2_box(self, r2: float) -> ContentElement:
        """Create interpretation box for R²."""
        if r2 > 0.8:
            return SuccessBox("✅ Sehr gute Anpassung")
        elif r2 > 0.5:
            return InfoBox("ℹ️ Akzeptable Anpassung")
        else:
            return WarningBox("⚠️ Schwache Anpassung")
    
    def _interpret_p_value_box(self, p: float, test_name: str) -> ContentElement:
        """Create interpretation box for p-value."""
        if p < 0.05:
            return SuccessBox(f"✅ {test_name} signifikant bei α=0.05")
        else:
            return WarningBox(f"⚠️ {test_name} nicht signifikant")
    
    def _interpret_robust_se(self, s: Dict[str, Any]) -> ContentElement:
        """Create interpretation for robust vs normal SE comparison."""
        se_normal = s.get('se_slope', 1)
        se_hc3 = s.get('se_slope_hc3', 1)
        
        if se_normal and se_hc3:
            diff_pct = (se_hc3 - se_normal) / se_normal * 100
            if abs(diff_pct) > 20:
                return WarningBox(f"⚠️ Die robusten SE weichen um {diff_pct:.1f}% ab - mögliche Heteroskedastizität!")
            else:
                return SuccessBox(f"✅ Die SE sind ähnlich (Differenz: {diff_pct:.1f}%) - keine starke Heteroskedastizität")
        return InfoBox("Robuste SE konnten nicht berechnet werden.")
    
    def _build_interpretation_section(self, s: Dict[str, Any]) -> Markdown:
        """Build the interpretation section for model results."""
        slope = s.get('slope', 0)
        intercept = s.get('intercept', 0)
        x_mean = s.get('x_mean', 0)
        x_std = s.get('x_std', 1)
        
        direction = "Zunahme" if slope > 0 else "Abnahme"
        
        return Markdown(f"""
### 📊 Vollständige Interpretation

**Das Modell:**

{s.get('y_label', 'Y')} = {self.fmt(intercept)} + {self.fmt(slope)} × {s.get('x_label', 'X')}

**Interpretation des Intercepts (β₀ = {self.fmt(intercept)}):**

Wenn {s.get('x_label', 'X')} = 0, dann erwarten wir {s.get('y_label', 'Y')} = {self.fmt(intercept, 2)} {s.get('y_unit', '')}.
⚠️ Diese Interpretation ist nur sinnvoll wenn X=0 im relevanten Bereich liegt!

**Interpretation der Steigung (β₁ = {self.fmt(slope)}):**

Für jede Einheit Zunahme in {s.get('x_label', 'X')} erwarten wir:
- Eine {direction} von **{self.fmt(abs(slope))}** {s.get('y_unit', '')} in {s.get('y_label', 'Y')}

**Praktisches Beispiel:**
- Bei {s.get('x_label', 'X')} = {self.fmt(x_mean, 2)}: ŷ = {self.fmt(intercept + slope * x_mean, 2)}
- Bei {s.get('x_label', 'X')} = {self.fmt(x_mean + x_std, 2)}: ŷ = {self.fmt(intercept + slope * (x_mean + x_std), 2)}
""")
    
    def _build_confidence_intervals_table(self, s: Dict[str, Any]) -> Table:
        """Build confidence intervals table."""
        intercept = s.get('intercept', 0)
        slope = s.get('slope', 0)
        se_intercept = s.get('se_intercept', 0)
        se_slope = s.get('se_slope', 0)
        
        return Table(
            headers=["Parameter", "Schätzwert", "95% KI"],
            rows=[
                ["β₀", self.fmt(intercept), f"[{self.fmt(intercept - 1.96*se_intercept)}, {self.fmt(intercept + 1.96*se_intercept)}]"],
                ["β₁", self.fmt(slope), f"[{self.fmt(slope - 1.96*se_slope)}, {self.fmt(slope + 1.96*se_slope)}]"],
            ],
            caption="95% Konfidenzintervalle"
        )
    
    def _build_coefficient_table(self, s: Dict[str, Any]) -> Table:
        """Build coefficient table."""
        return Table(
            headers=["Parameter", "Schätzwert", "Std.Error", "t-Wert", "p-Wert", "Signif."],
            rows=[
                ["β₀ (Intercept)", 
                 self.fmt(s.get('intercept', 0)), 
                 self.fmt(s.get('se_intercept', 0)),
                 self.fmt(s.get('t_intercept', 0), 3),
                 self.fmt(s.get('p_intercept', 0)),
                 self._get_stars(s.get('p_intercept', 1))],
                [f"β₁ ({s.get('x_label', 'X')})", 
                 self.fmt(s.get('slope', 0)), 
                 self.fmt(s.get('se_slope', 0)),
                 self.fmt(s.get('t_slope', 0), 3),
                 self.fmt(s.get('p_slope', 0)),
                 self._get_stars(s.get('p_slope', 1))],
            ]
        )
    
    def _build_anova_table(self, s: Dict[str, Any]) -> Table:
        """Build ANOVA table."""
        msr = s.get('ssr', 0) / 1 if s.get('ssr') else 0
        mse = s.get('sse', 0) / s.get('df', 1) if s.get('sse') and s.get('df') else 0
        f_stat = msr / mse if mse else 0
        
        return Table(
            headers=["Quelle", "SS", "df", "MS", "F", "p-Wert"],
            rows=[
                ["Regression", self.fmt(s.get('ssr', 0)), "1", self.fmt(msr), self.fmt(f_stat), self.fmt(s.get('p_f', 0))],
                ["Residuen", self.fmt(s.get('sse', 0)), str(s.get('df', 0)), self.fmt(mse), "", ""],
                ["Total", self.fmt(s.get('sst', 0)), str(s.get('n', 0) - 1), "", "", ""],
            ],
            caption="ANOVA-Tabelle"
        )
    
    def _build_r_output(self, s: Dict[str, Any]) -> str:
        """Build R-style output string."""
        residuals = s.get('residuals', [0, 0, 0, 0, 0])
        if not isinstance(residuals, (list, np.ndarray)) or len(residuals) < 5:
            residuals = [0, 0, 0, 0, 0]
        
        return f"""
Call:
lm(formula = {s.get('y_label', 'Y')} ~ {s.get('x_label', 'X')})

Residuals:
     Min       1Q   Median       3Q      Max 
{np.min(residuals):8.4f} {np.percentile(residuals, 25):8.4f} {np.median(residuals):8.4f} {np.percentile(residuals, 75):8.4f} {np.max(residuals):8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  {s.get('intercept', 0):9.4f}   {s.get('se_intercept', 0):9.4f}  {s.get('t_intercept', 0):7.3f}  {s.get('p_intercept', 0):8.4f} {self._get_stars(s.get('p_intercept', 1))}
{s.get('x_label', 'X'):12s} {s.get('slope', 0):9.4f}   {s.get('se_slope', 0):9.4f}  {s.get('t_slope', 0):7.3f}  {s.get('p_slope', 0):8.4f} {self._get_stars(s.get('p_slope', 1))}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {np.sqrt(s.get('mse', 0)):0.4f} on {s.get('df', 0)} degrees of freedom
Multiple R-squared:  {s.get('r_squared', 0):.4f},	Adjusted R-squared:  {s.get('r_squared_adj', 0):.4f}
F-statistic: {s.get('f_statistic', 0):.2f} on 1 and {s.get('df', 0)} DF,  p-value: {s.get('p_slope', 0):.4e}
"""
