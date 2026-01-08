"""
Multiple Regression Educational Content - Framework Agnostic.

This module defines ALL educational content for multiple regression as DATA STRUCTURES.
NO UI imports, NO framework dependencies.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

from .structure import (
    ContentElement, Chapter, Section, EducationalContent,
    Markdown, Metric, MetricRow, Formula, Plot, Table,
    Columns, Expander, InfoBox, WarningBox, SuccessBox,
    CodeBlock, Divider
)
from .builder import ContentBuilder


class MultipleRegressionContent(ContentBuilder):
    """
    Builds complete educational content for multiple linear regression.
    
    All content is returned as DATA - no UI code here.
    """
    
    def build(self) -> EducationalContent:
        """Build all chapters of multiple regression content."""
        return EducationalContent(
            title="📊 Multiple Linear Regression",
            subtitle="Advanced regression analysis with multiple predictors",
            chapters=[
                self._chapter_1_introduction(),
                self._chapter_2_model(),
                self._chapter_3_ols_matrix(),
                self._chapter_4_interpretation(),
                self._chapter_5_model_quality(),
                self._chapter_6_multicollinearity(),
                self._chapter_7_dummy_variables(),
                self._chapter_8_diagnostics(),
                self._chapter_9_prediction(),
            ]
        )
    
    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    def _chapter_1_introduction(self) -> Chapter:
        """Chapter 1: Introduction to multiple regression."""
        s = self.stats
        
        return Chapter(
            number="1.0",
            title="Introduction - Multiple Regression",
            icon="📖",
            sections=[
                InfoBox(f"""
**Kontext:** {s.get('context_title', 'Multiple Regressionsanalyse')}

{s.get('context_description', 'Analyse des Zusammenhangs zwischen einer abhängigen Variable und mehreren Prädiktoren.')}

**Zentrale Fragestellung:** Wie beeinflussen **{s.get('x1_label', 'X₁')}** und **{s.get('x2_label', 'X₂')}** 
gemeinsam die Variable **{s.get('y_label', 'Y')}**?
"""),
                
                Markdown("""
### 🎯 Lernziele dieses Moduls

Nach Abschluss werden Sie verstehen:

1. **Modell**: Multiple Regression mit k Prädiktoren
2. **Matrixform**: OLS in Matrixnotation
3. **Interpretation**: Partielle vs. totale Effekte
4. **Diagnostik**: Multikollinearität, VIF
5. **Dummy-Variablen**: Kategoriale Prädiktoren
6. **Prognose**: Vorhersagen mit Konfidenzintervallen
"""),
                
                Markdown("### 📊 Übersicht der Ergebnisse"),
                MetricRow([
                    Metric("R²", self.fmt(s.get('r_squared', 0)), "Erklärte Varianz"),
                    Metric("R² adj.", self.fmt(s.get('r_squared_adj', 0)), "Adjustiert"),
                    Metric("F", self.fmt(s.get('f_statistic', 0), 2), "F-Statistik"),
                    Metric("p(F)", self.fmt(s.get('p_f', 0)), "p-Wert"),
                    Metric("n", str(s.get('n', 0)), "Stichprobe"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 2: THE MODEL
    # =========================================================================
    def _chapter_2_model(self) -> Chapter:
        """Chapter 2: The multiple regression model."""
        s = self.stats
        
        return Chapter(
            number="2.0",
            title="The Multiple Regression Model",
            icon="📖",
            sections=[
                Markdown("""
Das **multiple lineare Regressionsmodell** erweitert die einfache Regression auf k Prädiktoren:
"""),
                Formula(r"Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + ... + \beta_k X_{ki} + \varepsilon_i"),
                
                Expander("📐 Das Modell im Detail", [
                    Table(
                        headers=["Symbol", "Bedeutung", "Interpretation"],
                        rows=[
                            ["Yᵢ", "Abhängige Variable", "Zu erklärende Variable"],
                            ["X₁ᵢ...Xₖᵢ", "Prädiktoren", "Erklärende Variablen"],
                            ["β₀", "Intercept", "Y wenn alle X = 0"],
                            ["β₁...βₖ", "Partielle Steigungen", "Effekt von Xⱼ bei konstantem Xₖ"],
                            ["εᵢ", "Störterm", "E(ε)=0, Var(ε)=σ²"],
                        ]
                    ),
                    InfoBox(f"""
**Unser Modell:**

{s.get('y_label', 'Y')} = β₀ + β₁·{s.get('x1_label', 'X₁')} + β₂·{s.get('x2_label', 'X₂')} + ε

Mit:
- β₀ = {self.fmt(s.get('intercept', 0))}
- β₁ = {self.fmt(s.get('beta1', 0))}
- β₂ = {self.fmt(s.get('beta2', 0))}
"""),
                ], expanded=True),
                
                Markdown("### 📊 3D Visualisierung der Regressionsebene"),
                Plot("regression_plane_3d", "3D Regressionsebene", height=500),
            ]
        )
    
    # =========================================================================
    # CHAPTER 3: OLS IN MATRIX FORM
    # =========================================================================
    def _chapter_3_ols_matrix(self) -> Chapter:
        """Chapter 3: OLS in matrix notation."""
        s = self.stats
        
        return Chapter(
            number="3.0",
            title="OLS in Matrix Form",
            icon="📖",
            sections=[
                Markdown("""
Multiple Regression wird elegant in **Matrixnotation** dargestellt:
"""),
                Formula(r"\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}"),
                
                Expander("📐 Die Matrizen", [
                    Columns([
                        [
                            Markdown("**Y-Vektor (n×1):**"),
                            Formula(r"\mathbf{Y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}"),
                        ],
                        [
                            Markdown("**X-Matrix (n×(k+1)):**"),
                            Formula(r"\mathbf{X} = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1k} \\ 1 & x_{21} & \cdots & x_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{nk} \end{pmatrix}"),
                        ],
                        [
                            Markdown("**β-Vektor ((k+1)×1):**"),
                            Formula(r"\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_k \end{pmatrix}"),
                        ],
                    ]),
                ], expanded=True),
                
                Expander("🔬 Der OLS-Schätzer", [
                    Markdown("Die **Normalgleichungen** liefern:"),
                    Formula(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}"),
                    
                    Markdown("**Varianz-Kovarianz-Matrix der Schätzer:**"),
                    Formula(r"\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}'\mathbf{X})^{-1}"),
                    
                    InfoBox("""
Die Diagonalelemente von (X'X)⁻¹ · σ² geben die **Varianzen** der einzelnen βⱼ.
Die Wurzeln sind die **Standardfehler** SE(βⱼ).
"""),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 4: INTERPRETATION
    # =========================================================================
    def _chapter_4_interpretation(self) -> Chapter:
        """Chapter 4: Coefficient interpretation."""
        s = self.stats
        
        beta1 = s.get('beta1', 0)
        beta2 = s.get('beta2', 0)
        
        return Chapter(
            number="4.0",
            title="Interpretation of Coefficients",
            icon="📖",
            sections=[
                WarningBox("""
**Wichtig:** In der multiplen Regression sind die βⱼ **partielle** Effekte!

βⱼ = Änderung in Y für eine Einheit Änderung in Xⱼ, **wenn alle anderen X konstant gehalten werden**.
"""),
                
                Expander("📊 Partielle vs. Totale Effekte", [
                    Columns([
                        [
                            Markdown(f"""
**Partieller Effekt β₁ = {self.fmt(beta1)}:**

Wenn {s.get('x1_label', 'X₁')} um 1 Einheit steigt 
und {s.get('x2_label', 'X₂')} **konstant** bleibt,
dann ändert sich {s.get('y_label', 'Y')} um {self.fmt(beta1)}.
"""),
                        ],
                        [
                            Markdown(f"""
**Partieller Effekt β₂ = {self.fmt(beta2)}:**

Wenn {s.get('x2_label', 'X₂')} um 1 Einheit steigt 
und {s.get('x1_label', 'X₁')} **konstant** bleibt,
dann ändert sich {s.get('y_label', 'Y')} um {self.fmt(beta2)}.
"""),
                        ],
                    ]),
                    Plot("partial_effects", "Partielle Effekte Visualisierung", height=400),
                ], expanded=True),
                
                Expander("📋 Koeffizienten-Tabelle", [
                    self._build_coefficient_table(s),
                    Markdown("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1"),
                ]),
                
                Expander("🔄 Standardisierte Koeffizienten (Beta)", [
                    Markdown("""
**Standardisierte Koeffizienten** ermöglichen den Vergleich der relativen Wichtigkeit:
"""),
                    Formula(r"\beta^*_j = \beta_j \cdot \frac{s_{X_j}}{s_Y}"),
                    
                    Metric("β*₁", self.fmt(s.get('beta1_std', 0)), "Standardisiert"),
                    Metric("β*₂", self.fmt(s.get('beta2_std', 0)), "Standardisiert"),
                    
                    InfoBox("""
Standardisierte Koeffizienten zeigen, um wie viele **Standardabweichungen** sich Y ändert,
wenn Xⱼ um eine Standardabweichung steigt.
"""),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 5: MODEL QUALITY
    # =========================================================================
    def _chapter_5_model_quality(self) -> Chapter:
        """Chapter 5: Model quality assessment."""
        s = self.stats
        
        return Chapter(
            number="5.0",
            title="Model Quality - R² and F-Test",
            icon="📖",
            sections=[
                Expander("📊 R² vs. Adjustiertes R²", [
                    Columns([
                        [
                            Markdown("**R² (Bestimmtheitsmass):**"),
                            Formula(r"R^2 = 1 - \frac{SSE}{SST}"),
                            Metric("R²", self.fmt(s.get('r_squared', 0))),
                            WarningBox("⚠️ R² steigt **immer** wenn Variablen hinzugefügt werden!"),
                        ],
                        [
                            Markdown("**Adjustiertes R²:**"),
                            Formula(r"R^2_{adj} = 1 - \frac{SSE/(n-k-1)}{SST/(n-1)}"),
                            Metric("R² adj.", self.fmt(s.get('r_squared_adj', 0))),
                            SuccessBox("✅ Bestraft für zusätzliche Variablen"),
                        ],
                    ]),
                    Markdown(f"""
**Interpretation:**
- **{s.get('r_squared', 0)*100:.1f}%** der Varianz in {s.get('y_label', 'Y')} wird erklärt
- Nach Adjustierung: **{s.get('r_squared_adj', 0)*100:.1f}%**
"""),
                ], expanded=True),
                
                Expander("⚖️ Der F-Test für Gesamtsignifikanz", [
                    Markdown("""
**H₀:** β₁ = β₂ = ... = βₖ = 0 (Modell erklärt nichts)
**H₁:** Mindestens ein βⱼ ≠ 0
"""),
                    Formula(r"F = \frac{(SST - SSE)/k}{SSE/(n-k-1)} = \frac{MSR}{MSE}"),
                    
                    MetricRow([
                        Metric("F", self.fmt(s.get('f_statistic', 0), 3)),
                        Metric("p-Wert", self.fmt(s.get('p_f', 0))),
                        Metric("df1, df2", f"{s.get('k', 2)}, {s.get('df', 0)}"),
                    ]),
                    
                    self._interpret_p_value_box(s.get('p_f', 1), "Modell"),
                ]),
                
                Plot("variance_decomposition", "Varianzzerlegung", height=350),
            ]
        )
    
    # =========================================================================
    # CHAPTER 6: MULTICOLLINEARITY
    # =========================================================================
    def _chapter_6_multicollinearity(self) -> Chapter:
        """Chapter 6: Multicollinearity diagnosis."""
        s = self.stats
        
        return Chapter(
            number="6.0",
            title="Multicollinearity",
            icon="📖",
            sections=[
                WarningBox("""
**Multikollinearität** liegt vor, wenn die Prädiktoren stark miteinander korrelieren.

**Konsequenzen:**
- Instabile Koeffizienten (große Standardfehler)
- Schwierige Interpretation
- Koeffizienten können "falsche" Vorzeichen haben
"""),
                
                Expander("📊 Korrelation der Prädiktoren", [
                    MetricRow([
                        Metric("Korr(X₁, X₂)", self.fmt(s.get('corr_x1_x2', 0))),
                    ]),
                    Plot("predictor_correlation", "Korrelation der Prädiktoren", height=350),
                    
                    self._interpret_correlation_box(s.get('corr_x1_x2', 0)),
                ], expanded=True),
                
                Expander("🔬 VIF - Variance Inflation Factor", [
                    Markdown("""
Der **VIF** quantifiziert wie stark ein Prädiktor durch die anderen erklärt wird:
"""),
                    Formula(r"VIF_j = \frac{1}{1 - R^2_j}"),
                    
                    Markdown("""
**Interpretation:**
- VIF = 1: Keine Multikollinearität
- VIF < 5: Akzeptabel
- VIF 5-10: Problematisch
- VIF > 10: Schwere Multikollinearität
"""),
                    
                    MetricRow([
                        Metric(f"VIF({s.get('x1_label', 'X₁')})", self.fmt(s.get('vif_x1', 1), 2)),
                        Metric(f"VIF({s.get('x2_label', 'X₂')})", self.fmt(s.get('vif_x2', 1), 2)),
                    ]),
                    
                    self._interpret_vif_box(max(s.get('vif_x1', 1), s.get('vif_x2', 1))),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7: DUMMY VARIABLES
    # =========================================================================
    def _chapter_7_dummy_variables(self) -> Chapter:
        """Chapter 7: Dummy variables for categorical predictors."""
        return Chapter(
            number="7.0",
            title="Dummy Variables",
            icon="📖",
            sections=[
                Markdown("""
**Dummy-Variablen** kodieren kategoriale Prädiktoren als 0/1 Variablen:
"""),
                
                Expander("📊 Beispiel: Geschlecht als Prädiktor", [
                    Table(
                        headers=["Person", "Geschlecht", "Dummy (D)"],
                        rows=[
                            ["1", "männlich", "0"],
                            ["2", "weiblich", "1"],
                            ["3", "männlich", "0"],
                            ["4", "weiblich", "1"],
                        ]
                    ),
                    Markdown("""
**Das Modell:**

Y = β₀ + β₁·X + β₂·D + ε

- D = 0 (männlich): Y = β₀ + β₁·X
- D = 1 (weiblich): Y = (β₀ + β₂) + β₁·X

**β₂** ist der Unterschied im Intercept zwischen den Gruppen.
"""),
                    Plot("dummy_variable_demo", "Dummy Variable Demo", height=400),
                ], expanded=True),
                
                Expander("⚠️ Die Dummy-Falle", [
                    WarningBox("""
**Dummy-Falle:** Bei k Kategorien dürfen nur **k-1** Dummy-Variablen verwendet werden!

Mit k Dummies entsteht perfekte Multikollinearität (D₁ + D₂ + ... + Dₖ = 1).
"""),
                    Markdown("""
**Beispiel mit 3 Gruppen (A, B, C):**

Korrekt: 2 Dummies (D_B, D_C), Gruppe A ist Referenz
- Y = β₀ + β₁·D_B + β₂·D_C + ε

Falsch: 3 Dummies → X'X ist singulär!
"""),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8: DIAGNOSTICS
    # =========================================================================
    def _chapter_8_diagnostics(self) -> Chapter:
        """Chapter 8: Residual diagnostics."""
        s = self.stats
        
        return Chapter(
            number="8.0",
            title="Residual Diagnostics",
            icon="📖",
            sections=[
                Markdown("""
Die **Residuenanalyse** prüft die Modellannahmen:
"""),
                
                Expander("📋 Die Annahmen im Überblick", [
                    Table(
                        headers=["Annahme", "Was prüfen?", "Diagnose-Plot"],
                        rows=[
                            ["Linearität", "E(ε|X) = 0", "Residuen vs. Fitted"],
                            ["Homoskedastizität", "Var(ε|X) = σ²", "Scale-Location"],
                            ["Normalität", "ε ~ N(0, σ²)", "Q-Q Plot"],
                            ["Unabhängigkeit", "Cov(εᵢ, εⱼ) = 0", "Residuen vs. Zeit"],
                        ]
                    ),
                ], expanded=True),
                
                Plot("diagnostics_4panel", "4-Panel Residuendiagnostik", height=600),
                
                Expander("🔬 Durbin-Watson Test", [
                    Markdown("Testet auf **Autokorrelation** der Residuen:"),
                    Formula(r"DW = \frac{\sum_{i=2}^{n}(e_i - e_{i-1})^2}{\sum_{i=1}^{n}e_i^2}"),
                    
                    Metric("Durbin-Watson", self.fmt(s.get('durbin_watson', 2), 3)),
                    
                    Markdown("""
**Interpretation:**
- DW ≈ 2: Keine Autokorrelation ✅
- DW < 1.5: Positive Autokorrelation ⚠️
- DW > 2.5: Negative Autokorrelation ⚠️
"""),
                    self._interpret_dw_box(s.get('durbin_watson', 2)),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9: PREDICTION
    # =========================================================================
    def _chapter_9_prediction(self) -> Chapter:
        """Chapter 9: Prediction and forecasting."""
        s = self.stats
        
        return Chapter(
            number="9.0",
            title="Prediction",
            icon="📖",
            sections=[
                Markdown("""
Mit dem geschätzten Modell können wir **Vorhersagen** für neue Beobachtungen machen:
"""),
                Formula(r"\hat{y}_{new} = \hat{\beta}_0 + \hat{\beta}_1 x_{1,new} + \hat{\beta}_2 x_{2,new}"),
                
                Expander("🎛️ Interaktive Prognose", [
                    Plot("interactive_prediction", "Interaktive Prognose", height=400),
                    InfoBox("""
**Konfidenzintervall:** Unsicherheit über den **mittleren** Y-Wert bei gegebenem X

**Prognoseintervall:** Unsicherheit über einen **einzelnen** Y-Wert bei gegebenem X
(breiter, da zusätzliche Zufallsvariation)
"""),
                ], expanded=True),
                
                Expander("📏 Konfidenz- vs. Prognoseintervall", [
                    Formula(r"SE_{CI} = s_e \sqrt{\mathbf{x}_{new}'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_{new}}"),
                    Formula(r"SE_{PI} = s_e \sqrt{1 + \mathbf{x}_{new}'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_{new}}"),
                    
                    Markdown("""
**Unterschied:**
- **Konfidenzintervall**: Schätzt E(Y|X) - den Mittelwert
- **Prognoseintervall**: Schätzt Y|X - einen einzelnen Wert
"""),
                ]),
                
                # Conclusion
                Markdown("---"),
                SuccessBox(f"""
### 📝 Zusammenfassung

**Modell:** {s.get('y_label', 'Y')} = {self.fmt(s.get('intercept', 0))} + {self.fmt(s.get('beta1', 0))}·{s.get('x1_label', 'X₁')} + {self.fmt(s.get('beta2', 0))}·{s.get('x2_label', 'X₂')}

**Güte:** R² = {self.fmt(s.get('r_squared', 0))}, R²adj = {self.fmt(s.get('r_squared_adj', 0))}

**Signifikanz:** F = {self.fmt(s.get('f_statistic', 0), 2)}, p = {self.fmt(s.get('p_f', 0))}

**Stichprobe:** n = {s.get('n', 0)}
"""),
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
    
    def _build_coefficient_table(self, s: Dict[str, Any]) -> Table:
        """Build coefficient table for multiple regression."""
        return Table(
            headers=["Parameter", "Schätzwert", "Std.Error", "t-Wert", "p-Wert", "Signif."],
            rows=[
                ["β₀ (Intercept)", 
                 self.fmt(s.get('intercept', 0)), 
                 self.fmt(s.get('se_intercept', 0)),
                 self.fmt(s.get('t_intercept', 0), 3),
                 self.fmt(s.get('p_intercept', 0)),
                 self._get_stars(s.get('p_intercept', 1))],
                [f"β₁ ({s.get('x1_label', 'X₁')})", 
                 self.fmt(s.get('beta1', 0)), 
                 self.fmt(s.get('se_beta1', 0)),
                 self.fmt(s.get('t_beta1', 0), 3),
                 self.fmt(s.get('p_beta1', 0)),
                 self._get_stars(s.get('p_beta1', 1))],
                [f"β₂ ({s.get('x2_label', 'X₂')})", 
                 self.fmt(s.get('beta2', 0)), 
                 self.fmt(s.get('se_beta2', 0)),
                 self.fmt(s.get('t_beta2', 0), 3),
                 self.fmt(s.get('p_beta2', 0)),
                 self._get_stars(s.get('p_beta2', 1))],
            ]
        )
    
    def _interpret_p_value_box(self, p: float, test_name: str) -> ContentElement:
        """Create interpretation box for p-value."""
        if p < 0.05:
            return SuccessBox(f"✅ {test_name} signifikant bei α=0.05")
        else:
            return WarningBox(f"⚠️ {test_name} nicht signifikant")
    
    def _interpret_correlation_box(self, r: float) -> ContentElement:
        """Create interpretation box for correlation."""
        abs_r = abs(r) if r else 0
        if abs_r > 0.8:
            return WarningBox("⚠️ Sehr hohe Korrelation - Multikollinearität möglich!")
        elif abs_r > 0.6:
            return InfoBox("ℹ️ Moderate Korrelation - beobachten")
        else:
            return SuccessBox("✅ Geringe Korrelation - kein Problem")
    
    def _interpret_vif_box(self, vif: float) -> ContentElement:
        """Create interpretation box for VIF."""
        if vif < 5:
            return SuccessBox("✅ Keine problematische Multikollinearität")
        elif vif < 10:
            return WarningBox("⚠️ Moderate Multikollinearität")
        else:
            return WarningBox("🚨 Schwere Multikollinearität - Massnahmen erforderlich!")
    
    def _interpret_dw_box(self, dw: float) -> ContentElement:
        """Create interpretation box for Durbin-Watson."""
        if 1.5 <= dw <= 2.5:
            return SuccessBox("✅ Keine signifikante Autokorrelation")
        elif dw < 1.5:
            return WarningBox("⚠️ Positive Autokorrelation möglich")
        else:
            return WarningBox("⚠️ Negative Autokorrelation möglich")
