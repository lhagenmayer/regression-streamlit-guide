"""
R-Output Formatter.

Generates simulated R-style output strings from statistical dictionaries.
Used to provide context for AI interpretation capabilities.
"""

from typing import Dict, Any, List
import numpy as np

class ROutputFormatter:
    """Format statistics as R-style output."""

    @staticmethod
    def format(stats: Dict[str, Any]) -> str:
        """
        Dispatch formatting based on available keys.
        """
        method = stats.get("method")
        
        if method == "knn":
            return ROutputFormatter.format_knn(stats)
        elif method == "logistic":
            return ROutputFormatter.format_logistic(stats)
        else:
            # Default to linear regression (simple or multiple)
            return ROutputFormatter.format_linear(stats)

    @staticmethod
    def format_linear(stats: Dict[str, Any]) -> str:
        """Format Linear Output (lm)."""
        # Handle residuals
        residuals = stats.get('residuals', [0, 0, 0, 0, 0])
        if hasattr(residuals, 'tolist'): residuals = residuals.tolist()
        if not residuals or len(residuals) < 5: residuals = [0, 0, 0, 0, 0]
        
        try:
            res_min = float(np.min(residuals))
            res_q1 = float(np.percentile(residuals, 25))
            res_med = float(np.median(residuals))
            res_q3 = float(np.percentile(residuals, 75))
            res_max = float(np.max(residuals))
        except:
            res_min = res_q1 = res_med = res_q3 = res_max = 0.0
        
        def get_stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            if p < 0.1: return "."
            return ""
        
        x_label = str(stats.get('x_label', 'X'))[:12]
        y_label = str(stats.get('y_label', 'Y'))
        
        # Check if multiple regression
        if "b1" in stats or "beta1" in stats:
            return ROutputFormatter._format_multiple_linear(stats, y_label, residuals)

        # Simple Regression
        intercept = float(stats.get('intercept', 0))
        slope = float(stats.get('slope', 0))
        
        # Try to get SE/t/p (default to safe values if missing)
        se_int = float(stats.get('se_intercept', 0))
        se_slope = float(stats.get('se_slope', 0))
        t_int = float(stats.get('t_intercept', 0))
        t_slope = float(stats.get('t_slope', 0))
        p_int = float(stats.get('p_intercept', 1))
        p_slope = float(stats.get('p_slope', 1))
        
        r2 = float(stats.get('r_squared', 0))
        r2_adj = float(stats.get('r_squared_adj', 0))
        f_stat = float(stats.get('f_statistic', 0))
        df = int(stats.get('df', 0))
        
        import math
        mse = float(stats.get('mse', 0))
        rmse = math.sqrt(mse) if mse > 0 else 0

        return f"""Call:
lm(formula = {y_label} ~ {x_label})

Residuals:
     Min       1Q   Median       3Q      Max 
{res_min:8.4f} {res_q1:8.4f} {res_med:8.4f} {res_q3:8.4f} {res_max:8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  {intercept:9.4f}   {se_int:9.4f}  {t_int:7.3f}   {p_int:.2e} {get_stars(p_int)}
{x_label:12s} {slope:9.4f}   {se_slope:9.4f}  {t_slope:7.3f}   {p_slope:.2e} {get_stars(p_slope)}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rmse:.4f} on {df} degrees of freedom
Multiple R-squared:  {r2:.4f},    Adjusted R-squared:  {r2_adj:.4f}
F-statistic: {f_stat:.2f} on 1 and {df} DF,  p-value: {p_slope:.2e}"""

    @staticmethod
    def _format_multiple_linear(stats: Dict[str, Any], y_label: str, residuals: List[float]) -> str:
        """Helper for Multiple Regression Output."""
        try:
            res_min = float(np.min(residuals))
            res_q1 = float(np.percentile(residuals, 25))
            res_med = float(np.median(residuals))
            res_q3 = float(np.percentile(residuals, 75))
            res_max = float(np.max(residuals))
        except:
            res_min = res_q1 = res_med = res_q3 = res_max = 0.0

        def get_stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            if p < 0.1: return "."
            return ""

        # Collect Coefficients
        intercept = float(stats.get('intercept', 0))
        se_intercept = float(stats.get('se_intercept', 0))
        t_intercept = float(stats.get('t_intercept', 0))
        p_intercept = float(stats.get('p_intercept', 1))

        # We need a robust way to iterate coefficients for multiple regression
        # The flattened dict might have 'b1', 'beta1', 'slopes', or specific keys
        # We try 'b1', 'b2' pattern or iterate if 'coefficients'/ 'slopes' is list
        
        coeffs_str = f"(Intercept)  {intercept:9.4f}   {se_intercept:9.4f}  {t_intercept:7.3f}   {p_intercept:.2e} {get_stars(p_intercept)}\n"
        
        # Check for list-based coefficients (preferred for general multiple)
        slopes = stats.get('slopes')
        se_slopes = stats.get('se_slopes') # Might need to standardize key for SE list
        # We previously used individual keys like 'se_beta1'
        
        x1_label = str(stats.get('x1_label', 'X1'))
        x2_label = str(stats.get('x2_label', 'X2'))
        
        # Fallback to specific keys if list is missing (legacy compat)
        if slopes is None:
            # Try beta1/beta2
            b1 = float(stats.get('b1') or stats.get('beta1') or 0)
            b2 = float(stats.get('b2') or stats.get('beta2') or 0)
            se_b1 = float(stats.get('se_beta1', 0))
            se_b2 = float(stats.get('se_beta2', 0))
            t_b1 = float(stats.get('t_beta1', 0))
            t_b2 = float(stats.get('t_beta2', 0))
            p_b1 = float(stats.get('p_beta1', 1))
            p_b2 = float(stats.get('p_beta2', 1))
            
            coeffs_str += f"{x1_label:12s} {b1:9.4f}   {se_b1:9.4f}  {t_b1:7.3f}   {p_b1:.2e} {get_stars(p_b1)}\n"
            coeffs_str += f"{x2_label:12s} {b2:9.4f}   {se_b2:9.4f}  {t_b2:7.3f}   {p_b2:.2e} {get_stars(p_b2)}"
        else:
            # Not fully implemented for list yet, fallback to naive
            pass

        r2 = float(stats.get('r_squared', 0))
        r2_adj = float(stats.get('r_squared_adj', 0))
        f_stat = float(stats.get('f_statistic', 0))
        df = int(stats.get('df', 0))
        p_f = float(stats.get('f_p_value') or stats.get('p_f') or 1.0)
        
        import math
        mse = float(stats.get('mse', 0))
        rmse = math.sqrt(mse) if mse > 0 else 0

        return f"""Call:
lm(formula = {y_label} ~ {x1_label} + {x2_label})

Residuals:
     Min       1Q   Median       3Q      Max 
{res_min:8.4f} {res_q1:8.4f} {res_med:8.4f} {res_q3:8.4f} {res_max:8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
{coeffs_str}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rmse:.4f} on {df} degrees of freedom
Multiple R-squared:  {r2:.4f},    Adjusted R-squared:  {r2_adj:.4f}
F-statistic: {f_stat:.2f} on 2 and {df} DF,  p-value: {p_f:.2e}"""

    @staticmethod
    def format_logistic(stats: Dict[str, Any]) -> str:
        """Format Logistic Regression (glm) Output."""
        y_label = stats.get('y_label', 'Class')
        feature_names = stats.get('feature_names', ['X'])
        
        intercept = float(stats.get('intercept', 0))
        coeffs = stats.get('coefficients', [])
        if not isinstance(coeffs, list): coeffs = [coeffs]
        
        accuracy = float(stats.get('accuracy', 0))
        aic = stats.get('aic', 100.0) # Placeholder default
        
        # Note: We don't have SE/z-values for Logistic yet in our simple implementation
        # We fill with NA to be honest, but keep structure perfect
        
        coef_str = f"(Intercept)  {intercept:9.4f}   NA         NA      NA\n"
        for i, name in enumerate(feature_names):
            val = coeffs[i] if i < len(coeffs) else 0.0
            coef_str += f"{name:12s} {float(val):9.4f}   NA         NA      NA\n"

        cm = stats.get('confusion_matrix')
        cm_str = str(np.array(cm)) if cm is not None else "NA"

        return f"""Call:
glm(formula = {y_label} ~ {' + '.join(feature_names)}, family = binomial)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
     NA       NA       NA       NA       NA  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)
{coef_str}
(Dispersion parameter for binomial family taken to be 1)

    Null deviance: NA  on NA  degrees of freedom
Residual deviance: NA  on NA  degrees of freedom
AIC: {aic}

Number of Fisher Scoring iterations: NA

Confusion Matrix:
{cm_str}

Accuracy: {accuracy:.4f}"""

    @staticmethod
    def format_knn(stats: Dict[str, Any]) -> str:
        """Format KNN (caret::confusionMatrix style) Output."""
        k = stats.get('k', 3)
        accuracy = float(stats.get('accuracy', 0))
        cm = stats.get('confusion_matrix')
        
        # R's caret package confusionMatrix output structure
        
        return f"""Confusion Matrix and Statistics

          Reference
Prediction {np.array(cm) if cm is not None else 'NA'}

Overall Statistics
                                          
               Accuracy : {accuracy:.4f}          
                 95% CI : (NA, NA)
    No Information Rate : NA              
    P-Value [Acc > NIR] : NA              
                                          
                  Kappa : NA              
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: 0  Class: 1
Sensitivity                NA        NA
Specificity                NA        NA
Pos Pred Value             NA        NA
Neg Pred Value             NA        NA
Prevalence                 NA        NA
Detection Rate             NA        NA
Detection Prevalence       NA        NA
Balanced Accuracy          NA        NA

Model Information:
Method: k-Nearest Neighbors (k={k})"""
