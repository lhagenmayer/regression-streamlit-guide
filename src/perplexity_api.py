"""
Perplexity API integration for statistical interpretation.

This module provides functions to interact with the Perplexity API
to get LLM-based interpretations of statistical model results.
"""

import os
from typing import Dict, Any, Optional
from openai import OpenAI

from .logger import get_logger

logger = get_logger(__name__)


def get_perplexity_api_key() -> Optional[str]:
    """
    Get the Perplexity API key from environment variables or Streamlit secrets.
    
    Returns:
        API key if found, None otherwise
    """
    # First try environment variable
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    
    if api_key:
        return api_key
    
    # Then try Streamlit secrets (if available)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "PERPLEXITY_API_KEY" in st.secrets:
            return st.secrets["PERPLEXITY_API_KEY"]
    except Exception:
        pass
    
    return None


def is_api_configured() -> bool:
    """
    Check if the Perplexity API is properly configured.
    
    Returns:
        True if API key is available, False otherwise
    """
    return get_perplexity_api_key() is not None


def extract_model_statistics(model: Any, feature_names: list) -> Dict[str, Any]:
    """
    Extract relevant statistics from the statsmodels regression model.
    
    Args:
        model: Fitted statsmodels regression model
        feature_names: List of feature names used in the model
        
    Returns:
        Dictionary containing key statistics for interpretation
    """
    try:
        stats = {
            "model_type": "Linear Regression",
            "n_observations": int(model.nobs),
            "n_predictors": len(feature_names),
            "feature_names": feature_names,
            "coefficients": {},
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "residual_std_error": float(model.mse_resid ** 0.5),
            "df_residual": int(model.df_resid),
            "df_model": int(model.df_model),
        }
        
        # Extract coefficient information
        for i, name in enumerate(["Intercept"] + feature_names):
            if i < len(model.params):
                stats["coefficients"][name] = {
                    "estimate": float(model.params[i]),
                    "std_error": float(model.bse[i]),
                    "t_value": float(model.tvalues[i]),
                    "p_value": float(model.pvalues[i]),
                    "significant": model.pvalues[i] < 0.05
                }
        
        # Residual summary statistics
        residuals = model.resid
        stats["residuals"] = {
            "min": float(residuals.min()),
            "q1": float(residuals.quantile(0.25)),
            "median": float(residuals.median()),
            "q3": float(residuals.quantile(0.75)),
            "max": float(residuals.max())
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error extracting model statistics: {e}", exc_info=True)
        return {}


def create_interpretation_prompt(stats: Dict[str, Any]) -> str:
    """
    Create a structured prompt for the Perplexity API to interpret the model.
    
    Args:
        stats: Dictionary containing model statistics
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Analysiere bitte folgendes Regressionsmodell und gib eine verständliche Interpretation in deutscher Sprache:

**Modellübersicht:**
- Modelltyp: {stats.get('model_type', 'N/A')}
- Anzahl Beobachtungen (n): {stats.get('n_observations', 'N/A')}
- Anzahl Prädiktoren: {stats.get('n_predictors', 'N/A')}
- Prädiktoren: {', '.join(stats.get('feature_names', []))}

**Modellgüte:**
- R²: {stats.get('r_squared', 'N/A'):.4f} ({stats.get('r_squared', 0)*100:.1f}% der Varianz erklärt)
- Adjustiertes R²: {stats.get('adj_r_squared', 'N/A'):.4f}
- F-Statistik: {stats.get('f_statistic', 'N/A'):.2f}
- p-Wert (F-Test): {stats.get('f_pvalue', 'N/A'):.6f}
- Residual Standard Error: {stats.get('residual_std_error', 'N/A'):.4f}
- Freiheitsgrade (Residual): {stats.get('df_residual', 'N/A')}

**Koeffizienten:**
"""
    
    # Add coefficient details
    for name, coef in stats.get('coefficients', {}).items():
        signif = "***" if coef['p_value'] < 0.001 else "**" if coef['p_value'] < 0.01 else "*" if coef['p_value'] < 0.05 else ""
        prompt += f"\n- {name}: {coef['estimate']:.4f} (SE: {coef['std_error']:.4f}, t: {coef['t_value']:.2f}, p: {coef['p_value']:.6f}) {signif}"
    
    # Add residual information
    residuals = stats.get('residuals', {})
    prompt += f"""

**Residuen:**
- Min: {residuals.get('min', 'N/A'):.4f}
- 1Q: {residuals.get('q1', 'N/A'):.4f}
- Median: {residuals.get('median', 'N/A'):.4f}
- 3Q: {residuals.get('q3', 'N/A'):.4f}
- Max: {residuals.get('max', 'N/A'):.4f}

Bitte gib eine strukturierte Interpretation, die folgende Punkte abdeckt:

1. **Modellqualität**: Wie gut ist das Modell? Was sagen R² und F-Statistik aus?
2. **Koeffizienten-Interpretation**: Was bedeuten die Koeffizienten praktisch? Welche Prädiktoren sind signifikant?
3. **Residuen**: Was sagen die Residuen über die Modellanpassung aus?
4. **Praktische Bedeutung**: Was sind die wichtigsten Erkenntnisse für die Praxis?
5. **Empfehlungen**: Gibt es Hinweise auf Verbesserungspotential oder Vorsichtsmaßnahmen?

Halte die Interpretation präzise, verständlich und praxisorientiert (ca. 300-400 Wörter).
"""
    
    return prompt


def get_interpretation_from_perplexity(
    stats: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "llama-3.1-sonar-large-128k-online"
) -> Dict[str, Any]:
    """
    Get interpretation from Perplexity API.
    
    Args:
        stats: Dictionary containing model statistics
        api_key: Perplexity API key (if None, will try to get from environment)
        model_name: Model to use for interpretation
        
    Returns:
        Dictionary with 'success', 'interpretation', and optional 'error' fields
    """
    if api_key is None:
        api_key = get_perplexity_api_key()
    
    if not api_key:
        logger.warning("Perplexity API key not found")
        return {
            "success": False,
            "error": "API-Schlüssel nicht konfiguriert. Bitte setzen Sie die Umgebungsvariable PERPLEXITY_API_KEY."
        }
    
    try:
        # Initialize Perplexity client (OpenAI-compatible API)
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
        # Create the prompt
        prompt = create_interpretation_prompt(stats)
        
        logger.info(f"Calling Perplexity API with model {model_name}")
        
        # Make API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein erfahrener Statistiker und Datenanalyst, der komplexe statistische Modelle in verständlicher deutscher Sprache erklären kann. Gib präzise, praxisorientierte Interpretationen."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=1500
        )
        
        # Extract the interpretation
        interpretation = response.choices[0].message.content
        
        logger.info("Successfully received interpretation from Perplexity API")
        
        return {
            "success": True,
            "interpretation": interpretation
        }
        
    except Exception as e:
        error_msg = f"Fehler bei der API-Anfrage: {str(e)}"
        logger.error(f"Error calling Perplexity API: {e}", exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }


def interpret_model(model: Any, feature_names: list) -> Dict[str, Any]:
    """
    High-level function to interpret a regression model using Perplexity API.
    
    Args:
        model: Fitted statsmodels regression model
        feature_names: List of feature names used in the model
        
    Returns:
        Dictionary with interpretation results
    """
    logger.info("Starting model interpretation")
    
    # Extract statistics
    stats = extract_model_statistics(model, feature_names)
    
    if not stats:
        return {
            "success": False,
            "error": "Fehler beim Extrahieren der Modellstatistiken."
        }
    
    # Get interpretation from API
    result = get_interpretation_from_perplexity(stats)
    
    return result
