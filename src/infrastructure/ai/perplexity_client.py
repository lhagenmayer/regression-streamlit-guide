"""
Perplexity AI Client - 100% Platform Agnostic.

Pure Python implementation with NO framework dependencies.
Can be used by ANY frontend: Flask, Streamlit, Next.js, Vite, etc.

All interaction happens via:
1. Direct Python calls (for Python backends)
2. REST API (for any frontend via /api/interpret)
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Generator, List
from enum import Enum

import requests

# Use standard logging - no framework dependencies
import logging
logger = logging.getLogger(__name__)


class PerplexityModel(Enum):
    """Available Perplexity models."""
    SONAR_SMALL = "llama-3.1-sonar-small-128k-online"
    SONAR_LARGE = "llama-3.1-sonar-large-128k-online"
    SONAR_HUGE = "llama-3.1-sonar-huge-128k-online"


@dataclass
class PerplexityConfig:
    """Configuration for Perplexity API - no framework dependencies."""
    api_key: Optional[str] = None
    model: PerplexityModel = PerplexityModel.SONAR_SMALL
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 60
    
    def __post_init__(self):
        """Load API key from environment only - framework agnostic."""
        if self.api_key is None:
            self.api_key = os.environ.get("PERPLEXITY_API_KEY")


@dataclass
class PerplexityResponse:
    """
    Response from Perplexity API - serializable to JSON.
    
    Can be directly returned via REST API to any frontend.
    """
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)
    cached: bool = False
    latency_ms: float = 0.0
    error: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "citations": self.citations,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ROutputData:
    """
    Structured R-Output data - platform agnostic.
    
    This is the data structure that any frontend can send
    to request an interpretation.
    """
    # Required fields
    x_label: str
    y_label: str
    n: int
    intercept: float
    slope: float
    r_squared: float
    
    # Optional fields with defaults
    context_title: str = "Regressionsanalyse"
    context_description: str = ""
    r_squared_adj: float = 0.0
    se_intercept: float = 0.0
    se_slope: float = 0.0
    t_intercept: float = 0.0
    t_slope: float = 0.0
    p_intercept: float = 1.0
    p_slope: float = 1.0
    mse: float = 0.0
    sse: float = 0.0
    ssr: float = 0.0
    df: int = 0
    residuals: List[float] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROutputData":
        """Create from dictionary (e.g., from JSON request)."""
        return cls(
            x_label=data.get("x_label", "X"),
            y_label=data.get("y_label", "Y"),
            n=data.get("n", 0),
            intercept=data.get("intercept", 0.0),
            slope=data.get("slope", 0.0),
            r_squared=data.get("r_squared", 0.0),
            context_title=data.get("context_title", "Regressionsanalyse"),
            context_description=data.get("context_description", ""),
            r_squared_adj=data.get("r_squared_adj", 0.0),
            se_intercept=data.get("se_intercept", 0.0),
            se_slope=data.get("se_slope", 0.0),
            t_intercept=data.get("t_intercept", 0.0),
            t_slope=data.get("t_slope", 0.0),
            p_intercept=data.get("p_intercept", 1.0),
            p_slope=data.get("p_slope", 1.0),
            mse=data.get("mse", 0.0),
            sse=data.get("sse", 0.0),
            ssr=data.get("ssr", 0.0),
            df=data.get("df", 0),
            residuals=data.get("residuals", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PerplexityClient:
    """
    Platform-Agnostic Perplexity AI Client.
    
    100% framework independent - works with:
    - Flask
    - Streamlit  
    - FastAPI
    - Next.js (via REST API)
    - Vite/React (via REST API)
    - Any HTTP client
    
    Usage (Python):
        client = PerplexityClient()
        response = client.interpret_r_output(stats_dict)
        print(response.content)
    
    Usage (REST API - any frontend):
        POST /api/ai/interpret
        Body: { "stats": { ... } }
        Response: { "content": "...", "model": "...", ... }
    """
    
    BASE_URL = "https://api.perplexity.ai/chat/completions"
    
    # System prompt - can be customized
    SYSTEM_PROMPT = """Du bist ein erfahrener Statistik-Professor, der Studierenden Regressionsanalysen erklärt.

Deine Aufgabe: Interpretiere den R-Output einer Regressionsanalyse GESAMTHEITLICH und VERSTÄNDLICH.

Struktur deiner Antwort:
1. **Zusammenfassung** (2-3 Sätze): Was sagt das Modell aus?
2. **Koeffizienten-Interpretation**: Was bedeuten β₀ und β₁ praktisch?
3. **Modellgüte**: Wie gut erklärt das Modell die Daten? (R², F-Test)
4. **Signifikanz**: Sind die Ergebnisse statistisch bedeutsam? (p-Werte, t-Tests)
5. **Praktische Bedeutung**: Was bedeutet das für die Praxis?
6. **Einschränkungen**: Worauf sollte man achten?

Regeln:
- Antworte IMMER auf Deutsch
- Verwende die konkreten Variablennamen aus dem Output
- Interpretiere ALLE Werte, nicht nur einzelne
- Erkläre für Studierende verständlich, aber fachlich korrekt
- Verwende Emojis sparsam für Struktur (📊, ✅, ⚠️)"""

    def __init__(self, config: Optional[PerplexityConfig] = None):
        """Initialize client with optional config."""
        self.config = config or PerplexityConfig()
        self._cache: Dict[str, PerplexityResponse] = {}
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.config.api_key)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get client status - useful for health checks.
        
        Returns:
            Dict with status info for any frontend
        """
        return {
            "configured": self.is_configured,
            "model": self.config.model.value,
            "cache_size": len(self._cache),
        }
    
    def interpret_r_output(
        self, 
        stats: Dict[str, Any],
        use_cache: bool = True
    ) -> PerplexityResponse:
        """
        Interpret R-output comprehensively.
        
        This is the main method - works with any frontend.
        
        Args:
            stats: Dictionary with regression statistics
                   (can come from JSON request body)
            use_cache: Whether to use cached responses
            
        Returns:
            PerplexityResponse that can be serialized to JSON
        """
        if not self.is_configured:
            return PerplexityResponse(
                content=self._get_fallback_interpretation(stats),
                model="fallback",
                cached=False,
                error=False  # Fallback is not an error, just no API
            )
        
        # Generate R-output text
        r_output = self.generate_r_output(stats)
        
        # Check cache
        cache_key = self._cache_key(r_output)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.cached = True
            return cached
        
        # Build prompt
        user_prompt = self._build_prompt(stats, r_output)
        
        # Make API request
        return self._make_request(user_prompt, cache_key, use_cache)
    
    def interpret_from_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret from a REST API request body.
        
        Designed for easy integration with any REST framework.
        
        Args:
            request_data: JSON request body with 'stats' field
            
        Returns:
            Dictionary ready for JSON response
        """
        stats = request_data.get("stats", {})
        use_cache = request_data.get("use_cache", True)
        
        response = self.interpret_r_output(stats, use_cache)
        return response.to_dict()
    
    def generate_r_output(self, stats: Dict[str, Any]) -> str:
        """
        Generate R-style output from statistics.
        
        Public method - can be called by any frontend to show
        the R-output before interpretation.
        
        Args:
            stats: Dictionary with regression statistics
            
        Returns:
            Formatted R-output string
        """
        import numpy as np
        
        # Handle residuals
        residuals = stats.get('residuals', [0, 0, 0, 0, 0])
        if hasattr(residuals, 'tolist'):
            residuals = residuals.tolist()
        if len(residuals) < 5:
            residuals = [0, 0, 0, 0, 0]
        
        res_min = float(np.min(residuals))
        res_q1 = float(np.percentile(residuals, 25))
        res_med = float(np.median(residuals))
        res_q3 = float(np.percentile(residuals, 75))
        res_max = float(np.max(residuals))
        
        def get_stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            if p < 0.1: return "."
            return ""
        
        x_label = str(stats.get('x_label', 'X'))[:12]
        y_label = str(stats.get('y_label', 'Y'))
        
        intercept = float(stats.get('intercept', 0))
        slope = float(stats.get('slope', 0))
        se_intercept = float(stats.get('se_intercept', 0))
        se_slope = float(stats.get('se_slope', 0))
        t_intercept = float(stats.get('t_intercept', 0))
        t_slope = float(stats.get('t_slope', 0))
        p_intercept = float(stats.get('p_intercept', 1))
        p_slope = float(stats.get('p_slope', 1))
        
        r_squared = float(stats.get('r_squared', 0))
        r_squared_adj = float(stats.get('r_squared_adj', 0))
        mse = float(stats.get('mse', 0))
        df = int(stats.get('df', 0))
        
        # F-statistic
        ssr = float(stats.get('ssr', 0))
        sse = float(stats.get('sse', 1))
        f_stat = (ssr / 1) / (sse / df) if df > 0 and sse > 0 else 0
        
        import math
        rmse = math.sqrt(mse) if mse > 0 else 0
        
        return f"""Call:
lm(formula = {y_label} ~ {x_label})

Residuals:
     Min       1Q   Median       3Q      Max 
{res_min:8.4f} {res_q1:8.4f} {res_med:8.4f} {res_q3:8.4f} {res_max:8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  {intercept:9.4f}   {se_intercept:9.4f}  {t_intercept:7.3f}   {p_intercept:.2e} {get_stars(p_intercept)}
{x_label:12s} {slope:9.4f}   {se_slope:9.4f}  {t_slope:7.3f}   {p_slope:.2e} {get_stars(p_slope)}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rmse:.4f} on {df} degrees of freedom
Multiple R-squared:  {r_squared:.4f},    Adjusted R-squared:  {r_squared_adj:.4f}
F-statistic: {f_stat:.2f} on 1 and {df} DF,  p-value: {p_slope:.2e}"""

    def stream_interpretation(
        self, 
        stats: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Stream interpretation chunks.
        
        For real-time display in any frontend supporting SSE.
        
        Yields:
            Text chunks as they arrive
        """
        if not self.is_configured:
            yield self._get_fallback_interpretation(stats)
            return
        
        r_output = self.generate_r_output(stats)
        user_prompt = self._build_prompt(stats, r_output)
        
        try:
            response = requests.post(
                self.BASE_URL,
                headers=self._get_headers(),
                json={
                    "model": self.config.model.value,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True,
                },
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            pass
                            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n\n❌ Streaming fehlgeschlagen: {e}"
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the response cache.
        
        Returns:
            Status dict for API response
        """
        count = len(self._cache)
        self._cache.clear()
        return {"cleared": count, "cache_size": 0}
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
    
    def _cache_key(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _build_prompt(self, stats: Dict[str, Any], r_output: str) -> str:
        return f"""Hier ist der vollständige R-Output einer Regressionsanalyse.
Bitte interpretiere ALLE Werte gesamtheitlich und erkläre, was sie bedeuten.

**Kontext der Analyse:**
{stats.get('context_title', 'Regressionsanalyse')}
{stats.get('context_description', '')}

**R-Output:**
```
{r_output}
```

**Zusätzliche Informationen:**
- Unabhängige Variable (X): {stats.get('x_label', 'X')}
- Abhängige Variable (Y): {stats.get('y_label', 'Y')}
- Stichprobengrösse: n = {stats.get('n', 'N/A')}

Bitte gib eine vollständige, gesamtheitliche Interpretation."""

    def _make_request(
        self, 
        user_prompt: str, 
        cache_key: str, 
        use_cache: bool
    ) -> PerplexityResponse:
        try:
            start_time = time.time()
            
            response = requests.post(
                self.BASE_URL,
                headers=self._get_headers(),
                json={
                    "model": self.config.model.value,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout
            )
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 429:
                return PerplexityResponse(
                    content="⚠️ Rate Limit erreicht. Bitte versuche es in einer Minute erneut.",
                    model="error",
                    error=True,
                    latency_ms=latency
                )
            
            response.raise_for_status()
            data = response.json()
            
            result = PerplexityResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.config.model.value),
                usage=data.get("usage", {}),
                citations=data.get("citations", []),
                cached=False,
                latency_ms=latency
            )
            
            if use_cache:
                self._cache[cache_key] = result
            
            logger.info(f"Perplexity interpretation in {latency:.0f}ms")
            return result
            
        except requests.exceptions.Timeout:
            return PerplexityResponse(
                content="⚠️ Zeitüberschreitung bei der API-Anfrage.",
                model="error",
                error=True
            )
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            return PerplexityResponse(
                content=f"❌ API-Fehler: {str(e)}",
                model="error",
                error=True
            )
    
    def _get_fallback_interpretation(self, stats: Dict[str, Any]) -> str:
        """Generate interpretation without API."""
        r2 = float(stats.get('r_squared', 0))
        slope = float(stats.get('slope', 0))
        p_slope = float(stats.get('p_slope', 1))
        n = int(stats.get('n', 0))
        x_label = stats.get('x_label', 'X')
        y_label = stats.get('y_label', 'Y')
        intercept = float(stats.get('intercept', 0))
        
        # R² interpretation
        if r2 >= 0.8:
            r2_quality = "sehr gut"
            r2_emoji = "✅"
        elif r2 >= 0.6:
            r2_quality = "gut"
            r2_emoji = "✅"
        elif r2 >= 0.4:
            r2_quality = "moderat"
            r2_emoji = "⚠️"
        else:
            r2_quality = "schwach"
            r2_emoji = "⚠️"
        
        # Significance
        if p_slope < 0.001:
            sig_text = "höchst signifikant (p < 0.001) ✅"
            sig_stars = "***"
        elif p_slope < 0.01:
            sig_text = "sehr signifikant (p < 0.01) ✅"
            sig_stars = "**"
        elif p_slope < 0.05:
            sig_text = "signifikant (p < 0.05) ✅"
            sig_stars = "*"
        else:
            sig_text = "nicht signifikant (p ≥ 0.05) ⚠️"
            sig_stars = ""
        
        direction = "positiven" if slope > 0 else "negativen"
        change = "steigt" if slope > 0 else "sinkt"
        
        return f"""## 📊 Interpretation der Regressionsanalyse

### 1. Zusammenfassung
Das Modell zeigt einen **{direction}** Zusammenhang zwischen {x_label} und {y_label}. 
Bei einer Stichprobe von n = {n} erklärt das Modell {r2*100:.1f}% der Varianz.

### 2. Koeffizienten-Interpretation

**Intercept (β₀ = {intercept:.4f}):**
Wenn {x_label} = 0, beträgt der erwartete Wert von {y_label} etwa {intercept:.2f}.

**Steigung (β₁ = {slope:.4f}):**
Pro Einheit Zunahme in {x_label} {change} {y_label} um **{abs(slope):.4f}** Einheiten.

### 3. Modellgüte
- **R² = {r2:.4f}** → Das Modell erklärt **{r2*100:.1f}%** der Varianz {r2_emoji}
- Bewertung: {r2_quality}

### 4. Signifikanz
Der Zusammenhang ist **{sig_text}**
- p-Wert: {p_slope:.4f} {sig_stars}

### 5. Praktische Bedeutung
{"Der Zusammenhang ist statistisch bedeutsam und kann für Vorhersagen genutzt werden." if p_slope < 0.05 else "Der Zusammenhang ist statistisch nicht gesichert. Vorsicht bei Interpretationen."}

### 6. Einschränkungen
- Korrelation ≠ Kausalität
- Gültigkeit nur im beobachteten Wertebereich
- Prüfe Residuen auf Muster

---
*Automatische Interpretation (ohne Perplexity AI)*"""
