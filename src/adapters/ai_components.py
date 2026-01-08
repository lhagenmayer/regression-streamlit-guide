"""
AI UI Components - Framework-specific UI for Perplexity Integration.

This module belongs to the ADAPTERS layer, not the AI layer,
because it contains framework-specific code (Streamlit, Flask/HTML).

The core PerplexityClient is in src/ai/ (framework-agnostic).

Provides ready-to-use components for Streamlit and HTML/Flask.
"""

from typing import Dict, Any, Optional
from ..ai import PerplexityClient


class AIInterpretationStreamlit:
    """
    Streamlit component for AI interpretation.
    
    Usage:
        ai_component = AIInterpretationStreamlit(stats)
        ai_component.render()
    """
    
    def __init__(self, stats: Dict[str, Any], client: Optional[PerplexityClient] = None):
        self.stats = stats
        self.client = client or PerplexityClient()
    
    def render(self):
        """Render the AI interpretation component in Streamlit."""
        import streamlit as st
        
        st.markdown("---")
        st.markdown("### 🤖 AI-Interpretation (Perplexity)")
        
        # Status anzeigen
        if self.client.is_configured:
            st.success("✅ Perplexity API verbunden")
        else:
            st.warning("⚠️ Kein API-Key konfiguriert. Fallback-Interpretation wird verwendet.")
            st.caption("API-Key in `.streamlit/secrets.toml` oder `PERPLEXITY_API_KEY` Umgebungsvariable setzen.")
        
        # Interpretation Button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            Klicke auf den Button, um eine **gesamtheitliche AI-Interpretation** 
            des R-Outputs zu erhalten. Die AI erklärt alle statistischen Werte 
            verständlich und im Kontext deiner Analyse.
            """)
        
        with col2:
            interpret_button = st.button(
                "🔍 Interpretieren",
                type="primary",
                use_container_width=True,
                key="ai_interpret_btn"
            )
        
        # Session State für Interpretation
        if "ai_interpretation" not in st.session_state:
            st.session_state.ai_interpretation = None
        
        if interpret_button:
            with st.spinner("🤖 AI analysiert den R-Output..."):
                response = self.client.interpret_r_output(self.stats)
                st.session_state.ai_interpretation = response
        
        # Interpretation anzeigen
        if st.session_state.ai_interpretation:
            response = st.session_state.ai_interpretation
            
            # Container mit Styling
            with st.container():
                st.markdown(response.content)
                
                # Metadaten
                if not response.error:
                    cols = st.columns(4)
                    cols[0].caption(f"📡 Model: {response.model}")
                    cols[1].caption(f"⏱️ {response.latency_ms:.0f}ms")
                    if response.usage:
                        cols[2].caption(f"📝 {response.usage.get('total_tokens', 'N/A')} Tokens")
                    cols[3].caption(f"💾 Cache: {'Ja' if response.cached else 'Nein'}")
                
                # Zitationen falls vorhanden
                if response.citations:
                    with st.expander("📚 Quellen"):
                        for i, citation in enumerate(response.citations, 1):
                            st.markdown(f"{i}. {citation}")
    
    def render_streaming(self):
        """Render with streaming response."""
        import streamlit as st
        
        st.markdown("---")
        st.markdown("### 🤖 AI-Interpretation (Streaming)")
        
        if st.button("🔍 Interpretieren (Live)", type="primary", key="ai_stream_btn"):
            response_container = st.empty()
            full_response = ""
            
            for chunk in self.client.stream_interpretation(self.stats):
                full_response += chunk
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)


class AIInterpretationHTML:
    """
    HTML/Flask component for AI interpretation.
    
    Returns HTML that can be embedded in Flask templates.
    """
    
    def __init__(self, stats: Dict[str, Any], client: Optional[PerplexityClient] = None):
        self.stats = stats
        self.client = client or PerplexityClient()
    
    def render_button(self) -> str:
        """Render the interpretation button HTML."""
        is_configured = self.client.is_configured
        status_class = "success" if is_configured else "warning"
        status_text = "API verbunden" if is_configured else "Kein API-Key"
        
        return f'''
        <div class="ai-interpretation-section mt-4 p-4 rounded" style="background: var(--bg-secondary); border: 1px solid var(--border-color);">
            <div class="d-flex align-items-center justify-content-between mb-3">
                <h4 class="mb-0">
                    <i class="bi bi-robot me-2"></i>
                    AI-Interpretation
                </h4>
                <span class="badge bg-{status_class}">
                    <i class="bi bi-{"check-circle" if is_configured else "exclamation-triangle"} me-1"></i>
                    {status_text}
                </span>
            </div>
            
            <p class="text-secondary mb-3">
                Erhalte eine gesamtheitliche AI-Interpretation des R-Outputs. 
                Perplexity AI erklärt alle statistischen Werte verständlich.
            </p>
            
            <button class="btn btn-primary" 
                    id="ai-interpret-btn"
                    hx-post="/api/interpret"
                    hx-target="#ai-response"
                    hx-swap="innerHTML"
                    hx-indicator="#ai-loading">
                <i class="bi bi-search me-2"></i>
                R-Output interpretieren
            </button>
            
            <div id="ai-loading" class="htmx-indicator ms-3">
                <div class="spinner-border spinner-border-sm text-primary" role="status">
                    <span class="visually-hidden">Lädt...</span>
                </div>
                <span class="ms-2">AI analysiert...</span>
            </div>
            
            <div id="ai-response" class="mt-4"></div>
        </div>
        '''
    
    def render_response(self, response) -> str:
        """Render the AI response as HTML."""
        import markdown
        
        # Convert markdown to HTML
        content_html = markdown.markdown(
            response.content,
            extensions=['tables', 'fenced_code']
        )
        
        error_class = "border-danger" if response.error else "border-primary"
        
        metadata_html = ""
        if not response.error:
            metadata_html = f'''
            <div class="d-flex gap-3 mt-3 pt-3 border-top small text-secondary">
                <span><i class="bi bi-cpu me-1"></i>{response.model}</span>
                <span><i class="bi bi-clock me-1"></i>{response.latency_ms:.0f}ms</span>
                <span><i class="bi bi-database me-1"></i>{"Cached" if response.cached else "Live"}</span>
            </div>
            '''
        
        citations_html = ""
        if response.citations:
            citations_list = "".join(f'<li><a href="{c}" target="_blank">{c}</a></li>' 
                                     for c in response.citations)
            citations_html = f'''
            <details class="mt-3">
                <summary class="text-secondary cursor-pointer">
                    <i class="bi bi-book me-1"></i>Quellen ({len(response.citations)})
                </summary>
                <ol class="mt-2 small">{citations_list}</ol>
            </details>
            '''
        
        return f'''
        <div class="card {error_class} fade-in">
            <div class="card-body">
                <div class="ai-content markdown-content">
                    {content_html}
                </div>
                {citations_html}
                {metadata_html}
            </div>
        </div>
        '''


def get_interpretation_for_content(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get AI interpretation data for ContentBuilder integration.
    
    Returns a dict that can be added to the stats for content building.
    """
    client = PerplexityClient()
    
    return {
        "ai_available": client.is_configured,
        "ai_client": client,
    }
