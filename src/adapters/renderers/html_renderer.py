"""
HTML Content Renderer - Interprets ContentStructure for Flask/Jinja2.

This renderer takes framework-agnostic ContentStructure and renders it
as HTML that can be used in Flask templates or standalone.
"""

import json
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

from ...content.structure import (
    ContentElement, Chapter, Section, EducationalContent,
    Markdown, Metric, MetricRow, Formula, Plot, Table,
    Columns, Expander, InfoBox, WarningBox, SuccessBox,
    CodeBlock, Divider, ElementType
)


class HTMLContentRenderer:
    """
    Renders EducationalContent as HTML.
    
    The same ContentStructure can be rendered by different renderers
    (StreamlitContentRenderer, HTMLContentRenderer, etc.)
    """
    
    def __init__(
        self, 
        plots: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        stats: Dict[str, Any] = None
    ):
        """
        Initialize renderer with plot figures and data.
        
        Args:
            plots: Dictionary of plot_key -> Plotly figure JSON
            data: Dictionary with x, y arrays and metadata
            stats: Dictionary with regression statistics
        """
        self.plots = plots or {}
        self.data = data or {}
        self.stats_data = stats or {}
        
        # Counter for unique IDs
        self._id_counter = 0
    
    def _get_unique_id(self, prefix: str = "elem") -> str:
        """Generate unique element ID."""
        self._id_counter += 1
        return f"{prefix}_{self._id_counter}"
    
    def render(self, content: EducationalContent) -> str:
        """Render complete educational content as HTML."""
        html_parts = [
            f'<div class="educational-content">',
            f'<h1>{content.title}</h1>',
            f'<p class="subtitle"><em>{content.subtitle}</em></p>',
        ]
        
        for chapter in content.chapters:
            html_parts.append(self._render_chapter(chapter))
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _render_chapter(self, chapter: Chapter) -> str:
        """Render a chapter with its sections."""
        html_parts = [
            f'<div class="chapter fade-in" id="chapter-{chapter.number.replace(".", "-")}">',
            f'''<div class="chapter-header">
                <span class="chapter-icon">{chapter.icon}</span>
                <span>Kapitel {chapter.number}: {chapter.title}</span>
            </div>''',
        ]
        
        for section in chapter.sections:
            html_parts.append(self._render_element(section))
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _render_element(self, element: ContentElement) -> str:
        """Render a single content element as HTML."""
        if isinstance(element, Markdown):
            return self._render_markdown(element)
        
        elif isinstance(element, Metric):
            return self._render_metric(element)
        
        elif isinstance(element, MetricRow):
            return self._render_metric_row(element)
        
        elif isinstance(element, Formula):
            return self._render_formula(element)
        
        elif isinstance(element, Plot):
            return self._render_plot(element)
        
        elif isinstance(element, Table):
            return self._render_table(element)
        
        elif isinstance(element, Columns):
            return self._render_columns(element)
        
        elif isinstance(element, Expander):
            return self._render_expander(element)
        
        elif isinstance(element, InfoBox):
            return self._render_info_box(element)
        
        elif isinstance(element, WarningBox):
            return self._render_warning_box(element)
        
        elif isinstance(element, SuccessBox):
            return self._render_success_box(element)
        
        elif isinstance(element, CodeBlock):
            return self._render_code_block(element)
        
        elif isinstance(element, Divider):
            return '<hr>'
        
        elif isinstance(element, Section):
            return self._render_section(element)
        
        return f'<!-- Unknown element type: {type(element).__name__} -->'
    
    def _render_markdown(self, element: Markdown) -> str:
        """Render markdown as HTML."""
        # For simplicity, just wrap in div with markdown class
        # In production, use a markdown library
        return f'<div class="markdown-content">{self._simple_md_to_html(element.text)}</div>'
    
    def _simple_md_to_html(self, text: str) -> str:
        """Simple markdown to HTML conversion."""
        import re
        
        # Headers
        text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        
        # Bold and italic
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        
        # Lists
        text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.+</li>\n)+', r'<ul>\g<0></ul>', text)
        
        # Numbered lists
        text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        
        # Checkboxes
        text = re.sub(r'\[ \]', '☐', text)
        text = re.sub(r'\[x\]', '☑', text)
        
        # Paragraphs
        text = re.sub(r'\n\n', '</p><p>', text)
        
        return f'<p>{text}</p>'
    
    def _render_metric(self, element: Metric) -> str:
        """Render a metric display."""
        help_attr = f'title="{element.help_text}"' if element.help_text else ''
        delta_html = f'<div class="metric-delta">{element.delta}</div>' if element.delta else ''
        
        return f'''
        <div class="metric-card" {help_attr}>
            <div class="metric-label">{element.label}</div>
            <div class="metric-value">{element.value}</div>
            {delta_html}
        </div>
        '''
    
    def _render_metric_row(self, element: MetricRow) -> str:
        """Render a row of metrics."""
        metrics_html = '\n'.join(self._render_metric(m) for m in element.metrics)
        return f'<div class="metrics-grid">{metrics_html}</div>'
    
    def _render_formula(self, element: Formula) -> str:
        """Render LaTeX formula."""
        if element.inline:
            return f'<span class="math-inline">\\({element.latex}\\)</span>'
        else:
            return f'<div class="math-display">\\[{element.latex}\\]</div>'
    
    def _render_plot(self, element: Plot) -> str:
        """Render a plot element."""
        plot_id = self._get_unique_id("plot")
        
        # Check if we have a pre-generated plot
        if element.plot_key in self.plots:
            plot_json = self.plots[element.plot_key]
            if isinstance(plot_json, str):
                json_str = plot_json
            else:
                # It's a Plotly figure, convert to JSON
                json_str = plot_json.to_json() if hasattr(plot_json, 'to_json') else json.dumps(plot_json)
            
            return f'''
            <div class="plot-container">
                {f'<div class="plot-title"><i class="bi bi-graph-up me-2"></i>{element.title}</div>' if element.title else ''}
                <div id="{plot_id}_chart" style="height: {element.height}px;"></div>
                <script>
                    (function() {{
                        const plotData = {json_str};
                        const layout = Object.assign({{}}, plotData.layout, getPlotlyLayout());
                        Plotly.newPlot('{plot_id}_chart', plotData.data, layout, {{responsive: true}});
                    }})();
                </script>
            </div>
            '''
        
        # Generate interactive plot
        plot_html = self._generate_interactive_plot(element.plot_key, plot_id, element.height)
        if plot_html:
            return plot_html
        
        return f'''
        <div class="plot-container">
            <div class="p-5 text-center">
                <i class="bi bi-bar-chart-fill text-secondary" style="font-size: 3rem;"></i>
                <p class="text-secondary mt-3 mb-0">
                    Interaktiver Plot: <strong>{element.plot_key}</strong>
                </p>
                <small class="text-muted">Verfügbar in der Streamlit-Version</small>
            </div>
        </div>
        '''
    
    def _generate_interactive_plot(self, key: str, plot_id: str, height: int) -> Optional[str]:
        """Generate HTML/JS for interactive plots."""
        # For Flask, we generate the plot data server-side and embed it
        x = self.data.get('x', [])
        y = self.data.get('y', [])
        
        if key == "raw_scatter" and len(x) > 0:
            return self._generate_scatter_plot(plot_id, x, y, height)
        
        elif key == "ols_regression" and len(x) > 0:
            return self._generate_regression_plot(plot_id, x, y, height)
        
        elif key == "variance_decomposition":
            return self._generate_variance_decomposition_plot(plot_id, height)
        
        elif key == "correlation_examples":
            return self._generate_correlation_examples_plot(plot_id, height)
        
        return None
    
    def _generate_scatter_plot(self, plot_id: str, x: list, y: list, height: int) -> str:
        """Generate scatter plot HTML (3D)."""
        x_list = list(x) if hasattr(x, 'tolist') else x
        y_list = list(y) if hasattr(y, 'tolist') else y
        z_zeros = [0] * len(x_list)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        
        trace_data = {
            "data": [
                {
                    "x": x_list,
                    "y": y_list,
                    "z": z_zeros,
                    "mode": "markers",
                    "type": "scatter3d",
                    "marker": {"size": 5, "color": "#3498db", "opacity": 0.7},
                    "name": "Datenpunkte"
                },
                {
                    "x": [x_mean],
                    "y": [y_mean],
                    "z": [0],
                    "mode": "markers",
                    "type": "scatter3d",
                    "marker": {"size": 8, "color": "red", "symbol": "cross"},
                    "name": f"Schwerpunkt ({x_mean:.2f}, {y_mean:.2f})"
                }
            ],
            "layout": {
                "title": "Schritt 1: Visualisierung der Rohdaten (3D)",
                "scene": {
                    "xaxis": {"title": self.data.get('x_label', 'X')},
                    "yaxis": {"title": self.data.get('y_label', 'Y')},
                    "zaxis": {"title": "Z"}
                },
                "template": "plotly_white",
            }
        }
        
        return f'''
        <div class="plot-container">
            <div id="{plot_id}_chart" style="height: {height}px;"></div>
            <script>
                Plotly.newPlot('{plot_id}_chart', {json.dumps(trace_data["data"])}, {json.dumps(trace_data["layout"])});
            </script>
        </div>
        '''
    
    def _generate_regression_plot(self, plot_id: str, x: list, y: list, height: int) -> str:
        """Generate regression plot HTML (3D)."""
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        slope = self.stats_data.get('slope', 0)
        intercept = self.stats_data.get('intercept', 0)
        
        x_line = np.linspace(min(x_arr), max(x_arr), 100).tolist()
        y_line = [intercept + slope * xi for xi in x_line]
        z_line = [0] * len(x_line)
        
        trace_data = {
            "data": [
                {
                    "x": x_arr.tolist(),
                    "y": y_arr.tolist(),
                    "z": [0] * len(x_arr),
                    "mode": "markers",
                    "type": "scatter3d",
                    "marker": {"size": 5, "color": "#3498db"},
                    "name": "Datenpunkte"
                },
                {
                    "x": x_line,
                    "y": y_line,
                    "z": z_line,
                    "mode": "lines",
                    "type": "scatter3d",
                    "line": {"color": "red", "width": 5},
                    "name": "Regressionsgerade"
                }
            ],
            "layout": {
                "title": "OLS Regression (3D)",
                "scene": {
                    "xaxis": {"title": self.data.get('x_label', 'X')},
                    "yaxis": {"title": self.data.get('y_label', 'Y')},
                    "zaxis": {"title": "Z"}
                },
                "template": "plotly_white"
            }
        }
        
        return f'''
        <div class="plot-container">
            <div id="{plot_id}_chart" style="height: {height}px;"></div>
            <script>
                Plotly.newPlot('{plot_id}_chart', {json.dumps(trace_data["data"])}, {json.dumps(trace_data["layout"])});
            </script>
        </div>
        '''
    
    def _generate_variance_decomposition_plot(self, plot_id: str, height: int) -> str:
        """Generate variance decomposition bar chart (3D)."""
        sst = self.stats_data.get('sst', 100)
        ssr = self.stats_data.get('ssr', 60)
        sse = self.stats_data.get('sse', 40)
        r2 = self.stats_data.get('r_squared', 0.6)
        
        # Simulate bars with lines in 3D
        names = ["SST (Total)", "SSR (Erklärt)", "SSE (Unerklärt)"]
        values = [sst, ssr, sse]
        colors = ["gray", "#2ecc71", "#e74c3c"]
        
        traces = []
        for i, (name, val, col) in enumerate(zip(names, values, colors)):
            traces.append({
                "x": [i, i],
                "y": [0, 0],
                "z": [0, val],
                "mode": "lines",
                "type": "scatter3d",
                "line": {"color": col, "width": 15},
                "name": name
            })
            # Top marker
            traces.append({
                "x": [i],
                "y": [0],
                "z": [val],
                "mode": "markers+text",
                "type": "scatter3d",
                "marker": {"size": 5, "color": col},
                "text": [f"{val:.1f}"],
                "textposition": "top center",
                "showlegend": False
            })
        
        trace_data = {
            "data": traces,
            "layout": {
                "title": f"Varianzzerlegung: R² = {r2:.4f} (3D)",
                "scene": {
                    "xaxis": {"title": "", "ticktext": names, "tickvals": [0, 1, 2]},
                    "yaxis": {"title": ""},
                    "zaxis": {"title": "Quadratsumme"}
                },
                "template": "plotly_white"
            }
        }
        
        return f'''
        <div class="plot-container">
            <div id="{plot_id}_chart" style="height: {height}px;"></div>
            <script>
                Plotly.newPlot('{plot_id}_chart', {json.dumps(trace_data["data"])}, {json.dumps(trace_data["layout"])});
            </script>
        </div>
        '''
    
    def _generate_correlation_examples_plot(self, plot_id: str, height: int) -> str:
        """Generate correlation examples subplot (3D)."""
        np.random.seed(42)
        n = 50
        correlations = [-0.95, -0.50, 0.0, 0.50, 0.95]
        
        traces = []
        # Plotly subplots in JSON are complex. 
        # For simplicity, we'll just create independent 3D plots if we want, or try to use subplots.
        # But `grid` layout for 3D scenes is supported in Plotly.js.
        # However, it's easier to just assume they are 3D scatter plots in a grid.
        
        # NOTE: Plotly.js 3D subplots need 'scene', 'scene2', etc.
        
        for i, rho in enumerate(correlations):
            row = i // 3
            col = i % 3
            
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            data_gen = np.random.multivariate_normal(mean, cov, n)
            
            scene_id = f"scene{i+1}" if i > 0 else "scene"
            
            traces.append({
                "x": data_gen[:, 0].tolist(),
                "y": data_gen[:, 1].tolist(),
                "z": [0] * n,
                "mode": "markers",
                "type": "scatter3d",
                "marker": {"size": 3},
                "name": f"r = {rho}",
                "scene": scene_id
            })

        # Nonlinear
        x_nl = np.linspace(-2, 2, n)
        y_nl = x_nl**2 + np.random.normal(0, 0.3, n)
        traces.append({
            "x": x_nl.tolist(),
            "y": y_nl.tolist(),
            "z": [0] * n,
            "mode": "markers",
            "type": "scatter3d",
            "marker": {"size": 3},
            "name": "r = 0 (nonlinear)",
            "scene": "scene6"
        })
        
        layout = {
            "title": "Korrelations-Beispiele (3D View)",
            "height": height,
            "showlegend": False,
            # We need to define scenes and their positions
            # This is tricky without 'make_subplots' logic in Python, but we can approximation
            # or just rely on 'grid' if Plotly JS supports it for scenes easily.
            # Actually, let's define the domains for each scene.
            "scene":  {"domain": {"x": [0, 0.33], "y": [0.5, 1]}, "annotations": [{"text": "r=-0.95"}]},
            "scene2": {"domain": {"x": [0.33, 0.66], "y": [0.5, 1]}, "annotations": [{"text": "r=-0.50"}]},
            "scene3": {"domain": {"x": [0.66, 1], "y": [0.5, 1]}, "annotations": [{"text": "r=0.00"}]},
            "scene4": {"domain": {"x": [0, 0.33], "y": [0, 0.5]}, "annotations": [{"text": "r=0.50"}]},
            "scene5": {"domain": {"x": [0.33, 0.66], "y": [0, 0.5]}, "annotations": [{"text": "r=0.95"}]},
            "scene6": {"domain": {"x": [0.66, 1], "y": [0, 0.5]}, "annotations": [{"text": "nonlinear"}]},
        }
        
        return f'''
        <div class="plot-container">
            <div id="{plot_id}_chart" style="height: {height}px;"></div>
            <script>
                Plotly.newPlot('{plot_id}_chart', {json.dumps(traces)}, {json.dumps(layout)});
            </script>
        </div>
        '''
    
    def _render_table(self, element: Table) -> str:
        """Render a table element."""
        header_html = ''.join(f'<th scope="col">{h}</th>' for h in element.headers)
        rows_html = '\n'.join(
            '<tr>' + ''.join(f'<td><code>{cell}</code></td>' if i > 0 and self._is_numeric(cell) else f'<td>{cell}</td>' 
                            for i, cell in enumerate(row)) + '</tr>'
            for row in element.rows
        )
        
        caption_html = f'<caption class="caption-top">{element.caption}</caption>' if element.caption else ''
        
        return f'''
        <div class="table-responsive">
            <table class="table table-hover">
                {caption_html}
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        '''
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string value looks numeric."""
        try:
            float(value.replace(',', '.'))
            return True
        except (ValueError, AttributeError):
            return False
    
    def _render_columns(self, element: Columns) -> str:
        """Render multi-column layout."""
        total_width = sum(element.widths)
        col_widths = [int(12 * w / total_width) for w in element.widths]
        
        cols_html = []
        for col_width, content in zip(col_widths, element.columns):
            content_html = '\n'.join(self._render_element(item) for item in content)
            cols_html.append(f'<div class="col-md-{col_width}">{content_html}</div>')
        
        return f'<div class="row">{" ".join(cols_html)}</div>'
    
    def _render_expander(self, element: Expander) -> str:
        """Render expandable section."""
        expander_id = self._get_unique_id("expander")
        content_html = '\n'.join(self._render_element(item) for item in element.content)
        expanded_class = "show" if element.expanded else ""
        collapsed_class = "" if element.expanded else "collapsed"
        
        return f'''
        <div class="accordion-item mb-3">
            <h2 class="accordion-header" id="heading-{expander_id}">
                <button class="accordion-button {collapsed_class}" 
                        type="button" 
                        data-bs-toggle="collapse" 
                        data-bs-target="#collapse-{expander_id}"
                        aria-expanded="{"true" if element.expanded else "false"}"
                        aria-controls="collapse-{expander_id}">
                    {element.title}
                </button>
            </h2>
            <div id="collapse-{expander_id}" 
                 class="accordion-collapse collapse {expanded_class}"
                 aria-labelledby="heading-{expander_id}">
                <div class="accordion-body">
                    {content_html}
                </div>
            </div>
        </div>
        '''
    
    def _render_info_box(self, element: InfoBox) -> str:
        """Render information box."""
        return f'''
        <div class="alert alert-info d-flex align-items-start">
            <i class="bi bi-info-circle-fill me-3 mt-1"></i>
            <div>{self._simple_md_to_html(element.content)}</div>
        </div>
        '''
    
    def _render_warning_box(self, element: WarningBox) -> str:
        """Render warning box."""
        return f'''
        <div class="alert alert-warning d-flex align-items-start">
            <i class="bi bi-exclamation-triangle-fill me-3 mt-1"></i>
            <div>{self._simple_md_to_html(element.content)}</div>
        </div>
        '''
    
    def _render_success_box(self, element: SuccessBox) -> str:
        """Render success box."""
        return f'''
        <div class="alert alert-success d-flex align-items-start">
            <i class="bi bi-check-circle-fill me-3 mt-1"></i>
            <div>{self._simple_md_to_html(element.content)}</div>
        </div>
        '''
    
    def _render_code_block(self, element: CodeBlock) -> str:
        """Render code block."""
        # Escape HTML in code
        import html
        escaped_code = html.escape(element.code)
        return f'''
        <div class="position-relative">
            <pre class="mb-0"><code class="language-{element.language}">{escaped_code}</code></pre>
            <button class="btn btn-sm btn-outline-secondary position-absolute top-0 end-0 m-2" 
                    onclick="navigator.clipboard.writeText(this.parentElement.querySelector('code').textContent)"
                    title="In Zwischenablage kopieren">
                <i class="bi bi-clipboard"></i>
            </button>
        </div>
        '''
    
    def _render_section(self, element: Section) -> str:
        """Render a section."""
        content_html = '\n'.join(self._render_element(item) for item in element.content)
        return f'''
        <div class="section">
            <h4>{element.icon} {element.title}</h4>
            {content_html}
        </div>
        '''
    
    def render_to_dict(self, content: EducationalContent) -> Dict[str, Any]:
        """
        Render content structure to a dictionary for template use.
        
        This is useful when you want to pass the content to a Jinja2 template
        that does the final HTML rendering.
        """
        return {
            "title": content.title,
            "subtitle": content.subtitle,
            "chapters": [
                {
                    "number": ch.number,
                    "title": ch.title,
                    "icon": ch.icon,
                    "html": self._render_chapter(ch),
                }
                for ch in content.chapters
            ],
            "full_html": self.render(content),
        }
