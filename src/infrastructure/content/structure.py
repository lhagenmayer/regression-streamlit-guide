"""
Content Structure - Data classes for educational content.

These are pure DATA STRUCTURES with no UI dependencies.
They describe WHAT to render, not HOW to render it.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class ElementType(Enum):
    """Types of content elements."""
    MARKDOWN = "markdown"
    METRIC = "metric"
    METRIC_ROW = "metric_row"
    FORMULA = "formula"
    PLOT = "plot"
    TABLE = "table"
    COLUMNS = "columns"
    EXPANDER = "expander"
    INFO_BOX = "info_box"
    WARNING_BOX = "warning_box"
    SUCCESS_BOX = "success_box"
    CODE_BLOCK = "code_block"
    DIVIDER = "divider"
    CHAPTER = "chapter"
    SECTION = "section"


@dataclass
class ContentElement:
    """Base class for all content elements."""
    element_type: ElementType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"type": self.element_type.value}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ContentElement":
        """Factory method to create element from dictionary."""
        etype = data.get("type", "").lower()
        content = data.get("content")
        
        if etype == "markdown": return Markdown(data.get("text", ""))
        if etype == "metric": return Metric(data.get("label"), data.get("value"), data.get("help_text"), data.get("delta"))
        if etype == "metric_row": return MetricRow([ContentElement.from_dict(m) for m in data.get("metrics", [])])
        if etype == "formula": return Formula(data.get("latex"), data.get("inline", False))
        if etype == "plot": return Plot(data.get("plot_key"), data.get("title", ""), data.get("description", ""), data.get("height", 400))
        if etype == "table": return Table(data.get("headers"), data.get("rows"), data.get("caption", ""))
        if etype == "columns": return Columns([ [ContentElement.from_dict(e) for e in col] for col in data.get("columns", []) ], data.get("widths"))
        if etype == "expander": return Expander(data.get("title"), [ContentElement.from_dict(e) for e in content], data.get("expanded", False))
        if etype == "info_box": return InfoBox(content)
        if etype == "warning_box": return WarningBox(content)
        if etype == "success_box": return SuccessBox(content)
        if etype == "code_block": return CodeBlock(data.get("code"), data.get("language", "python"))
        if etype == "divider": return Divider()
        if etype == "section": return Section.from_dict(data)
        
        # Fallback or unknown
        return Markdown(f"**Unknown Element:** {etype}")


@dataclass
class Markdown(ContentElement):
    """Markdown text content."""
    text: str
    
    def __init__(self, text: str):
        super().__init__(ElementType.MARKDOWN)
        self.text = text
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "markdown", "text": self.text}


@dataclass
class Metric(ContentElement):
    """Single metric display."""
    label: str
    value: str
    help_text: str = ""
    delta: str = ""
    
    def __init__(self, label: str, value: str, help_text: str = "", delta: str = ""):
        super().__init__(ElementType.METRIC)
        self.label = label
        self.value = value
        self.help_text = help_text
        self.delta = delta
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "metric",
            "label": self.label,
            "value": self.value,
            "help_text": self.help_text,
            "delta": self.delta,
        }


@dataclass
class MetricRow(ContentElement):
    """Row of metrics."""
    metrics: List[Metric]
    
    def __init__(self, metrics: List[Metric]):
        super().__init__(ElementType.METRIC_ROW)
        self.metrics = metrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "metric_row",
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass
class Formula(ContentElement):
    """LaTeX formula."""
    latex: str
    inline: bool = False
    
    def __init__(self, latex: str, inline: bool = False):
        super().__init__(ElementType.FORMULA)
        self.latex = latex
        self.inline = inline
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "formula", "latex": self.latex, "inline": self.inline}


@dataclass
class Plot(ContentElement):
    """Plotly figure reference."""
    plot_key: str  # Key to lookup in plots dictionary
    title: str = ""
    description: str = ""
    height: int = 400
    
    def __init__(self, plot_key: str, title: str = "", description: str = "", height: int = 400):
        super().__init__(ElementType.PLOT)
        self.plot_key = plot_key
        self.title = title
        self.description = description
        self.height = height
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "plot",
            "plot_key": self.plot_key,
            "title": self.title,
            "description": self.description,
            "height": self.height,
        }


@dataclass
class Table(ContentElement):
    """Data table."""
    headers: List[str]
    rows: List[List[str]]
    caption: str = ""
    
    def __init__(self, headers: List[str], rows: List[List[str]], caption: str = ""):
        super().__init__(ElementType.TABLE)
        self.headers = headers
        self.rows = rows
        self.caption = caption
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "table",
            "headers": self.headers,
            "rows": self.rows,
            "caption": self.caption,
        }


@dataclass
class Columns(ContentElement):
    """Multi-column layout."""
    columns: List[List["ContentElement"]]
    widths: List[float] = None  # Relative widths
    
    def __init__(self, columns: List[List["ContentElement"]], widths: List[float] = None):
        super().__init__(ElementType.COLUMNS)
        self.columns = columns
        self.widths = widths or [1.0] * len(columns)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "columns",
            "columns": [[e.to_dict() for e in col] for col in self.columns],
            "widths": self.widths,
        }


@dataclass
class Expander(ContentElement):
    """Expandable section."""
    title: str
    content: List["ContentElement"]
    expanded: bool = False
    
    def __init__(self, title: str, content: List["ContentElement"], expanded: bool = False):
        super().__init__(ElementType.EXPANDER)
        self.title = title
        self.content = content
        self.expanded = expanded
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "expander",
            "title": self.title,
            "content": [e.to_dict() for e in self.content],
            "expanded": self.expanded,
        }


@dataclass
class InfoBox(ContentElement):
    """Information box."""
    content: str
    
    def __init__(self, content: str):
        super().__init__(ElementType.INFO_BOX)
        self.content = content
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "info_box", "content": self.content}


@dataclass
class WarningBox(ContentElement):
    """Warning box."""
    content: str
    
    def __init__(self, content: str):
        super().__init__(ElementType.WARNING_BOX)
        self.content = content
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "warning_box", "content": self.content}


@dataclass
class SuccessBox(ContentElement):
    """Success box."""
    content: str
    
    def __init__(self, content: str):
        super().__init__(ElementType.SUCCESS_BOX)
        self.content = content
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "success_box", "content": self.content}


@dataclass
class CodeBlock(ContentElement):
    """Code block."""
    code: str
    language: str = "python"
    
    def __init__(self, code: str, language: str = "python"):
        super().__init__(ElementType.CODE_BLOCK)
        self.code = code
        self.language = language
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "code_block", "code": self.code, "language": self.language}


@dataclass
class Divider(ContentElement):
    """Horizontal divider."""
    
    def __init__(self):
        super().__init__(ElementType.DIVIDER)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "divider"}


@dataclass
class Section(ContentElement):
    """Section within a chapter."""
    title: str
    icon: str
    content: List[ContentElement]
    
    def __init__(self, title: str, icon: str, content: List[ContentElement]):
        super().__init__(ElementType.SECTION)
        self.title = title
        self.icon = icon
        self.content = content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "section",
            "title": self.title,
            "icon": self.icon,
            "content": [e.to_dict() for e in self.content],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Section":
        return Section(
            title=data.get("title", ""),
            icon=data.get("icon", ""),
            content=[ContentElement.from_dict(e) for e in data.get("content", [])]
        )


@dataclass
class Chapter(ContentElement):
    """Main chapter with sections."""
    number: str
    title: str
    icon: str
    sections: List[Union[Section, ContentElement]]
    
    def __init__(self, number: str, title: str, icon: str, sections: List[Union[Section, ContentElement]]):
        super().__init__(ElementType.CHAPTER)
        self.number = number
        self.title = title
        self.icon = icon
        self.sections = sections
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "chapter",
            "number": self.number,
            "title": self.title,
            "icon": self.icon,
            "sections": [s.to_dict() for s in self.sections],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Chapter":
        # Sections can be Section or other elements (rarely)
        sections = []
        for sec_data in data.get("sections", []):
            if sec_data.get("type") == "section":
                sections.append(Section.from_dict(sec_data))
            else:
                sections.append(ContentElement.from_dict(sec_data))
        
        return Chapter(
            number=data.get("number", ""),
            title=data.get("title", ""),
            icon=data.get("icon", ""),
            sections=sections
        )


@dataclass
class EducationalContent:
    """Complete educational content for an analysis."""
    title: str
    subtitle: str
    chapters: List[Chapter]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "chapters": [c.to_dict() for c in self.chapters],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EducationalContent":
        return EducationalContent(
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            chapters=[Chapter.from_dict(c) for c in data.get("chapters", [])]
        )
