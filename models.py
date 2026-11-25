"""Data models for Research Agent.

Defines structured schemas for findings, sources, and research output reports.
"""

from typing import Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A concrete claim or finding discovered from PDF documents or web search."""
    claim: str = Field(
        ...,
        description="A specific, factual claim or insight learned during research."
    )
    source_type: Literal["pdf", "web"] = Field(
        ...,
        description="Whether this finding was derived from indexed PDF documents ('pdf') or live web search ('web')."
    )
    source: str = Field(
        ...,
        description="Specific citation: file name/page number for PDF or URL/query title for web."
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Estimated confidence score between 0.0 and 1.0."
    )


class Research(BaseModel):
    """Comprehensive research report synthesized by the research agent."""
    topic: str = Field(
        ...,
        description="The research topic or question explored."
    )
    summary: str = Field(
        ...,
        description="Executive summary synthesizing knowledge from both local documents and web search."
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="List of concrete findings, each tied to its source type and reference."
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="3-5 high-impact bullet points summarizing the most important takeaways."
    )
    numbers_and_dates: list[str] = Field(
        default_factory=list,
        description="Notable statistics, metrics, dates, and figures found during research."
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Open questions or topics that could not be fully resolved."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="All unique sources cited across findings."
    )

    def format_markdown(self) -> str:
        """Format research report as rich Markdown."""
        lines = [
            f"# Research Report: {self.topic}",
            "",
            "## Executive Summary",
            self.summary,
            "",
            "## Key Takeaways",
        ]
        for takeaway in self.key_takeaways:
            lines.append(f"- {takeaway}")

        lines.extend(["", "## Detailed Findings"])
        for f in self.findings:
            badge = "[PDF]" if f.source_type == "pdf" else "[WEB]"
            lines.append(f"- {badge} **{f.claim}**")
            lines.append(f"  - Source: `{f.source}` (confidence: {f.confidence:.2f})")

        if self.numbers_and_dates:
            lines.extend(["", "## Key Numbers & Dates"])
            for item in self.numbers_and_dates:
                lines.append(f"- {item}")

        if self.open_questions:
            lines.extend(["", "## Open Questions"])
            for q in self.open_questions:
                lines.append(f"- {q}")

        if self.sources:
            lines.extend(["", "## Sources Consulted"])
            for s in self.sources:
                lines.append(f"- {s}")

        return "\n".join(lines)
