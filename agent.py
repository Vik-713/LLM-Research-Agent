"""Autonomous Research Agent powered by Pydantic AI and Groq.

Combines local document retrieval (PDF RAG) with live web search to produce
comprehensive, grounded, structured research briefs.
"""

import os
import sys
import time
import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.exceptions import ModelRateLimitError, ModelHTTPError

# Import tools and indexer
from tools import search_web as web_search_tool, search_pdf as pdf_search_tool
from indexer import load_index, build_index, create_sample_pdf, VectorStore

# Load environment variables
load_dotenv()

# Default Groq model and max tokens
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "850"))


@dataclass
class AgentDeps:
    """Dependencies injected into agent runs and tools."""
    vectorstore: object
    max_pdf_results: int = 2
    max_web_results: int = 2


class Finding(BaseModel):
    """A single factual claim or insight found during research."""
    claim: str = Field(description="Concrete fact or claim discovered.")
    source_type: Literal["pdf", "web"] = Field(
        description="Whether this fact came from indexed PDF documents ('pdf') or web search ('web')."
    )
    source: str = Field(
        description="Specific reference citation: document page or web URL / query."
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )


class Research(BaseModel):
    """Comprehensive structured research report synthesized by the agent."""
    topic: str = Field(description="The research topic or question explored.")
    summary: str = Field(
        description="Executive summary synthesizing findings from both local PDF documents and web search."
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="List of concrete findings with source type and citation."
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="3-5 bullet points summarizing the most important takeaways."
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
        description="All unique references cited across findings."
    )

    def format_markdown(self) -> str:
        """Format the research report as clean Markdown."""
        lines = [
            f"# Research Report: {self.topic}",
            "",
            "## Executive Summary",
            self.summary,
            "",
            "## Key Takeaways",
        ]
        for t in self.key_takeaways:
            lines.append(f"- {t}")

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


RESEARCH_SYSTEM_PROMPT = """You are a concise, grounded Research Agent.
Capabilities:
1. `search_pdf`: Query local PDF knowledge base for internal facts and benchmarks.
2. `search_web`: Query live web for current updates and news.

Guidelines:
- Keep tool calls minimal (at most 1-2 tool calls total).
- Once you retrieve facts, synthesize them into the structured Research brief.
- Keep findings concise (1 sentence per finding) to respect token budgets.
- Accurately tag each finding with source_type ('pdf' or 'web').
"""


async def search_pdf(ctx: RunContext[AgentDeps], query: str) -> list[dict]:
    """Search indexed PDF documents for relevant excerpts, facts, domain data, and citations.

    Input: a 3-8 word query describing the concept or data needed from local PDF docs.
    Returns matching document chunks with page numbers and relevance scores.
    """
    return pdf_search_tool(
        vectorstore=ctx.deps.vectorstore,
        query=query,
        max_results=ctx.deps.max_pdf_results,
    )


async def search_web(ctx: RunContext[AgentDeps], query: str) -> list[dict]:
    """Search the live web for recent news, industry statistics, and external information.

    Input: a 3-8 word search query.
    Returns web search results with titles, snippets, and source URLs.
    """
    return web_search_tool(
        query=query,
        max_results=ctx.deps.max_web_results,
    )


def build_agent(
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Agent[AgentDeps, Research]:
    """Build and return the configured Research Agent."""
    model = model_name or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    tokens_limit = max_tokens or DEFAULT_MAX_TOKENS
    groq_model = GroqModel(model)

    agent: Agent[AgentDeps, Research] = Agent(
        groq_model,
        deps_type=AgentDeps,
        output_type=Research,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        model_settings={"max_tokens": tokens_limit},
        retries=2,
    )

    # Register tools
    agent.tool(search_pdf)
    agent.tool(search_web)

    return agent


async def run_research(
    topic: str,
    deps: AgentDeps,
    agent: Optional[Agent[AgentDeps, Research]] = None
) -> Research:
    """Execute research asynchronously on the given topic with automatic rate-limit backoff."""
    agent_instance = agent or build_agent()
    prompt = f"Conduct concise, grounded research on: {topic}"

    for attempt in range(3):
        try:
            result = await agent_instance.run(prompt, deps=deps)
            brief = result.output
            brief.topic = topic
            return brief
        except (ModelRateLimitError, ModelHTTPError) as exc:
            if attempt < 2 and "rate_limit" in str(exc).lower():
                wait_seconds = 7 * (attempt + 1)
                print(f"\n[Rate Limit] Groq OTPM quota active. Waiting {wait_seconds}s for token reset...")
                await asyncio.sleep(wait_seconds)
            else:
                raise exc

    raise RuntimeError("Research failed after retries.")


def run_research_sync(
    topic: str,
    deps: AgentDeps,
    agent: Optional[Agent[AgentDeps, Research]] = None
) -> Research:
    """Execute research synchronously on the given topic with automatic rate-limit backoff."""
    agent_instance = agent or build_agent()
    prompt = f"Conduct concise, grounded research on: {topic}"

    for attempt in range(3):
        try:
            result = agent_instance.run_sync(prompt, deps=deps)
            brief = result.output
            brief.topic = topic
            return brief
        except (ModelRateLimitError, ModelHTTPError) as exc:
            if attempt < 2 and "rate_limit" in str(exc).lower():
                wait_seconds = 7 * (attempt + 1)
                print(f"\n[Rate Limit] Groq OTPM quota active. Waiting {wait_seconds}s for token reset...")
                time.sleep(wait_seconds)
            else:
                raise exc

    raise RuntimeError("Research failed after retries.")


def main():
    """Main CLI entrypoint."""
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "State of Autonomous AI Agents in 2026"

    index_dir = "./doc_index"
    pdf_path = "./sample_research.pdf"

    # Ensure vector store index is available
    if os.path.exists(os.path.join(index_dir, "index.json")):
        print(f"Loading existing document index from {index_dir} ...")
        vectorstore = load_index(index_dir)
    else:
        if not os.path.exists(pdf_path):
            print("No PDF document found. Generating sample PDF...")
            create_sample_pdf(pdf_path)
        print(f"Indexing PDF document: {pdf_path} ...")
        vectorstore = build_index(pdf_path, output_dir=index_dir)

    deps = AgentDeps(
        vectorstore=vectorstore,
        max_pdf_results=2,
        max_web_results=2,
    )

    print("=" * 70)
    print(f"RESEARCH TOPIC: {topic}")
    print("=" * 70)
    print("Running Research Agent (PDF Search + Web Search)...\n")

    report = run_research_sync(topic, deps=deps)

    print("=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)
    print(report.format_markdown())


if __name__ == "__main__":
    main()