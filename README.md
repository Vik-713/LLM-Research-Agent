# Autonomous Research Agent (Pydantic AI + Groq)

A dual-retrieval AI research agent that combines **local document retrieval (PDF RAG)** and **live web search (DuckDuckGo)** to generate grounded, structured research briefs.

## Architecture

```
User Query / Research Topic
            │
            ▼
┌──────────────────────────────────────────────┐
│        Pydantic AI Agent (Groq)              │
│                                              │
│   ┌──────────────────┐ ┌──────────────────┐  │
│   │   search_pdf     │ │   search_web     │  │
│   │ (Vector Store)   │ │  (DuckDuckGo)    │  │
│   └────────┬─────────┘ └────────┬─────────┘  │
└────────────┼────────────────────┼────────────┘
             ▼                    ▼
     Local PDF Chunks        Live Web News
             │                    │
             └──────────┬─────────┘
                        ▼
            Typed Research Schema
            (Pydantic Models)
```

## Files

| File | Purpose |
|---|---|
| `agent.py` | Core Agent definition, dependency injection (`AgentDeps`), tool registration, and execution loop. |
| `indexer.py` | PDF ingestion, chunking with overlap, pure-Python TF-IDF `VectorStore`, index persistence, and sample PDF generator. |
| `models.py` | Pydantic models (`Finding`, `Research`) defining the structured agent output. |
| `tools.py` | Search tools: `search_web` (via `ddgs`) and `search_pdf` (via vector store). |
| `.env` | Environment configuration (`GROQ_API_KEY`, `GROQ_MODEL`). |
| `requirements.txt` | Package dependencies. |

## Quick Start

1. **Configure Environment**:
   Ensure `.env` contains your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=qwen/qwen3.8-27b
   ```

2. **Index a Document**:
   ```bash
   python indexer.py [path_to_pdf]
   ```
   *(If no path is specified, it automatically creates and indexes `sample_research.pdf`)*

3. **Run the Research Agent**:
   ```bash
   python agent.py "State of Autonomous AI Agents in 2026"
   ```
