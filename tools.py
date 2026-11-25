"""Tools for the Research Agent: Web Search and PDF Vector Store Search.

Provides:
- search_web: Query live web via DuckDuckGo (ddgs).
- search_pdf: Query local indexed PDF vector store.
"""

import time
import logging
from typing import Any

# Configure logger
logger = logging.getLogger(__name__)

# Support both ddgs and duckduckgo_search packages
try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException
except ImportError:
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RatelimitException
    except ImportError:
        DDGS = None
        RatelimitException = Exception

SNIPPET_MAX_CHARS = 250


def search_web(query: str, max_results: int = 3) -> list[dict]:
    """Search the live web using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to return.

    Returns:
        List of dicts with keys: 'title', 'body', 'url'.
    """
    query_str = query.strip()
    if not query_str:
        return [{"title": "No query", "body": "Empty search query provided.", "url": ""}]

    if DDGS is None:
        return [{
            "title": "Error",
            "body": "DuckDuckGo search library (ddgs) is not installed.",
            "url": ""
        }]

    results = []
    for attempt in range(2):
        try:
            ddgs = DDGS(timeout=10)
            raw_results = list(ddgs.text(query_str, max_results=max_results))
            for item in raw_results:
                title = (item.get("title") or "").strip()
                body = (item.get("body") or "").strip().replace("\n", " ")
                if len(body) > SNIPPET_MAX_CHARS:
                    body = body[:SNIPPET_MAX_CHARS] + "..."
                url = item.get("href") or item.get("link") or ""
                results.append({
                    "title": title,
                    "body": body,
                    "url": url,
                })
            break
        except RatelimitException:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return [{
                "title": "Rate Limited",
                "body": "DuckDuckGo search rate limit reached. Proceed with local document context.",
                "url": ""
            }]
        except Exception as exc:
            logger.warning(f"Web search error: {exc}")
            if attempt == 1:
                return [{
                    "title": "Search Notice",
                    "body": f"Web search could not retrieve live results for '{query_str}'.",
                    "url": ""
                }]
            time.sleep(0.5)

    if not results:
        return [{
            "title": "No results",
            "body": f"No live web results found for query: '{query_str}'.",
            "url": ""
        }]

    return results


def search_pdf(vectorstore: Any, query: str, max_results: int = 3) -> list[dict]:
    """Search indexed PDF documents from a vector store.

    Args:
        vectorstore: Vector store object or ChromaDB collection.
        query: Search query string.
        max_results: Max matching chunks to return.

    Returns:
        List of dicts with keys: 'text', 'page', 'source', 'score'.
    """
    query_str = query.strip()
    if not query_str or vectorstore is None:
        return [{
            "text": f"No document results available for query: '{query_str}'.",
            "page": 0,
            "source": "PDF Store",
            "score": 0.0,
        }]

    hits: list[dict] = []

    # 1. Custom VectorStore / DocumentIndex
    if hasattr(vectorstore, "similarity_search"):
        try:
            hits = vectorstore.similarity_search(query_str, k=max_results)
            for hit in hits:
                if len(hit.get("text", "")) > SNIPPET_MAX_CHARS:
                    hit["text"] = hit["text"][:SNIPPET_MAX_CHARS] + "..."
        except Exception as exc:
            logger.warning(f"Error querying custom vectorstore: {exc}")

    # 2. ChromaDB Collection
    elif hasattr(vectorstore, "query"):
        try:
            res = vectorstore.query(
                query_texts=[query_str],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            docs = res.get("documents", [[]])[0] if res.get("documents") else []
            metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
            dists = res.get("distances", [[]])[0] if res.get("distances") else []

            for i, text in enumerate(docs):
                meta = metas[i] if i < len(metas) and metas[i] else {}
                dist = dists[i] if i < len(dists) else 0.0
                snippet = text.strip().replace("\n", " ")
                if len(snippet) > SNIPPET_MAX_CHARS:
                    snippet = snippet[:SNIPPET_MAX_CHARS] + "..."
                hits.append({
                    "text": snippet,
                    "page": meta.get("page", 1),
                    "source": meta.get("source", "PDF Document"),
                    "score": round(1.0 - min(dist, 1.0), 3) if dist is not None else 1.0,
                })
        except Exception as exc:
            logger.warning(f"Error querying ChromaDB collection: {exc}")

    # 3. Simple list of chunk dicts fallback
    elif isinstance(vectorstore, list):
        terms = set(query_str.lower().split())
        scored = []
        for chunk in vectorstore:
            text = chunk.get("text", "")
            matches = sum(1 for term in terms if term in text.lower())
            if matches > 0:
                scored.append((matches, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        res = [c for _, c in scored[:max_results]]
        for r in res:
            if len(r.get("text", "")) > SNIPPET_MAX_CHARS:
                r["text"] = r["text"][:SNIPPET_MAX_CHARS] + "..."
        hits = res

    if not hits:
        return [{
            "text": f"No matching sections found in PDF documents for query: '{query_str}'.",
            "page": 0,
            "source": "PDF Store",
            "score": 0.0,
        }]

    return hits


if __name__ == "__main__":
    print("Testing Web Search:")
    web_res = search_web("AI agents", max_results=2)
    for r in web_res:
        print(f"- {r['title']} ({r['url']})\n  {r['body']}\n")
