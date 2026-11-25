"""PDF and Document Indexer with Vector Store capabilities.

Provides:
- load_and_chunk_pdf: Extract text from PDF documents and create overlapping chunks.
- VectorStore: Lightweight, pure-Python vector index with TF-IDF cosine similarity and persistence.
- build_index / load_index: Convenient helpers to build and reload document indices.
- create_sample_pdf: Generates a realistic sample research document if no PDF is provided.
"""

import os
import re
import json
import math
import logging
from typing import Any
from collections import Counter
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# PDF reader
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def extract_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract page number and text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if PdfReader is None:
        raise ImportError("pypdf is required to read PDF files. Please install pypdf.")

    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = text.strip()
        if cleaned:
            pages.append((idx, cleaned))
    return pages


def chunk_text(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """Split page texts into overlapping chunks with metadata."""
    chunks: list[dict] = []
    chunk_id = 0

    for page_num, text in pages:
        # Split by paragraphs or sentences
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current_chunk = ""

        for para in paragraphs:
            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + len(para) + 1 <= chunk_size:
                current_chunk += "\n" + para
            else:
                chunks.append({
                    "text": current_chunk,
                    "page": page_num,
                    "source": source_name,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1
                # Retain overlap from end of current_chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page": page_num,
                "source": source_name,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return chunks


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """Load a PDF and return structured, overlapping chunks."""
    pages = extract_pdf_pages(pdf_path)
    source_name = os.path.basename(pdf_path)
    return chunk_text(pages, source_name=source_name, chunk_size=chunk_size, overlap=overlap)


class VectorStore:
    """Lightweight TF-IDF cosine-similarity vector store with persistence.

    Requires no external C-extensions, guaranteeing fast and reliable
    performance across all environments.
    """

    def __init__(self):
        self.documents: list[dict] = []
        self.doc_vectors: list[dict[str, float]] = []
        self.idf: dict[str, float] = {}
        self.total_docs: int = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple alphanumeric tokenizer with lowercase normalization."""
        return re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())

    def add_documents(self, documents: list[dict]) -> None:
        """Add and index document chunks."""
        self.documents.extend(documents)
        self.total_docs = len(self.documents)

        # Compute document frequency (DF) for each term
        df: Counter = Counter()
        tokenized_docs = []
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            tokenized_docs.append(tokens)
            unique_terms = set(tokens)
            df.update(unique_terms)

        # Compute Inverse Document Frequency (IDF)
        self.idf = {
            term: math.log((self.total_docs + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

        # Compute TF-IDF vectors (normalized)
        self.doc_vectors = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            doc_len = len(tokens) or 1
            vec: dict[str, float] = {}
            norm_sq = 0.0

            for term, count in tf.items():
                tfidf = (count / doc_len) * self.idf.get(term, 1.0)
                vec[term] = tfidf
                norm_sq += tfidf * tfidf

            # Normalize vector to unit length
            norm = math.sqrt(norm_sq) or 1.0
            normalized_vec = {t: val / norm for t, val in vec.items()}
            self.doc_vectors.append(normalized_vec)

    def similarity_search(self, query: str, k: int = 4) -> list[dict]:
        """Search top-k most similar chunks for a query string."""
        if not self.documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Build normalized query TF-IDF vector
        tf = Counter(tokens)
        query_len = len(tokens) or 1
        query_vec: dict[str, float] = {}
        norm_sq = 0.0

        for term, count in tf.items():
            if term in self.idf:
                tfidf = (count / query_len) * self.idf[term]
                query_vec[term] = tfidf
                norm_sq += tfidf * tfidf

        if not query_vec:
            # Query terms not in vocabulary; fallback to substring search
            fallback_results = []
            q_lower = query.lower()
            for doc in self.documents:
                if q_lower in doc["text"].lower():
                    fallback_results.append({**doc, "score": 0.5})
            return fallback_results[:k]

        norm = math.sqrt(norm_sq) or 1.0
        normalized_query = {t: val / norm for t, val in query_vec.items()}

        # Compute cosine similarity with all doc vectors
        scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            dot_product = sum(
                weight * doc_vec.get(term, 0.0)
                for term, weight in normalized_query.items()
            )
            if dot_product > 0.01:
                doc = self.documents[idx]
                scores.append({
                    "text": doc["text"],
                    "page": doc.get("page", 1),
                    "source": doc.get("source", "document"),
                    "chunk_id": doc.get("chunk_id", idx),
                    "score": round(dot_product, 3),
                })

        # Sort descending by similarity score
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:k]

    def save(self, directory: str) -> None:
        """Save vector store and documents to disk."""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, "index.json")
        data = {
            "documents": self.documents,
            "idf": self.idf,
            "total_docs": self.total_docs,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"VectorStore saved to {filepath}")

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load vector store from disk."""
        filepath = os.path.join(directory, "index.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls()
        store.add_documents(data.get("documents", []))
        return store


def build_index(
    pdf_path: str,
    output_dir: str = "./doc_index",
    chunk_size: int = 500,
    overlap: int = 50
) -> VectorStore:
    """Build and persist an index from a PDF file."""
    print(f"Loading and chunking PDF: {pdf_path} ...")
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks.")

    store = VectorStore()
    store.add_documents(chunks)
    store.save(output_dir)
    print(f"Index successfully saved to {output_dir}")
    return store


def load_index(index_dir: str = "./doc_index") -> VectorStore:
    """Load an existing index from disk."""
    return VectorStore.load(index_dir)


def create_sample_pdf(output_path: str = "./sample_research.pdf") -> str:
    """Generate a sample PDF report for immediate demonstration and testing."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "State of Autonomous AI Agents (2026)", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Executive Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    overview_text = (
        "Autonomous AI agents are software systems powered by Large Language Models (LLMs) that "
        "can reason, break down complex goals into sub-tasks, execute tools sequentially or in parallel, "
        "and observe execution outcomes. In 2026, enterprise adoption of agentic workflows surged by 142% "
        "compared to previous years. The predominant architectures have shifted from rigid DAGs to "
        "dynamic tool-calling loops with strict schema-enforced structured outputs."
    )
    pdf.multi_cell(0, 6, overview_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Key Architectural Patterns", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    patterns_text = (
        "Modern agent design emphasizes typed dependency injection, deterministic schema validation "
        "using Pydantic, and hybrid retrieval. Rather than relying solely on web search or static RAG, "
        "dual-engine agents combine local domain knowledge bases (via PDF/vector stores) with live "
        "web grounding. Performance evaluations demonstrate that grounding reduces hallucination rates "
        "from 18.4% to under 2.1% in technical domains."
    )
    pdf.multi_cell(0, 6, patterns_text)
    pdf.ln(4)

    # Page 2
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Benchmark Numbers & Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    metrics_text = (
        "On the AgentBench-2026 benchmark, tool-use accuracy reached 94.6% when models were supplied with "
        "concise, pre-sanitized tool outputs. Latency optimization via specialized inference hardware (such as "
        "Groq LPUs) achieved median response times under 320ms per token generation. Total multi-turn "
        "research workflows averaged 4.2 tool invocations per user inquiry."
    )
    pdf.multi_cell(0, 6, metrics_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Open Challenges", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    challenges_text = (
        "Key unsolved challenges remain around long-context drift, cross-tool authentication security, "
        "and graceful degradation during external API rate-limiting events. Future work focuses on "
        "asynchronous multi-agent coordination protocols and Model Context Protocol (MCP) integrations."
    )
    pdf.multi_cell(0, 6, challenges_text)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf.output(output_path)
    return output_path


if __name__ == "__main__":
    import sys
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "./sample_research.pdf"
    if not os.path.exists(pdf_file):
        print(f"No PDF found at '{pdf_file}'. Creating sample research PDF...")
        pdf_file = create_sample_pdf(pdf_file)
        print(f"Created sample PDF at {pdf_file}")

    index = build_index(pdf_file, output_dir="./doc_index")
    print("\nTesting Query: 'benchmark numbers accuracy'")
    results = index.similarity_search("benchmark numbers accuracy", k=2)
    for r in results:
        print(f"Page {r['page']} (Score: {r['score']}): {r['text'][:150]}...\n")
