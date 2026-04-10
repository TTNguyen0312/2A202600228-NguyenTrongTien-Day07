from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

import yaml

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import ChunkingStrategyComparator, CustomChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

INDEX_PATH = "data/edu_policy_index.json"


#  Loaders 

def load_education_policy_documents(folder: str = "data/education_policy") -> list[Document]:
    """Load education policy documents with metadata from metadata.yml."""
    folder_path = Path(folder)
    schema = yaml.safe_load((folder_path / "metadata.yml").read_text(encoding="utf-8"))
    shared = {k: v for k, v in schema.items() if k != "documents"}
    per_doc: dict[str, dict] = schema.get("documents") or {}

    documents: list[Document] = []
    for path in sorted(folder_path.glob("*.md")):
        documents.append(
            Document(
                id=path.stem,
                content=path.read_text(encoding="utf-8"),
                metadata={
                    **shared,
                    "file": str(path),
                    **(per_doc.get(path.stem) or {}),
                },
            )
        )
    return documents


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from arbitrary file paths."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path}")
            continue
        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue
        documents.append(
            Document(
                id=path.stem,
                content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )
    return documents


# Chunking 
def chunk_documents(docs: list[Document], chunk_size: int = 1500) -> list[Document]:
    """
    Split each document into smaller chunks before embedding.

    Embedding a whole document produces one vector that averages all its
    meaning. Chunking lets the store retrieve the specific passage that answers
    the question, not just the most-similar document.
    """
    chunker = CustomChunker(max_size=chunk_size, overlap=chunk_size // 5)
    chunks: list[Document] = []
    for doc in docs:
        for i, text in enumerate(chunker.chunk(doc.content)):
            chunks.append(Document(
                id=f"{doc.id}_c{i}",
                content=text,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i},
            ))
    return chunks


# Index persistence 

def save_index(store: EmbeddingStore, path: str) -> None:
    """Serialize the store's chunk records (content + embeddings + metadata) to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store._store, f, ensure_ascii=False)
    print(f"  Saved {len(store._store)} chunks → {path}")


def load_index(path: str, embedder: Callable[[str], list[float]]) -> EmbeddingStore:
    """Restore a previously saved store from JSON (no re-embedding)."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    store = EmbeddingStore(collection_name="edu_policy_store", embedding_fn=embedder)
    store._store = records
    print(f"  Loaded {len(records)} chunks ← {path}")
    return store


# ── Embedder / LLM helpers ────────────────────────────────────────────────────

def build_embedder(provider: str) -> Callable[[str], list[float]]:
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            pass
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            pass
    return _mock_embed


def demo_llm(prompt: str) -> str:
    """Fallback mock LLM — configure EMBEDDING_PROVIDER=openai for real answers."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] {preview}..."


def build_llm_fn(provider: str) -> Callable[[str], str]:
    """
    Return a real LLM backed by OpenAI chat completions when available.

    demo_llm causes identical-looking answers because it returns the same
    fixed prefix of every prompt regardless of the question.
    """
    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI()

            def _llm(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or ""

            return _llm
        except Exception:
            import traceback
            print("  [warn] OpenAI LLM unavailable, falling back to demo_llm")
            traceback.print_exc()
    return demo_llm


# ── Commands ──────────────────────────────────────────────────────────────────

def build_index() -> int:
    """
    One-time indexing pipeline: load → chunk → embed → save to disk.

    Run with:  python main.py build
    """
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    print("=== 1. Loading Documents ===")
    docs = load_education_policy_documents()
    print(f"  {len(docs)} documents loaded")
    for doc in docs:
        print(f"    {doc.id}  category={doc.metadata.get('category')}  phase={doc.metadata.get('phase')}")

    print("\n=== 2. Chunking Strategy Comparison (first 3 docs) ===")
    comparator = ChunkingStrategyComparator()
    for doc in docs[:3]:
        print(f"\n  {doc.id}")
        for strategy, stats in comparator.compare(doc.content).items():
            print(f"    {strategy:15s}: {stats['count']:3d} chunks  avg_len={stats['avg_length']:.0f}")

    print("\n=== 3. Chunking for Embedding ===")
    chunks = chunk_documents(docs, chunk_size=1500)
    print(f"  {len(docs)} docs → {len(chunks)} chunks (CustomChunker, max_size=1500, overlap=300)")

    print("\n=== 4. Embedding ===")
    embedder = build_embedder(provider)
    print(f"  Embedder: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    store = EmbeddingStore(collection_name="edu_policy_store", embedding_fn=embedder)
    store.add_documents(chunks)
    print(f"  {store.get_collection_size()} chunk vectors computed")

    print("\n=== 5. Saving Index ===")
    save_index(store, INDEX_PATH)
    print("\nIndex ready. Run queries with:  python main.py <question>")
    return 0


def parse_filter(raw: str) -> tuple[dict, str]:
    """
    Parse an optional metadata filter prefix from the query string.

    Syntax:  @key=value <question>
    Example: @phase=appeals quy định phúc khảo là gì?
             @category=coi_thi thành phần ban coi thi gồm những ai?

    Returns (filter_dict, question). filter_dict is empty when no prefix found.
    """
    import re
    match = re.match(r'^(@\w+=\w+\s+)+', raw)
    if not match:
        return {}, raw

    filter_dict: dict = {}
    for key, value in re.findall(r'@(\w+)=(\w+)', match.group(0)):
        filter_dict[key] = value

    question = raw[match.end():].strip()
    return filter_dict, question


def run_query() -> int:
    """
    Interactive query loop: load saved index once, then answer questions until exit.

    Run with:  python main.py
    Exit with: 'quit', 'exit', or Ctrl-C / Ctrl-D

    Metadata filter syntax (optional prefix):
        @phase=appeals Quy định phúc khảo là gì?
        @category=coi_thi Thành phần ban coi thi gồm những ai?

    Available phase values:    general, preparation, registration, exam, grading, appeals
    Available category values: quy_dinh_chung, to_chuc_thi, dang_ky_thi, thi_sinh,
                                de_thi, coi_thi, cham_thi, phuc_khao
    """
    if not Path(INDEX_PATH).exists():
        print(f"No index found at {INDEX_PATH}. Run first:  python main.py build")
        return 1

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    print("=== Loading Index ===")
    embedder = build_embedder(provider)
    print(f"  Embedder: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    store = load_index(INDEX_PATH, embedder)

    llm_fn = build_llm_fn(provider)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    print("\nKnowledge base ready. Type your question (or 'quit' to exit).")
    print("Tip: prefix with @phase=<value> or @category=<value> to filter by metadata.")
    print("     e.g.  @phase=appeals quy định phúc khảo là gì?\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        metadata_filter, query = parse_filter(raw)

        if metadata_filter:
            print(f"  [filter] {metadata_filter}")
            results = store.search_with_filter(query, top_k=3, metadata_filter=metadata_filter)
        else:
            results = store.search(query, top_k=3)

        print("\n  [retrieved chunks]")
        for i, r in enumerate(results, 1):
            print(f"  {i}. score={r['score']:.4f}  doc={r['metadata'].get('doc_id')}  "
                  f"category={r['metadata'].get('category')}  phase={r['metadata'].get('phase')}")
            print(f"     {r['content'][:120].replace(chr(10), ' ')}...")

        if metadata_filter:
            # Build answer directly from filtered results (agent.answer uses store.search internally)
            context = "\n\n".join(r["content"] for r in results)
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            print("\nAgent:", llm_fn(prompt), "\n")
        else:
            print("\nAgent:", agent.answer(query, top_k=3), "\n")

    return 0


#  Entry point 

def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "build":
        return build_index()
    return run_query()


if __name__ == "__main__":
    raise SystemExit(main())
