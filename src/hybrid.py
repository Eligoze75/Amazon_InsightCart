"""Dual retrieval (BM25 + FAISS) with RRF fusion keyed by ``doc_id``.

Pipeline: optional query expansion → per-query BM25 + FAISS retrieval → RRF between
channels per query → RRF across expanded queries. Does not include cross-encoder
reranking (later milestone).

Run a smoke test from project root (requires built indices)::

    python src/hybrid.py \"best Nintendo console under 300\" --mode hybrid
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# Project root on path for ``from src.*`` (works for ``python src/hybrid.py`` too).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.query_expansion import DEFAULT_NUM_VARIANTS, expand_query

LOGGER = logging.getLogger(__name__)

PROCESSED_PARQUET = _ROOT / "data" / "processed" / "merged_reviews.parquet"

# Reciprocal rank fusion constant (typical range 50–80).
RRF_K = 60

SearchMode = Literal["semantic", "bm25", "hybrid"]


@dataclass
class HybridRetrievalResult:
    """Retrieval bundle: queries used and ranked hits with ``text_faiss`` context."""

    original_query: str
    expanded_queries: list[str]
    hits: list[dict]
    warnings: list[str] = field(default_factory=list)


def fusion_key(row: dict) -> str:
    """Stable id for RRF: prefer ``doc_id``, else ``parent_asin``."""
    doc_id = row.get("doc_id")
    if doc_id is not None and str(doc_id).strip():
        return f"doc:{doc_id}"
    asin = row.get("parent_asin")
    return f"asin:{asin}" if asin else ""


def bm25_search_as_dicts(bm25: Any, query: str, k: int) -> list[dict]:
    """BM25 top-k as dicts aligned with :meth:`SemanticRetriever.search` shape."""
    processed = bm25.preprocess_func(query)
    docs = bm25.vectorizer.get_top_n(processed, bm25.docs, n=k)
    results: list[dict] = []
    for rank, doc in enumerate(docs, start=1):
        row = dict(doc.metadata)
        row["content"] = doc.page_content
        row["score"] = float(k - rank + 1)
        row["rank"] = rank
        results.append(row)
    return results


def rrf_fuse(rank_lists: list[list[dict]]) -> list[dict]:
    """Merge ranked result lists using RRF; one row per fusion key."""
    rank_lists = [lst for lst in rank_lists if lst]
    if not rank_lists:
        return []

    scores: dict[str, float] = {}
    best_row: dict[str, dict] = {}
    for results in rank_lists:
        for rank, row in enumerate(results, start=1):
            key = fusion_key(row)
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in best_row:
                best_row[key] = dict(row)

    ordered_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    out: list[dict] = []
    for rank, key in enumerate(ordered_keys, start=1):
        merged = best_row[key].copy()
        merged["score"] = scores[key]
        merged["rank"] = rank
        out.append(merged)
    return out


def _enrich_text_faiss(hits: list[dict], parquet_path: Path) -> None:
    """Set ``content`` to the canonical ``text_faiss`` passage when ``doc_id`` maps."""
    ids: list[str] = []
    for h in hits:
        did = h.get("doc_id")
        if did is not None and str(did).strip():
            ids.append(str(did))
    if not ids:
        return
    if not parquet_path.is_file():
        LOGGER.warning("Cannot enrich text_faiss: missing %s", parquet_path)
        return
    df = pd.read_parquet(parquet_path, columns=["doc_id", "text_faiss"])
    sub = df[df["doc_id"].astype(str).isin(set(ids))]
    by_id = dict(zip(sub["doc_id"].astype(str), sub["text_faiss"]))
    for h in hits:
        did = h.get("doc_id")
        if did is None:
            continue
        key = str(did)
        if key in by_id:
            h["content"] = by_id[key]


def _default_pool(top_k: int) -> int:
    return max(top_k * 2, 8)


def search(
    queries: list[str],
    mode: SearchMode,
    top_k: int,
    *,
    semantic: Any | None = None,
    bm25: Any | None = None,
    pool: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Run retrieval for one or more query strings; returns hits and UI warnings.

    Hybrid mode applies RRF between FAISS and BM25 lists per query (FAISS list
    first for tie metadata), then RRF across queries when multiple strings are
    given.

    Args:
        queries: Non-empty stripped query strings (after expansion if any).
        mode: ``semantic``, ``bm25``, or ``hybrid``.
        top_k: Final number of hits.
        semantic: Loaded :class:`~semantic.SemanticRetriever`, or load default.
        bm25: Loaded BM25 retriever, or load from disk if present.
        pool: Per-query retrieval depth before fusion; default ``max(2*top_k, 8)``.

    Returns:
        Tuple of (hits with ``content``, ``score``, ``rank``, metadata) and warnings.
    """
    from src.semantic import SemanticRetriever
    from src import bm25 as bm25_mod

    warnings: list[str] = []
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        return [], warnings

    if semantic is None:
        sr = SemanticRetriever()
        if not (sr.index_dir / "index.faiss").is_file():
            return [], ["FAISS index missing. Run: python src/semantic.py"]
        sr.load()
        semantic = sr

    if bm25 is None and mode in ("bm25", "hybrid"):
        path = bm25_mod.BM25_INDEX_PATH
        if path.is_file():
            try:
                bm25 = bm25_mod.load_bm25_retriever(path)
            except OSError as exc:
                warnings.append(f"Could not load BM25: {exc}")
                bm25 = None
        else:
            bm25 = None

    if mode in ("bm25", "hybrid") and bm25 is None:
        warnings.append(
            "BM25 index not found — falling back to semantic retrieval only."
        )
        if mode == "hybrid":
            mode = "semantic"

    k_pool = pool if pool is not None else _default_pool(top_k)
    multi = len(queries) > 1

    if mode == "semantic":
        if multi:
            lists_sem = [semantic.search(q, k=k_pool) for q in queries]
            hits = rrf_fuse(lists_sem)[:top_k]
        else:
            hits = semantic.search(queries[0], k=top_k)
        _enrich_text_faiss(hits, PROCESSED_PARQUET)
        return hits, warnings

    if mode == "bm25":
        if bm25 is None:
            return [], warnings
        if multi:
            lists_bm = [bm25_search_as_dicts(bm25, q, k_pool) for q in queries]
            hits = rrf_fuse(lists_bm)[:top_k]
        else:
            hits = bm25_search_as_dicts(bm25, queries[0], top_k)
        _enrich_text_faiss(hits, PROCESSED_PARQUET)
        return hits, warnings

    # hybrid
    per_query_fused: list[list[dict]] = []
    for q in queries:
        sem_hits = semantic.search(q, k=k_pool)
        bm_hits = bm25_search_as_dicts(bm25, q, k_pool)
        per_query_fused.append(rrf_fuse([sem_hits, bm_hits]))

    if len(per_query_fused) == 1:
        hits = per_query_fused[0][:top_k]
    else:
        hits = rrf_fuse(per_query_fused)[:top_k]

    _enrich_text_faiss(hits, PROCESSED_PARQUET)
    return hits, warnings


def retrieve_with_expansion(
    query: str,
    *,
    mode: SearchMode = "hybrid",
    expand: bool = True,
    num_variants: int = DEFAULT_NUM_VARIANTS,
    top_k: int = 10,
    semantic: Any | None = None,
    bm25: Any | None = None,
) -> HybridRetrievalResult:
    """Expand the query (optional), then run :func:`search`."""
    q = query.strip()
    if not q:
        return HybridRetrievalResult(
            original_query="",
            expanded_queries=[],
            hits=[],
            warnings=["Empty query."],
        )

    expanded_queries = [q]
    warnings: list[str] = []
    if expand:
        try:
            result = expand_query(q, num_variants=num_variants)
            expanded_queries = result.all_queries()
        except Exception as exc:
            LOGGER.warning("Query expansion failed: %s", exc)
            warnings.append(f"Query expansion failed ({exc}); using original query only.")

    hits, w2 = search(expanded_queries, mode, top_k, semantic=semantic, bm25=bm25)
    warnings.extend(w2)
    return HybridRetrievalResult(
        original_query=q,
        expanded_queries=expanded_queries,
        hits=hits,
        warnings=warnings,
    )


def _mode_from_ui(label: str) -> SearchMode:
    m = label.strip().lower()
    if m in ("semantic", "bm25", "hybrid"):
        return m  # type: ignore[return-value]
    return "hybrid"


def search_from_ui_mode(
    queries: list[str],
    ui_mode: str,
    top_k: int,
    *,
    semantic: Any | None = None,
    bm25: Any | None = None,
) -> tuple[list[dict], list[str]]:
    """Adapter for Streamlit labels: ``Semantic`` / ``BM25`` / ``Hybrid``."""
    return search(queries, _mode_from_ui(ui_mode), top_k, semantic=semantic, bm25=bm25)


def main() -> None:
    """CLI: run a test retrieval and print context snippets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Hybrid BM25 + FAISS retrieval (RRF).")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is a good wireless controller for PS5?",
        help="Search query",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval mode",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of hits to return"
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Skip Claude query expansion",
    )
    args = parser.parse_args()

    result = retrieve_with_expansion(
        args.query,
        mode=args.mode,
        expand=not args.no_expand,
        top_k=args.top_k,
    )
    if result.warnings:
        for w in result.warnings:
            print(f"Warning: {w}", file=sys.stderr)
    print(f"Original: {result.original_query}")
    print(f"Queries used ({len(result.expanded_queries)}): {result.expanded_queries}")
    print(f"Hits: {len(result.hits)}\n")
    for h in result.hits:
        title = h.get("product_post_title") or h.get("title") or "?"
        content = (h.get("content") or "")[:400]
        print(f"--- rank {h.get('rank')} score={h.get('score', 0):.4f} | {title}")
        print(f"doc_id={h.get('doc_id')} parent_asin={h.get('parent_asin')}")
        print(content + ("…" if len(str(h.get("content", ""))) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
