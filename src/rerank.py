"""Cross-encoder re-ranking for retrieval hits.

Run from project root (requires indices built for full hybrid demo)::

    python src/rerank.py \"wireless PS5 controller\" --mode hybrid
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOGGER = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_POOL = 15


def load_cross_encoder() -> Any:
    """Loads a CrossEncoder; ``model_name`` defaults to MS MARCO MiniLM."""
    from sentence_transformers import CrossEncoder

    LOGGER.info("Loading cross-encoder %s", CROSS_ENCODER_MODEL)
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(
    query: str,
    hits: list[dict],
    top_k: int,
    *,
    model: Any | None = None,
) -> list[dict]:
    """Scores (query, passage) pairs and returns the top ``top_k`` rows by CE score.

    Each hit should include a ``content`` string (e.g. ``text_faiss`` passage).
    ``score`` is replaced with the cross-encoder score; prior score is kept as
    ``retrieval_score`` when present.
    """
    if not hits or top_k <= 0:
        return []

    ce = model or load_cross_encoder()
    texts = [str(h.get("content") or "") for h in hits]
    pairs = [[query, t] for t in texts]
    scores = ce.predict(pairs, show_progress_bar=False)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)
    out: list[dict] = []
    for rank, idx in enumerate(order[:top_k], start=1):
        row = dict(hits[idx])
        if "score" in row:
            row["retrieval_score"] = row["score"]
        row["score"] = float(scores[idx])
        row["rank"] = rank
        out.append(row)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.hybrid import SearchMode, retrieve_with_expansion

    parser = argparse.ArgumentParser(
        description="Retrieve then cross-encoder re-rank (MS MARCO MiniLM).",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="good wireless controller for PS5",
        help="Search query",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval mode before re-ranking",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Final number of hits after re-ranking",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=DEFAULT_RERANK_POOL,
        help="Candidates to retrieve before re-ranking",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Skip Claude query expansion",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Only run retrieval (no cross-encoder)",
    )
    args = parser.parse_args()

    mode: SearchMode = args.mode  # type: ignore[assignment]
    if args.no_rerank:
        result = retrieve_with_expansion(
            args.query,
            mode=mode,
            expand=not args.no_expand,
            top_k=args.top_k,
            rerank=False,
        )
    else:
        result = retrieve_with_expansion(
            args.query,
            mode=mode,
            expand=not args.no_expand,
            top_k=args.top_k,
            rerank=True,
            rerank_pool=args.pool,
        )

    if result.warnings:
        for w in result.warnings:
            print(f"Warning: {w}", file=sys.stderr)

    print(f"Original: {result.original_query}")
    print(f"Queries used ({len(result.expanded_queries)}): {result.expanded_queries}")
    print(f"Hits: {len(result.hits)}\n")

    for h in result.hits:
        title = h.get("product_post_title") or h.get("product_title") or "?"
        content = (h.get("content") or "")[:400]
        rs = h.get("retrieval_score")
        extra = f" retrieval={rs:.4f}" if isinstance(rs, (int, float)) else ""
        print(
            f"--- rank {h.get('rank')} score={h.get('score', 0):.4f}{extra} | {title}"
        )
        print(f"doc_id={h.get('doc_id')} parent_asin={h.get('parent_asin')}")
        print(content + ("…" if len(str(h.get("content", ""))) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
