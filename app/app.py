"""Streamlit app for Amazon Video Games product search.

Run from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Project root for ``src.*`` imports; ``src/`` itself so ``documents`` resolves
# (same as running scripts from ``src/``).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.hybrid import search_from_ui_mode
from src.query_expansion import DEFAULT_NUM_VARIANTS, expand_query
from src.semantic import SemanticRetriever
from src.utils import lookup_reviews

FEEDBACK_CSV = ROOT / "data" / "feedback.csv"
FEEDBACK_HEADERS = ["timestamp", "query", "mode", "parent_asin", "title", "feedback"]


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading semantic index…")
def _load_semantic() -> SemanticRetriever | None:
    index_dir = ROOT / "data" / "context_store" / "faiss_index"
    if not (index_dir / "index.faiss").exists():
        return None
    r = SemanticRetriever(index_dir=index_dir)
    r.load()
    return r


@st.cache_resource(show_spinner="Loading BM25 index…")
def _load_bm25():
    from src import bm25 as bm25_mod

    path = bm25_mod.BM25_INDEX_PATH
    if not path.is_file():
        return None
    try:
        return bm25_mod.load_bm25_retriever(path)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Search (delegates to src.hybrid)
# ---------------------------------------------------------------------------


def _run_search(queries: list[str], mode: str, k: int = 3) -> list[dict]:
    """Dispatch to :func:`search_from_ui_mode` with cached retrievers."""
    mode_l = mode.strip().lower()
    semantic = _load_semantic() if mode_l in ("semantic", "hybrid") else None
    if mode_l in ("semantic", "hybrid") and semantic is None:
        return []
    bm25 = _load_bm25()
    results, warnings = search_from_ui_mode(
        queries, mode, k, semantic=semantic, bm25=bm25
    )
    for w in warnings:
        st.warning(w)
    return results


# ---------------------------------------------------------------------------
# Helper: stars display
# ---------------------------------------------------------------------------


def _stars(rating: float | None) -> str:
    if rating is None:
        return ""
    full = int(round(float(rating)))
    return "★" * full + "☆" * (5 - full)


def _product_title(row: dict) -> str:
    return str(
        row.get("product_post_title") or row.get("title") or "Unknown product"
    )


# ---------------------------------------------------------------------------
# Helper: record feedback
# ---------------------------------------------------------------------------


def _save_feedback(
    query: str, mode: str, parent_asin: str, title: str, vote: str
) -> None:
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "mode": mode,
                "parent_asin": parent_asin,
                "title": title,
                "feedback": vote,
            }
        )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Video Games Search", layout="wide")
    st.title("Amazon Video Games Product Search")

    mode = st.radio(
        "Search Mode",
        ["Semantic", "BM25", "Hybrid"],
        horizontal=True,
        index=2
    )

    use_expansion = st.checkbox(
        "Expand query",
        value=True,
        help=(
            "Rewrites your query into paraphrases for broader recall. "
            "Requires ANTHROPIC_API_KEY in .env or the environment."
        ),
    )

    query = st.text_input(
        "Enter your search query", placeholder="e.g. wireless controller for PS5"
    )

    search_clicked = st.button("Search", type="primary")

    if mode in ("Semantic", "Hybrid") and _load_semantic() is None:
        st.error(
            "Semantic index not found. Build it first by running:\n\n"
            "```\npython src/semantic.py\n```"
        )
        return

    if search_clicked and query.strip():
        raw_q = query.strip()
        queries_for_search = [raw_q]

        if use_expansion:
            try:
                expanded = expand_query(raw_q, num_variants=DEFAULT_NUM_VARIANTS)
                queries_for_search = expanded.all_queries()
                st.session_state["expanded_queries"] = list(queries_for_search)
                st.session_state["expansion_original"] = expanded.original
            except Exception as exc:  # API, network, or missing key — fall back to single query
                st.session_state.pop("expanded_queries", None)
                st.session_state.pop("expansion_original", None)
                st.warning(
                    f"Query expansion failed ({exc}). Searching with the original query only."
                )
        else:
            st.session_state.pop("expanded_queries", None)
            st.session_state.pop("expansion_original", None)

        with st.spinner("Searching…"):
            results = _run_search(queries_for_search, mode, k=3)

        if not results:
            st.info("No results found.")
            st.session_state.pop("results", None)
            return

        asins = [r["parent_asin"] for r in results if r.get("parent_asin")]
        reviews_df = lookup_reviews(
            asins, parquet_path=ROOT / "data" / "processed" / "reviews.parquet"
        )
        st.session_state["results"] = results
        st.session_state["reviews_map"] = reviews_df.set_index("parent_asin").to_dict(
            orient="index"
        )
        st.session_state["search_query"] = raw_q
        st.session_state["search_mode"] = mode

    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    reviews_map = st.session_state["reviews_map"]
    display_query = st.session_state["search_query"]
    mode = st.session_state["search_mode"]

    st.markdown("---")

    if st.session_state.get("expanded_queries"):
        with st.expander("Queries used for retrieval", expanded=False):
            for i, q in enumerate(st.session_state["expanded_queries"], start=1):
                st.markdown(f"{i}. {q}")

    st.subheader(f"Top {len(results)} results — {mode} search")

    for result in results:
        asin = result.get("parent_asin", "")
        title = _product_title(result)
        avg_rating = result.get("average_rating")
        rating_num = result.get("rating_number")
        score = result.get("score", 0.0)

        review_info = reviews_map.get(asin, {})
        review_text = review_info.get("review_text", "")
        if review_text:
            review_text = review_text[:200] + ("…" if len(review_text) > 200 else "")

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{title}**")
                if avg_rating is not None:
                    stars = _stars(avg_rating)
                    count = f"({int(rating_num):,} reviews)" if rating_num else ""
                    st.markdown(f"{stars} {avg_rating:.1f} {count}")
                if review_text:
                    st.markdown(f'*"{review_text}"*')
                st.caption(f"Score: {score:.4f} | ASIN: {asin}")
            with col2:
                st.markdown("**Helpful?**")
                thumb_up = st.button("👍", key=f"up_{asin}")
                thumb_down = st.button("👎", key=f"down_{asin}")
                if thumb_up:
                    _save_feedback(display_query, mode, asin, title, "up")
                    st.success("Thanks!")
                if thumb_down:
                    _save_feedback(display_query, mode, asin, title, "down")
                    st.success("Noted!")


if __name__ == "__main__":
    main()
