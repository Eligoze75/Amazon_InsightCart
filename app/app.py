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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.hybrid import SearchMode, retrieve_with_expansion
from src.semantic import SemanticRetriever

FEEDBACK_CSV     = ROOT / "data" / "feedback.csv"
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


@st.cache_resource(show_spinner="Loading cross-encoder…")
def _load_cross_encoder():
    from src.rerank import load_cross_encoder

    return load_cross_encoder()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _ui_mode_to_search_mode(ui_label: str) -> SearchMode:
    mapping: dict[str, SearchMode] = {
        "Semantic": "semantic",
        "BM25": "bm25",
        "Hybrid": "hybrid",
    }
    return mapping.get(ui_label, "hybrid")


def _run_search(
    raw_query: str,
    ui_mode: str,
    *,
    use_expansion: bool,
    use_rerank: bool,
    final_k: int = 3,
    rerank_pool: int = 15,
):
    """Runs :func:`retrieve_with_expansion` with cached indices and optional CE."""
    semantic = _load_semantic()
    if ui_mode in ("Semantic", "Hybrid") and semantic is None:
        return [], [], []
    bm25 = _load_bm25()
    ce = _load_cross_encoder() if use_rerank else None
    mode = _ui_mode_to_search_mode(ui_mode)
    result = retrieve_with_expansion(
        raw_query,
        mode=mode,
        expand=use_expansion,
        top_k=final_k,
        rerank=use_rerank,
        rerank_pool=rerank_pool,
        cross_encoder=ce,
        semantic=semantic,
        bm25=bm25,
    )
    for w in result.warnings:
        st.warning(w)
    return result.hits, result.expanded_queries, result.warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stars(rating: float | None) -> str:
    if rating is None:
        return ""
    full = int(round(float(rating)))
    return "★" * full + "☆" * (5 - full)


def _title(row: dict) -> str:
    return str(row.get("product_title") or row.get("product_post_title") or "Unknown product")


def _save_feedback(query: str, mode: str, parent_asin: str, title: str, vote: str) -> None:
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "query":       query,
            "mode":        mode,
            "parent_asin": parent_asin,
            "title":       title,
            "feedback":    vote,
        })


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Video Games Search", layout="wide")
    st.title("Amazon Video Games Product Search")

    mode = st.radio("Search Mode", ["Semantic", "BM25", "Hybrid"], horizontal=True, index=2)

    use_expansion = st.checkbox(
        "Expand query with Claude",
        value=True,
        help="Rewrites your query into paraphrases for broader recall. Requires ANTHROPIC_API_KEY in .env.",
    )

    use_rerank = st.checkbox(
        "Re-rank with Cross-Encoder",
        value=True,
        help="Retrieves a larger candidate set, then scores query–passage pairs with cross-encoder/ms-marco-MiniLM-L-6-v2.",
    )

    query = st.text_input("Enter your search query", placeholder="e.g. wireless controller for PS5")
    search_clicked = st.button("Search", type="primary")

    if mode in ("Semantic", "Hybrid") and _load_semantic() is None:
        st.error("Semantic index not found. Run:\n\n```\npython src/semantic.py\n```")
        return

    if search_clicked and query.strip():
        raw_q = query.strip()

        with st.spinner("Searching…"):
            results, expanded_queries, _warnings = _run_search(
                raw_q,
                mode,
                use_expansion=use_expansion,
                use_rerank=use_rerank,
                final_k=3,
                rerank_pool=15,
            )

        if expanded_queries:
            st.session_state["expanded_queries"] = expanded_queries
        else:
            st.session_state.pop("expanded_queries", None)

        if not results:
            st.info("No results found.")
            st.session_state.pop("results", None)
            return

        st.session_state["results"]      = results
        st.session_state["search_query"] = raw_q
        st.session_state["search_mode"]  = mode

    if "results" not in st.session_state:
        return

    results       = st.session_state["results"]
    display_query = st.session_state["search_query"]
    mode          = st.session_state["search_mode"]

    st.markdown("---")

    if st.session_state.get("expanded_queries"):
        with st.expander("Queries used for retrieval", expanded=False):
            for i, q in enumerate(st.session_state["expanded_queries"], start=1):
                st.markdown(f"{i}. {q}")

    st.subheader(f"Top {len(results)} results — {mode} search")

    for result in results:
        asin       = result.get("parent_asin", "")
        title      = _title(result)
        avg_rating = result.get("average_rating")
        rating_num = result.get("rating_number")
        score      = result.get("score", 0.0)

        # Review snippet comes from aggregated review_texts stored in parquet metadata
        review_text = str(result.get("review_texts") or "")
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
                if st.button("👍", key=f"up_{asin}"):
                    _save_feedback(display_query, mode, asin, title, "up")
                    st.success("Thanks!")
                if st.button("👎", key=f"down_{asin}"):
                    _save_feedback(display_query, mode, asin, title, "down")
                    st.success("Noted!")


if __name__ == "__main__":
    main()
