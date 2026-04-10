"""Streamlit app for Amazon Video Games product search.

Run from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Ensure project root is on path so src.* imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.semantic import SemanticRetriever
from src.utils import lookup_reviews

FEEDBACK_CSV = ROOT / "data" / "feedback.csv"
FEEDBACK_HEADERS = ["timestamp", "query", "mode", "parent_asin", "title", "feedback"]


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading semantic index…")
def _load_semantic() -> SemanticRetriever | None:
    index_dir = ROOT / "data" / "processed" / "faiss_index"
    if not (index_dir / "index.faiss").exists():
        return None
    r = SemanticRetriever(index_dir=index_dir)
    r.load()
    return r


@st.cache_resource(show_spinner="Loading BM25 index…")
def _load_bm25():
    try:
        from src.bm25 import BM25Retriever  # type: ignore[import]

        r = BM25Retriever()
        r.load()
        return r
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: stars display
# ---------------------------------------------------------------------------


def _stars(rating: float | None) -> str:
    if rating is None:
        return ""
    full = int(round(float(rating)))
    return "★" * full + "☆" * (5 - full)


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
                "timestamp": datetime.now(datetime.UTC).isoformat(),
                "query": query,
                "mode": mode,
                "parent_asin": parent_asin,
                "title": title,
                "feedback": vote,
            }
        )


# ---------------------------------------------------------------------------
# Helper: run search
# ---------------------------------------------------------------------------


def _run_search(query: str, mode: str, k: int = 3) -> list[dict]:
    """Dispatch to the right retriever based on mode."""
    semantic = _load_semantic()
    bm25 = _load_bm25()

    if semantic is None:
        st.error(
            "Semantic index not found. Build it first by running:\n\n"
            "```\npython src/semantic.py\n```"
        )
        return []

    if mode == "Semantic":
        return semantic.search(query, k=k)

    if mode == "BM25":
        if bm25 is None:
            st.warning("BM25 index not built yet. Showing semantic results instead.")
            return semantic.search(query, k=k)
        return bm25.search(query, k=k)

    # Hybrid
    sem_results = semantic.search(query, k=k * 2)

    if bm25 is None:
        st.warning("BM25 index not available — showing semantic results only.")
        return sem_results[:k]

    bm25_results = bm25.search(query, k=k * 2)

    # Normalize BM25 scores (min-max)
    bm25_scores = {r["parent_asin"]: r["score"] for r in bm25_results}
    if bm25_scores:
        mn, mx = min(bm25_scores.values()), max(bm25_scores.values())
        denom = mx - mn if mx != mn else 1.0
        bm25_scores = {k: (v - mn) / denom for k, v in bm25_scores.items()}

    # Normalize semantic scores (FAISS L2 → lower is better, invert)
    sem_scores = {r["parent_asin"]: r["score"] for r in sem_results}
    if sem_scores:
        mn, mx = min(sem_scores.values()), max(sem_scores.values())
        denom = mx - mn if mx != mn else 1.0
        # For L2 distance, smaller = better → invert normalization
        sem_scores = {k: 1.0 - (v - mn) / denom for k, v in sem_scores.items()}

    # Merge by asin
    all_asins = set(bm25_scores) | set(sem_scores)
    meta_lookup = {r["parent_asin"]: r for r in sem_results + bm25_results}
    alpha = 0.5
    combined = []
    for asin in all_asins:
        b = bm25_scores.get(asin, 0.0)
        s = sem_scores.get(asin, 0.0)
        hybrid_score = alpha * b + (1 - alpha) * s
        row = meta_lookup[asin].copy()
        row["score"] = hybrid_score
        combined.append(row)

    combined.sort(key=lambda x: x["score"], reverse=True)
    for i, row in enumerate(combined[:k], start=1):
        row["rank"] = i
    return combined[:k]


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Video Games Search", layout="wide")
    st.title("Amazon Video Games Product Search")

    # Mode selector
    mode = st.radio(
        "Search Mode",
        ["Semantic", "BM25", "Hybrid"],
        horizontal=True,
    )

    # Query input
    query = st.text_input(
        "Enter your search query", placeholder="e.g. wireless controller for PS5"
    )

    search_clicked = st.button("Search", type="primary")

    if search_clicked and query.strip():
        with st.spinner("Searching…"):
            results = _run_search(query.strip(), mode, k=3)

        if not results:
            st.info("No results found.")
            return

        asins = [r["parent_asin"] for r in results]
        reviews_df = lookup_reviews(
            asins, parquet_path=ROOT / "data" / "processed" / "reviews.parquet"
        )
        st.session_state["results"] = results
        st.session_state["reviews_map"] = reviews_df.set_index("parent_asin").to_dict(orient="index")
        st.session_state["search_query"] = query.strip()
        st.session_state["search_mode"] = mode

    if "results" not in st.session_state:
        return

    results = st.session_state["results"]
    reviews_map = st.session_state["reviews_map"]
    query = st.session_state["search_query"]
    mode = st.session_state["search_mode"]

    st.markdown("---")
    st.subheader(f"Top {len(results)} results — {mode} search")

    for result in results:
        asin = result.get("parent_asin", "")
        title = result.get("title", "Unknown product")
        avg_rating = result.get("average_rating")
        rating_num = result.get("rating_number")
        score = result.get("score", 0.0)

        review_info = reviews_map.get(asin, {})
        review_text = review_info.get("review_text", "")
        if review_text:
            review_text = review_text[:200] + (
                "…" if len(review_text) > 200 else ""
            )

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
                    _save_feedback(query, mode, asin, title, "up")
                    st.success("Thanks!")
                if thumb_down:
                    _save_feedback(query, mode, asin, title, "down")
                    st.success("Noted!")


if __name__ == "__main__":
    main()
