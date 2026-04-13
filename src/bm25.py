"""Build and persist a BM25 index from the processed product parquet.

Run from project root:
    python src/bm25.py
"""

from __future__ import annotations
import logging
import pickle
import time
from pathlib import Path

import pandas as pd
from langchain_community.retrievers import BM25Retriever

LOGGER = logging.getLogger(__name__)

_ROOT             = Path(__file__).resolve().parent.parent
PROCESSED_PARQUET = _ROOT / "data" / "processed" / "merged_reviews.parquet"
BM25_INDEX_PATH   = _ROOT / "data" / "context_store" / "bm25_retriever.pkl"
BM25_K            = 3

# Columns stored in each Document's metadata — returned by hybrid search
# and used by the app to display results without an extra parquet lookup.
_META_COLS = [
    "doc_id", "parent_asin", "product_title",
    "average_rating", "rating_number", "review_texts",
]


def load_processed_dataframe(parquet_path: Path | None = None) -> pd.DataFrame:
    """Load only the columns needed to build the BM25 index."""
    path = parquet_path or PROCESSED_PARQUET
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing processed parquet: {path}. Run src/preprocess.py first."
        )
    return pd.read_parquet(path, columns=["text_bm25"] + _META_COLS)


def build_bm25_retriever(df: pd.DataFrame | None = None) -> BM25Retriever:
    """Fit BM25 over text_bm25; store display metadata on each document."""
    frame     = df if df is not None else load_processed_dataframe()
    texts     = frame["text_bm25"].astype(str).tolist()
    metadatas = frame[_META_COLS].to_dict(orient="records")
    LOGGER.info("Building BM25 over %s documents", len(texts))
    return BM25Retriever.from_texts(texts, metadatas=metadatas, k=BM25_K)


def save_bm25_retriever(retriever: BM25Retriever, path: Path | None = None) -> None:
    """Pickle the retriever to disk."""
    out = path or BM25_INDEX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(retriever, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Saved BM25 retriever → %s", out)


def load_bm25_retriever(path: Path | None = None) -> BM25Retriever:
    """Load a pickled BM25 retriever produced by save_bm25_retriever."""
    src = path or BM25_INDEX_PATH
    with open(src, "rb") as f:
        return pickle.load(f)


def build_and_save() -> BM25Retriever:
    retriever = build_bm25_retriever()
    save_bm25_retriever(retriever)
    return retriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_and_save()


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    LOGGER.info("Finished in %.1f minutes", (time.perf_counter() - start) / 60)
