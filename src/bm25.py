"""Build and persist a BM25 retriever from the processed parquet."""

from __future__ import annotations
import pandas as pd
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
import logging
import pickle

LOGGER = logging.getLogger(__name__)
PROCESSED_PARQUET = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "merged_reviews.parquet"
)
BM25_INDEX_PATH = (
    Path(__file__).resolve().parent.parent / "context_store" / "bm25_retriever.pkl"
)
BM25_K = 3 # Documents returned per query (BM25Retriever default is 4).


def load_processed_dataframe(parquet_path: Path | None = None) -> pd.DataFrame:
    """Loads BM25 text and stable doc_id (no packed details JSON)."""
    path = parquet_path or PROCESSED_PARQUET
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing processed parquet: {path}. Run src/preprocess.py first."
        )
    return pd.read_parquet(path, columns=["text_bm25", "doc_id"])

def build_bm25_retriever(df: pd.DataFrame | None = None) -> BM25Retriever:
    """Builds a BM25Retriever from text_bm25 with doc_id metadata."""
    frame = df if df is not None else load_processed_dataframe()
    texts = frame["text_bm25"].astype(str).tolist()
    metadatas = [{"doc_id": str(v)} for v in frame["doc_id"]]
    LOGGER.info("Building BM25 over %s documents", len(texts))
    return BM25Retriever.from_texts(texts, metadatas=metadatas, k=BM25_K)


def save_bm25_retriever(retriever: BM25Retriever, path: Path | None = None) -> None:
    """Pickles the retriever to disk."""
    out = path or BM25_INDEX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(retriever, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Saved BM25 retriever to %s", out)


def load_bm25_retriever(path: Path | None = None) -> BM25Retriever:
    """Loads a BM25 retriever from a pickle produced by save_bm25_retriever."""
    src = path or BM25_INDEX_PATH
    with open(src, "rb") as f:
        return pickle.load(f)


def build_and_save() -> BM25Retriever:
    """Loads parquet, fits BM25, persists pickle, returns the retriever."""
    retriever = build_bm25_retriever()
    save_bm25_retriever(retriever)
    return retriever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_and_save()


if __name__ == "__main__":
    main()
