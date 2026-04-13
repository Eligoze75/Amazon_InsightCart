"""Build and query a FAISS semantic index over product embeddings.

Run from project root:
    python src/semantic.py
"""

from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure project root is on path so ``from src.*`` works when run directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.documents import dataframe_to_documents  # noqa: E402

LOGGER = logging.getLogger(__name__)

PROCESSED_PARQUET  = _ROOT / "data" / "processed" / "merged_reviews.parquet"
FAISS_INDEX_DIR    = _ROOT / "data" / "context_store" / "faiss_index"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

# Columns stored in each Document's metadata — returned by search() and used
# by the app to display results without an extra parquet lookup.
_META_COLS = [
    "doc_id", "parent_asin", "product_title",
    "average_rating", "rating_number", "review_texts",
]


def load_processed_dataframe(parquet_path: Path | None = None) -> pd.DataFrame:
    """Load text_faiss and display metadata columns from the product parquet."""
    path = parquet_path or PROCESSED_PARQUET
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing processed parquet: {path}. Run src/preprocess.py first."
        )
    return pd.read_parquet(path, columns=["text_faiss"] + _META_COLS)


def _make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"batch_size": 128},
        show_progress=True,
    )


class SemanticRetriever:
    """Builds, persists, and queries a FAISS store over text_faiss documents."""

    def __init__(self, index_dir: Path | str | None = None) -> None:
        self.index_dir   = Path(index_dir) if index_dir is not None else FAISS_INDEX_DIR
        self._embeddings = None
        self._store      = None

    def _embeddings_model(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = _make_embeddings()
        return self._embeddings

    def build(self, df: pd.DataFrame | None = None) -> FAISS:
        """Embed text_faiss documents and save the FAISS index to disk."""
        frame     = df if df is not None else load_processed_dataframe()
        documents = dataframe_to_documents(frame, "text_faiss")
        LOGGER.info("Embedding %s documents into FAISS", len(documents))
        store = FAISS.from_documents(documents, embedding=self._embeddings_model())
        self.index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_dir))
        self._store = store
        LOGGER.info("FAISS index saved → %s", self.index_dir)
        return store

    def load(self) -> None:
        """Load a FAISS index written by build()."""
        self._store = FAISS.load_local(
            str(self.index_dir),
            self._embeddings_model(),
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k results with metadata, score, and rank."""
        if self._store is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")
        pairs = self._store.similarity_search_with_score(query, k=k)
        results = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            row          = dict(doc.metadata)
            row["content"] = doc.page_content
            row["score"]   = float(score)
            row["rank"]    = rank
            results.append(row)
        return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not PROCESSED_PARQUET.is_file():
        LOGGER.error("%s not found. Run src/preprocess.py first.", PROCESSED_PARQUET)
        sys.exit(1)

    df = load_processed_dataframe()
    LOGGER.info("Loaded %s products", len(df))
    SemanticRetriever().build(df)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    LOGGER.info("Finished in %.1f minutes", (time.perf_counter() - start) / 60)
