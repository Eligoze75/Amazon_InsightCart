"""FAISS semantic index over merged reviews (text_faiss) via LangChain."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from documents import dataframe_to_documents

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PARQUET = _PROJECT_ROOT / "data" / "processed" / "merged_reviews.parquet"
FAISS_INDEX_DIR = _PROJECT_ROOT / "context_store" / "faiss_index"
# Same embedding model must be used for build and load; documented in README.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_processed_dataframe(parquet_path: Path | None = None) -> pd.DataFrame:
    """Loads only FAISS text plus stable id (no packed details JSON)."""
    path = parquet_path or PROCESSED_PARQUET
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing processed parquet: {path}. Run src/preprocess.py first."
        )
    return pd.read_parquet(path, columns=["text_faiss", "doc_id"])


def _make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


class SemanticRetriever:
    """Builds, persists, and queries a FAISS store over text_faiss Documents."""

    def __init__(self, index_dir: Path | str | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir is not None else FAISS_INDEX_DIR
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._store: FAISS | None = None

    def _embeddings_model(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = _make_embeddings()
        return self._embeddings

    def build(self, df: pd.DataFrame | None = None) -> FAISS:
        """Builds FAISS from merged rows; page_content is text_faiss, metadata aligned."""
        frame = df if df is not None else load_processed_dataframe()
        documents = dataframe_to_documents(frame, "text_faiss")
        LOGGER.info("Embedding %s documents into FAISS", len(documents))
        embeddings = self._embeddings_model()
        store = FAISS.from_documents(documents, embedding=embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_dir))
        self._store = store
        LOGGER.info("FAISS index saved to %s", self.index_dir)
        return store

    def load(self) -> None:
        """Loads a FAISS index written by build()."""
        embeddings = self._embeddings_model()
        self._store = FAISS.load_local(
            str(self.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Returns top-k matches with merged-schema metadata plus score and rank.

        Metadata includes doc_id, parent_asin, product/review fields from preprocess,
        as stored on each Document (see documents.dataframe_to_documents).

        Args:
            query: Query text to embed and search.
            k: Number of neighbors.

        Returns:
            List of dicts: document metadata keys, plus 'score' and 'rank' (1-based).
        """
        if self._store is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")
        pairs = self._store.similarity_search_with_score(query, k=k)
        results: list[dict] = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            row = dict(doc.metadata)
            row["content"] = doc.page_content
            row["score"] = float(score)
            row["rank"] = rank
            results.append(row)
        return results


def build_and_save(parquet_path: Path | None = None) -> FAISS:
    """Loads parquet, builds FAISS on text_faiss, saves under data/context_store/faiss_index/."""
    df = load_processed_dataframe(parquet_path)
    retriever = SemanticRetriever()
    return retriever.build(df)


def main() -> None:
    """Build the FAISS index from merged_reviews.parquet.

    Requires data/processed/merged_reviews.parquet (run src/preprocess.py first).

    From project root:
        python src/semantic.py
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not PROCESSED_PARQUET.is_file():
        print(f"ERROR: {PROCESSED_PARQUET} not found.", file=sys.stderr)
        print(
            "Run `python src/preprocess.py` first to generate merged_reviews.parquet.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {PROCESSED_PARQUET} ...")
    df = load_processed_dataframe()
    print(f"  Loaded {len(df):,} rows")
    print(f"Building FAISS index (embeddings: {EMBEDDING_MODEL_NAME}) ...")
    build_and_save()
    print("\nDone. Index saved under:")
    print(f"  {FAISS_INDEX_DIR}")


if __name__ == "__main__":
    main()
