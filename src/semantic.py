"""Semantic retriever using sentence-transformers + LangChain FAISS."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = Path("data/processed/faiss_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticRetriever:
    """Build and query a FAISS vector index via LangChain."""

    def __init__(self, index_dir: str | Path = INDEX_DIR) -> None:
        self.index_dir = Path(index_dir)
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._store: FAISS | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        return self._embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, corpus: list[str], df_meta: pd.DataFrame) -> None:
        """Build FAISS index from corpus and save to disk.

        Args:
            corpus: One document string per product (title + features + description).
            df_meta: Metadata DataFrame aligned with corpus (same order).
                     Must contain columns: parent_asin, title, average_rating,
                     rating_number.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)

        keep_cols = ["parent_asin", "title", "average_rating", "rating_number"]
        metadatas = df_meta[keep_cols].fillna("").to_dict(orient="records")

        embeddings = self._get_embeddings()
        store = FAISS.from_texts(texts=corpus, embedding=embeddings, metadatas=metadatas)
        store.save_local(str(self.index_dir))
        self._store = store
        print(f"FAISS index saved to {self.index_dir}")

    def load(self) -> None:
        """Load a previously saved FAISS index from disk."""
        embeddings = self._get_embeddings()
        self._store = FAISS.load_local(
            str(self.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k results for query.

        Returns:
            List of dicts with keys: parent_asin, title, average_rating,
            rating_number, score, rank.
        """
        if self._store is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")

        pairs = self._store.similarity_search_with_score(query, k=k)
        results = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            meta = doc.metadata.copy()
            meta["score"] = float(score)
            meta["rank"] = rank
            results.append(meta)
        return results


def main() -> None:
    """Build the FAISS semantic index from processed metadata parquet.

    Requires data/processed/metadata_clean.parquet to exist.
    Run utils.py first if it doesn't:
        python src/utils.py

    Run from project root:
        python src/semantic.py
    """
    import pandas as pd
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from src.utils import build_corpus

    meta_parquet = project_root / "data" / "processed" / "metadata_clean.parquet"
    index_dir = project_root / "data" / "processed" / "faiss_index"

    if not meta_parquet.exists():
        print(f"ERROR: {meta_parquet} not found.")
        print("Run `python src/utils.py` first to generate processed data.")
        sys.exit(1)

    print(f"Loading metadata from {meta_parquet} ...")
    df_meta = pd.read_parquet(meta_parquet)
    print(f"  Loaded {len(df_meta):,} products")

    print("Building corpus documents ...")
    corpus = build_corpus(df_meta)
    print(f"  Built {len(corpus):,} documents")

    print(f"Building FAISS index (model: {MODEL_NAME}) ...")
    print("  Embedding 137K documents — device selected automatically by sentence-transformers.")
    retriever = SemanticRetriever(index_dir=index_dir)
    retriever.build(corpus=corpus, df_meta=df_meta)

    print("\nDone. Index saved to:")
    print(f"  {index_dir / 'index.faiss'}")
    print(f"  {index_dir / 'index.pkl'}")


if __name__ == "__main__":
    main()
