"""Build LangChain Documents from processed review DataFrames."""

from __future__ import annotations
from typing import Literal
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document

# Columns used as page_content; others are omitted from metadata to avoid duplication.
_TEXT_COLUMNS = ["text_bm25", "text_faiss"]


def _dataframe_for_loader(df: pd.DataFrame, page_content_column: str) -> pd.DataFrame:
    """Returns a copy with other text columns dropped so metadata stays small."""
    to_drop = [c for c in _TEXT_COLUMNS if c in df.columns and c != page_content_column]
    out = df.drop(columns=to_drop, errors="ignore")
    return out


def dataframe_to_documents(
    df: pd.DataFrame,
    page_content_column: Literal["text_bm25", "text_faiss"],
) -> list[Document]:
    """Loads rows as Documents; page_content is one text column, rest is metadata."""
    subset = _dataframe_for_loader(df, page_content_column)
    loader = DataFrameLoader(subset, page_content_column=page_content_column)
    return loader.load()


def build_dual_corpora(df: pd.DataFrame) -> tuple[list[Document], list[Document]]:
    """Builds BM25 and FAISS document lists from the same processed frame."""
    return (
        dataframe_to_documents(df, "text_bm25"),
        dataframe_to_documents(df, "text_faiss"),
    )
