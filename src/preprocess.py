"""ETL: raw Amazon Video Games JSONL to processed parquet with dual text columns."""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

REV_DROP_COLS = ["images", "asin", "helpful_vote", "verified_purchase"]
META_DROP_COLS = ["bought_together", "subtitle", "author", "images", "videos"]

REVIEWS_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Video_Games.jsonl"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "meta_Video_Games.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "merged_reviews.parquet"

CHUNK_SIZE = 500_000
PARQUET_COMPRESSION = "snappy"
USE_SMALLER_SAMPLE = True
START_DATE = "2022-06-01"


def _safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drops columns that exist in df."""
    to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=to_drop) if to_drop else df


def _join_list(val: object, *, empty: str) -> str:
    """Concatenates list items into one string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return empty
    if not isinstance(val, list):
        return str(val)
    if len(val) == 0:
        return empty
    return " ".join(str(x) for x in val)


def _format_details(val: object) -> str:
    """Serializes a details dict to a string; keeps full structure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def format_merged_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies review and metadata level formatting to the merged frame."""
    out = df.copy()

    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        out["timestamp"] = ts.dt.strftime("%Y-%m-%d")

    if "price" in out.columns:
        out["price"] = out["price"].map(
            lambda x: "unknown"
            if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == ""
            else str(x).strip()
        )

    if "average_rating" in out.columns:
        out["average_rating"] = pd.to_numeric(out["average_rating"], errors="coerce")

    if "features" in out.columns:
        out["features"] = out["features"].map(
            lambda x: _join_list(x, empty="no features provided")
        )

    if "description" in out.columns:
        out["description"] = out["description"].map(
            lambda x: _join_list(x, empty="no description provided")
        )

    if "categories" in out.columns:
        out["categories"] = out["categories"].map(lambda x: _join_list(x, empty=""))

    if "details" in out.columns:
        out["details"] = out["details"].map(_format_details)

    return out


def _cell_str(val: object) -> str:
    """Converts a cell to str; missing values become empty string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def _build_text_bm25(row: pd.Series) -> str:
    """Keyword template for BM25 indexing."""
    parent = _cell_str(row.get("parent_asin", ""))
    main_cat = _cell_str(row.get("main_category", ""))
    store = _cell_str(row.get("store", ""))
    cats = _cell_str(row.get("categories", ""))
    product_title = _cell_str(row.get("product_post_title", ""))
    review_title = _cell_str(row.get("review_title", ""))
    body = _cell_str(row.get("text", ""))
    avg_r = _cell_str(row.get("average_rating", ""))
    n_rat = _cell_str(row.get("rating_number", ""))
    rev_r = _cell_str(row.get("rating", ""))
    price = _cell_str(row.get("price", ""))
    return (
        f"Product ID: {parent}\n"
        f"Main category: {main_cat}\n"
        f"Seller: {store}\n"
        f"Related categories: {cats}\n"
        f"Product title: {product_title}\n"
        f"Review title: {review_title}\n"
        f"Review: {body}\n"
        f"Average product rating: {avg_r}\n"
        f"Number of ratings: {n_rat}\n"
        f"Review star rating: {rev_r}\n"
        f"Price: {price}\n"
    )


def _build_text_faiss(row: pd.Series) -> str:
    """Short narrative text for dense retrieval."""
    product_title = _cell_str(row.get("product_post_title", ""))
    review_title = _cell_str(row.get("review_title", ""))
    body = _cell_str(row.get("text", ""))
    store = _cell_str(row.get("store", ""))
    cats = _cell_str(row.get("categories", ""))
    desc = _cell_str(row.get("description", ""))
    # details = _cell_str(row.get("details", ""))
    avg_r = _cell_str(row.get("average_rating", ""))
    n_rat = _cell_str(row.get("rating_number", ""))
    rev_r = _cell_str(row.get("rating", ""))
    price = _cell_str(row.get("price", ""))
    parts = [
        f"Product: {product_title}. Sold by {store}. Categories: {cats}.",
        f"Listing description: {desc}" if desc else "",
        # f"Details: {details}" if details else "",
        f"Aggregate rating {avg_r} from {n_rat} ratings; price {price}.",
        f"Review titled {review_title}: {body}" if review_title else f"Review: {body}",
        f"The reviewer gave {rev_r} stars.",
    ]
    return " ".join(p for p in parts if p)


def _metadata_json_blob(row: pd.Series) -> str:
    """Serializes all fields except text_bm25/text_faiss for LangChain Document.metadata."""
    data: dict[str, object] = {}
    for k, v in row.items():
        if k in ("text_bm25", "text_faiss"):
            continue
        if v is None or (isinstance(v, float) and pd.isna(v)):
            data[k] = None
        else:
            data[k] = v
    return json.dumps(data, ensure_ascii=False, default=str)


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds text columns and keeps doc_id as a first-class parquet column."""
    out = df.copy()
    out["text_bm25"] = out.apply(_build_text_bm25, axis=1)
    out["text_faiss"] = out.apply(_build_text_faiss, axis=1)
    out["details"] = out.apply(_metadata_json_blob, axis=1)
    return out[["doc_id", "text_bm25", "text_faiss", "details"]]


def assign_doc_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Adds stable doc_id per row."""

    def _row_id(r: pd.Series) -> str:
        base = (
            f"{r.get('parent_asin', '')}|{r.get('timestamp', '')}|"
            f"{r.get('review_title', '')}|{r.get('text', '')}"
        )
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]

    out = df.copy()
    out["doc_id"] = out.apply(_row_id, axis=1)
    return out


def load_metadata() -> pd.DataFrame:
    """Loads metadata JSONL once: drop columns and rename product title."""
    df_meta = pd.read_json(META_PATH, lines=True)
    df_meta = _safe_drop(df_meta, META_DROP_COLS)
    df_meta = df_meta.rename(columns={"title": "product_post_title"})
    LOGGER.info("Metadata dataset loaded (%s rows)", len(df_meta))
    return df_meta


def prepare_reviews_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns and renames review title column."""
    df = _safe_drop(df, REV_DROP_COLS)
    if USE_SMALLER_SAMPLE:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cond = df["timestamp"] > pd.to_datetime(START_DATE)
        df = df[cond]
    LOGGER.info("Filtered reviews to %s rows", len(df))
    return df.rename(columns={"title": "review_title"})


def process_merged_chunk(merged: pd.DataFrame) -> pd.DataFrame:
    """Formats columns, assigns doc_id, then text columns + packed metadata (`details`)."""
    merged = format_merged_columns(merged)
    merged = assign_doc_ids(merged)
    return add_text_columns(merged)


def run_etl() -> int:
    """Stream review chunks, merge with metadata, write one Parquet file. Returns total rows."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df_meta = load_metadata()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    # `writer`: open Parquet file. `ref_schema`: column types from chunk 1; later chunks cast.
    writer = None
    ref_schema = None
    total = 0

    try:
        chunks = pd.read_json(REVIEWS_PATH, lines=True, chunksize=CHUNK_SIZE)
        for i, chunk in enumerate(chunks):
            merged = df_meta.merge(
                prepare_reviews_chunk(chunk), on="parent_asin", how="left"
            )
            merged = process_merged_chunk(merged)
            table = pa.Table.from_pandas(merged, preserve_index=False)

            if writer is None:
                ref_schema = table.schema
                writer = pq.ParquetWriter(
                    OUTPUT_PATH, ref_schema, compression=PARQUET_COMPRESSION
                )
            else:
                table = table.cast(ref_schema)

            writer.write_table(table)
            total += len(merged)
            LOGGER.info("chunk %s: +%s rows (total %s)", i + 1, len(merged), total)
    finally:
        if writer:
            writer.close()

    if total == 0:
        raise RuntimeError("No rows written.")

    LOGGER.info("Wrote %s rows -> %s", total, OUTPUT_PATH)
    return total


if __name__ == "__main__":
    run_etl()
