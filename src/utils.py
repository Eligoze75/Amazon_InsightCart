"""Auxiliary ETL and helpers for legacy app artifacts.

Produces ``metadata_clean.parquet`` and ``reviews.parquet`` when run as a script;
the Streamlit app uses ``lookup_reviews`` against ``reviews.parquet``. The
canonical merged table for BM25/FAISS is built by ``preprocess.py``
(``merged_reviews.parquet``).
"""

import duckdb
import pandas as pd
from pathlib import Path

# Paths (relative to project root — callers must set cwd or use absolute)
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
META_PATH = RAW_DIR / "meta_Video_Games.jsonl"
REVIEWS_PATH = RAW_DIR / "Video_Games.jsonl"
META_PARQUET = PROCESSED_DIR / "metadata_clean.parquet"
REVIEWS_PARQUET = PROCESSED_DIR / "reviews.parquet"


def load_metadata(path: str | Path = META_PATH) -> pd.DataFrame:
    """Load product metadata via DuckDB, selecting only retrieval-relevant columns."""
    path = str(path)
    return duckdb.sql(
        f"""
        SELECT
            parent_asin,
            title,
            features,
            description,
            CAST(average_rating AS FLOAT)   AS average_rating,
            CAST(rating_number  AS INTEGER) AS rating_number
        FROM read_json_auto('{path}', maximum_object_size=33554432)
    """
    ).df()


def save_reviews_parquet(
    reviews_path: str | Path = REVIEWS_PATH,
    output_path: str | Path = REVIEWS_PARQUET,
) -> None:
    """Convert the raw reviews JSONL to a parquet file with only needed columns.

    Keeps all reviews (no aggregation). DuckDB queries this parquet at
    search time to fetch reviews for matched products on demand.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    duckdb.sql(
        f"""
        COPY (
            SELECT parent_asin, title AS review_title, text AS review_text,
                   rating, helpful_vote
            FROM read_json_auto('{str(reviews_path)}', maximum_object_size=33554432)
        ) TO '{str(output_path)}' (FORMAT PARQUET)
    """
    )


def lookup_reviews(
    parent_asins: list[str],
    parquet_path: str | Path = REVIEWS_PARQUET,
) -> pd.DataFrame:
    """Fetch the best review per product for a small set of matched products.

    Queries reviews.parquet at runtime via DuckDB — only rows for the matched
    parent_asins are loaded. Picks the most helpful review per product.
    Returns DataFrame with columns: parent_asin, review_title, review_text, rating.
    """
    if not parent_asins:
        return pd.DataFrame(
            columns=["parent_asin", "review_title", "review_text", "rating"]
        )

    placeholders = ", ".join(f"'{a}'" for a in parent_asins)
    return duckdb.sql(
        f"""
        SELECT
            parent_asin,
            FIRST(review_title ORDER BY helpful_vote DESC, rating DESC) AS review_title,
            FIRST(review_text  ORDER BY helpful_vote DESC, rating DESC) AS review_text,
            FIRST(rating       ORDER BY helpful_vote DESC, rating DESC) AS rating
        FROM read_parquet('{str(parquet_path)}')
        WHERE parent_asin IN ({placeholders})
        GROUP BY parent_asin
    """
    ).df()


def main() -> None:
    """Write metadata_clean.parquet and reviews.parquet from raw JSONL.

    Optional alongside ``preprocess.py``; needed if you use ``lookup_reviews``
    with the default ``reviews.parquet`` path.

    Run from the project root:
        python src/utils.py
    """
    # Resolve paths relative to project root (parent of src/)
    project_root = Path(__file__).resolve().parent.parent
    meta_path = project_root / "data" / "raw" / "meta_Video_Games.jsonl"
    reviews_path = project_root / "data" / "raw" / "Video_Games.jsonl"
    processed_dir = project_root / "data" / "processed"
    meta_parquet = processed_dir / "metadata_clean.parquet"
    reviews_parquet = processed_dir / "reviews.parquet"

    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- Metadata ---
    print(f"Loading metadata from {meta_path} ...")
    df_meta = load_metadata(meta_path)
    df_meta.to_parquet(meta_parquet, index=False)
    print(f"  Saved {len(df_meta):,} products → {meta_parquet}")

    # --- Reviews ---
    print(f"Converting reviews from {reviews_path} ...")
    print("  (Streaming 4.6M rows via DuckDB — may take a minute)")
    save_reviews_parquet(reviews_path=reviews_path, output_path=reviews_parquet)
    print(f"  Saved → {reviews_parquet}")

    print("\nDone. Processed files:")
    print(f"  {meta_parquet}")
    print(f"  {reviews_parquet}")


if __name__ == "__main__":
    main()
