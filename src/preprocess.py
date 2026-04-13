"""ETL: raw Amazon JSONL → product-level parquet (137K rows) via DuckDB.

One row per product: metadata joined with top-N most helpful customer reviews
aggregated into a single text field. Output consumed by bm25.py and semantic.py.

Run from project root:
    python src/preprocess.py
"""

from __future__ import annotations
import logging
import time
from pathlib import Path

import duckdb

LOGGER = logging.getLogger(__name__)

_ROOT         = Path(__file__).resolve().parent.parent
REVIEWS_PATH  = _ROOT / "data" / "raw" / "Video_Games.jsonl"
META_PATH     = _ROOT / "data" / "raw" / "meta_Video_Games.jsonl"
OUTPUT_PATH   = _ROOT / "data" / "processed" / "merged_reviews.parquet"
TOP_N_REVIEWS = 10  # most helpful reviews to aggregate per product


def run_etl(
    reviews_path: Path = REVIEWS_PATH,
    meta_path: Path = META_PATH,
    output_path: Path = OUTPUT_PATH,
    top_n: int = TOP_N_REVIEWS,
) -> None:
    """Merge raw JSONL files into a product-level parquet via DuckDB.

    Steps:
      1. Rank each product's reviews by helpful_vote (desc).
      2. Aggregate the top-N review texts per product.
      3. Left-join aggregated reviews onto metadata (one row per product).
      4. Build text_bm25 (structured keyword template) and text_faiss
         (dense narrative) columns entirely in SQL.
      5. Write to parquet — no pandas in the hot path.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    n_rev = con.execute(
        f"SELECT COUNT(*) FROM read_json_auto('{reviews_path}', maximum_object_size=33554432)"
    ).fetchone()[0]
    LOGGER.info("Reviews:  %s rows", n_rev)

    n_meta = con.execute(
        f"SELECT COUNT(*) FROM read_json_auto('{meta_path}', maximum_object_size=33554432)"
    ).fetchone()[0]
    LOGGER.info("Metadata: %s products", n_meta)

    LOGGER.info("Aggregating and writing parquet ...")
    con.execute(f"""
    COPY (
        WITH
        reviews_ranked AS (
            -- rank each product's reviews by helpfulness
            SELECT
                parent_asin,
                text        AS review_text,
                rating,
                ROW_NUMBER() OVER (
                    PARTITION BY parent_asin
                    ORDER BY COALESCE(helpful_vote, 0) DESC
                ) AS rn
            FROM read_json_auto('{reviews_path}', maximum_object_size=33554432)
            WHERE text IS NOT NULL AND TRIM(text) != ''
        ),
        reviews_agg AS (
            -- keep only top-N reviews per product
            SELECT
                parent_asin,
                string_agg(review_text, ' ')    AS review_texts,
                ROUND(AVG(rating), 2)            AS avg_review_rating
            FROM reviews_ranked
            WHERE rn <= {top_n}
            GROUP BY parent_asin
        ),
        meta AS (
            SELECT
                parent_asin,
                title                                                                       AS product_title,
                main_category,
                CAST(average_rating AS DOUBLE)                                              AS average_rating,
                rating_number,
                store,
                COALESCE(NULLIF(TRIM(CAST(price AS VARCHAR)), ''), 'unknown')               AS price,
                COALESCE(NULLIF(array_to_string(features, ' '), ''), 'no features listed')  AS features,
                COALESCE(NULLIF(array_to_string(description, ' '), ''), '')                 AS description,
                COALESCE(array_to_string(categories, ' '), '')                              AS categories,
                CAST(details AS VARCHAR)                                                    AS details
            FROM read_json_auto('{meta_path}', maximum_object_size=33554432)
        ),
        merged AS (
            SELECT
                m.*,
                COALESCE(r.review_texts, '')    AS review_texts,
                r.avg_review_rating
            FROM meta m
            LEFT JOIN reviews_agg r USING (parent_asin)
        ),
        with_texts AS (
            SELECT
                *,
                -- BM25: structured keyword template for lexical matching
                'Product ID: '        || parent_asin                                        || chr(10) ||
                'Category: '          || COALESCE(main_category, '')                        || chr(10) ||
                'Seller: '            || COALESCE(store, '')                                || chr(10) ||
                'Categories: '        || categories                                         || chr(10) ||
                'Product title: '     || product_title                                      || chr(10) ||
                'Features: '          || features                                           || chr(10) ||
                'Description: '       || description                                        || chr(10) ||
                'Average rating: '    || COALESCE(CAST(average_rating AS VARCHAR), '')      || chr(10) ||
                'Number of ratings: ' || COALESCE(CAST(rating_number  AS VARCHAR), '')      || chr(10) ||
                'Price: '             || price                                              || chr(10) ||
                'Customer reviews: '  || review_texts                                       || chr(10)
                                                                        AS text_bm25,
                -- FAISS: short narrative for dense embedding
                'Product: '           || product_title ||
                '. Sold by '          || COALESCE(store, '') ||
                '. Categories: '      || categories || '. ' ||
                CASE WHEN description != ''
                    THEN 'Description: ' || description || ' ' ELSE '' END ||
                'Features: '          || features || '. ' ||
                'Rating: '            || COALESCE(CAST(average_rating AS VARCHAR), '') ||
                ' from '              || COALESCE(CAST(rating_number  AS VARCHAR), '') ||
                ' ratings. Price: '   || price || '. ' ||
                'Customer reviews: '  || review_texts
                                                                        AS text_faiss
            FROM merged
        )
        SELECT
            md5(parent_asin)    AS doc_id,
            parent_asin,
            product_title,
            main_category,
            average_rating,
            rating_number,
            store,
            price,
            review_texts,
            avg_review_rating,
            text_bm25,
            text_faiss,
            details
        FROM with_texts
    ) TO '{output_path}' (FORMAT PARQUET);
    """)

    n_out = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    con.close()
    LOGGER.info("Wrote %s products → %s", n_out, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = time.perf_counter()
    run_etl()
    LOGGER.info("Finished in %.1f minutes", (time.perf_counter() - start) / 60)
