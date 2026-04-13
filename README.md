# Smart Amazon Video Games Product Search

A retrieval-style search assistant over Amazon Video Games product data. Users submit natural language queries; the system retrieves relevant products using lexical search (BM25), dense semantic search (FAISS), or a hybrid of both — optionally with Claude-powered query expansion.

<details>
<summary>Pipeline Architecture</summary>

![Pipeline Architecture](img/architecture.png)

</details>

---

## Setup

### 1. Download the raw data

Download the **Video Games** category files from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and place them here:

```
data/raw/
├── Video_Games.jsonl          # 4.6M customer reviews
└── meta_Video_Games.jsonl     # 137K product listings
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate 575_jleg_rag_env
```

### 3. Build the pipeline

Run scripts from the project root in order:

```bash
# Step 1 — ETL: merge + aggregate → 137K product parquet
python src/preprocess.py

# Step 2 — Build indices (order does not matter, can run in parallel)
python src/bm25.py       # BM25 index  → context_store/bm25_retriever.pkl
python src/semantic.py   # FAISS index → context_store/faiss_index/
```

### 4. Run the app

```bash
streamlit run app/app.py
```

Add your `ANTHROPIC_API_KEY` to `.env` to enable query expansion (optional).

---

## Scripts

| Script | Purpose | Input | Output |
|---|---|---|---|
| `src/preprocess.py` | ETL — merge raw JSONL into product parquet | `data/raw/*.jsonl` | `data/processed/merged_reviews.parquet` |
| `src/bm25.py` | Build BM25 index | `merged_reviews.parquet` | `context_store/bm25_retriever.pkl` |
| `src/semantic.py` | Build FAISS index | `merged_reviews.parquet` | `context_store/faiss_index/` |
| `src/hybrid.py` | Hybrid retrieval with RRF fusion | FAISS + BM25 indices | `list[dict]` hits |
| `src/query_expansion.py` | Rewrite queries with Claude Haiku | user query | paraphrases |
| `src/documents.py` | Convert DataFrame rows to LangChain Documents | DataFrame | `list[Document]` |
| `app/app.py` | Streamlit search UI | indices | search results |

---

## Design Decisions

### 1. Product-level index (137K rows, not 4.6M)

The raw dataset has 4.6M reviews but only 137K unique products. Indexing at the review level creates 33× more documents with no retrieval benefit for product search — users search for products, not individual reviews.

**Decision:** aggregate the top-10 most helpful reviews per product (by `helpful_vote`) into a single text field, then join with product metadata. One row = one product.

**Trade-off:** individual review granularity is lost, but retrieval speed and index size improve by ~33×. The top-10 reviews still carry the most informative signal.

### 2. Two text columns per product

Each product has two purpose-built text representations:

- **`text_bm25`** — structured keyword template (`Product title: ... Features: ... Customer reviews: ...`). Designed for BM25's term-frequency scoring: field labels anchor important terms.
- **`text_faiss`** — short narrative prose (`Product: ... Sold by ... Rating: ... Customer reviews: ...`). Designed for dense embeddings: reads naturally so the sentence model produces meaningful vectors.

Both columns are built entirely in DuckDB SQL during preprocessing — no pandas row iteration.

### 3. DuckDB for ETL

All preprocessing runs inside a single SQL statement via DuckDB. This avoids loading 4.6M rows into pandas and eliminates slow `df.apply(axis=1)` calls. DuckDB executes the join, array flattening, aggregation, string building, and parquet write in one vectorised pass.

### 4. Hybrid retrieval with RRF

Search supports three modes:

- **Semantic** — FAISS L2 distance on `all-MiniLM-L6-v2` embeddings of `text_faiss`
- **BM25** — LangChain BM25Retriever over tokenised `text_bm25`
- **Hybrid** — Reciprocal Rank Fusion (RRF, k=60) merges BM25 and FAISS ranked lists per query, then across expanded queries

RRF was chosen over score normalisation because it is robust to score scale differences between retrievers and does not require tuning a mixing weight.

### 5. Query expansion with Claude Haiku

Before retrieval, Claude Haiku rewrites the user query into 3 paraphrases. Each paraphrase is retrieved independently; results are fused with RRF across queries. This improves recall for queries that use different vocabulary than the product text (e.g. "cheap controller" vs "budget gamepad").

Query expansion is optional — the app can run without an API key by unchecking the checkbox.

### 6. Metadata embedded in index documents

Each FAISS and BM25 document stores `parent_asin`, `product_title`, `average_rating`, `rating_number`, and `review_texts` in its metadata. Search results therefore carry all display fields without a secondary parquet lookup at query time.

---

## Project Structure

```
├── data/
│   ├── raw/                        # downloaded JSONL files (not tracked)
│   └── processed/
│       └── merged_reviews.parquet  # 137K product rows (built by preprocess.py)
│
├── context_store/
│   ├── bm25_retriever.pkl          # pickled BM25 index
│   └── faiss_index/                # FAISS index files
│
├── src/
│   ├── preprocess.py               # ETL
│   ├── bm25.py                     # BM25 index build
│   ├── semantic.py                 # FAISS index build + SemanticRetriever
│   ├── hybrid.py                   # RRF fusion + search entry point
│   ├── query_expansion.py          # Claude Haiku query rewriting
│   └── documents.py                # DataFrame → LangChain Document helper
│
├── app/
│   └── app.py                      # Streamlit UI
│
├── notebooks/
│   └── milestone1_exploration.ipynb
│
├── results/
│   └── milestone1_discussion.md
│
├── environment.yml
└── README.md
```

---

## Contributors

- **Eli Gonzalez**
- **Jackson Lu**
