# Smart Amazon Video Games Product Search

A retrieval-style search assistant over Amazon Video Games product data. Users submit natural language queries; the system retrieves relevant products using lexical search (BM25), dense semantic search (FAISS), or a hybrid of both — optionally with Claude-powered query expansion. After retrieval, you can **re-rank** the top candidates with a cross-encoder so the final list better matches what the user actually asked for.

<details>
<summary>Pipeline Architecture</summary>

![Pipeline Architecture](img/architecture.png)

</details>

---

## Setup

### 1. Download the raw data

Download the **Video Games** category files from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and place them here:

```mermaid
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
python src/bm25.py       # BM25 index  → data/context_store/bm25_retriever.pkl
python src/semantic.py   # FAISS index → data/context_store/faiss_index/
```

### 4. Run the app

```bash
streamlit run app/app.py
```

Add your `ANTHROPIC_API_KEY` to `.env` to enable query expansion (optional). The app can **re-rank** results with a cross-encoder (on by default); turn that off in the UI if you only want raw retrieval scores.

### 5. Quick CLI checks (same environment, indices already built)

You do not need Streamlit to exercise search. From the project root, after steps 3–4, you can run short end-to-end tests: retrieval pulls a **pool** of candidates (wider net), then the re-ranker keeps the top **`-k`** for printing.

```bash
# Dedicated re-rank entrypoint (retrieval + cross-encoder in one command)
python src/rerank.py "wireless PS5 controller" --mode hybrid --pool 10 -k 3 --no-expand

# Same pipeline via hybrid.py; --rerank turns on cross-encoder scoring after retrieval
python src/hybrid.py "your query" --mode hybrid --rerank --rerank-pool 10 -k 3 --no-expand
```

Use `--mode semantic` or `--mode bm25` to exercise a single channel. Drop `--no-expand` when you have an API key and want paraphrases. Omit `--rerank` on `hybrid.py` to see retrieval-only rankings.

---

## Scripts

| Script | Purpose | Input | Output |
|---|---|---|---|
| `src/preprocess.py` | ETL — merge raw JSONL into product parquet | `data/raw/*.jsonl` | `data/processed/merged_reviews.parquet` |
| `src/bm25.py` | Build BM25 index | `merged_reviews.parquet` | `data/context_store/bm25_retriever.pkl` |
| `src/semantic.py` | Build FAISS index | `merged_reviews.parquet` | `data/context_store/faiss_index/` |
| `src/hybrid.py` | Retrieval (BM25 / FAISS / hybrid RRF) and optional `--rerank` | FAISS + BM25 indices | `list[dict]` hits |
| `src/rerank.py` | Cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) + CLI | retriever hits | re-ordered hits |
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

### 7. Cross-encoder re-ranking

First-stage retrievers are fast but only approximate query–document match (BM25 counts words; the bi-encoder scores passage vectors against the query vector **separately**). A **cross-encoder** reads the query and each candidate passage **together** and outputs a single relevance score, which is closer to “does this passage answer this question?” — at the cost of more compute.

**In this project:** we retrieve a larger pool (default 15), then re-rank with `cross-encoder/ms-marco-MiniLM-L-6-v2` using the product’s `text_faiss` passage as `content`. The user’s **original** query is used for scoring (not each expansion), so paraphrases still help recall while the final order stays grounded in what they typed. The Streamlit app caches the cross-encoder so it is not reloaded on every click.

---

## Project Structure

```mermaid
├── data/
│   ├── raw/                        # downloaded JSONL files (not tracked)
│   ├── processed/
│   │   └── merged_reviews.parquet  # 137K product rows (built by preprocess.py)
│   └── context_store/
│       ├── bm25_retriever.pkl      # pickled BM25 index
│       └── faiss_index/            # FAISS index files
│
├── src/
│   ├── preprocess.py               # ETL
│   ├── bm25.py                     # BM25 index build
│   ├── semantic.py                 # FAISS index build + SemanticRetriever
│   ├── hybrid.py                   # RRF fusion + search + optional CLI re-rank
│   ├── rerank.py                   # Cross-encoder re-rank + CLI smoke tests
│   ├── query_expansion.py          # Claude Haiku query rewriting
│   └── documents.py                # DataFrame → LangChain Document helper
│
├── app/
│   └── app.py                      # Streamlit UI (retrieval + optional re-rank)
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

## RAG Retrieval Strategies Reference

### Only Lexical Search (e.g., BM25)

Best for exact keyword matching and precise term lookup.  
Performs well when queries contain identifiers, names, or domain specific vocabulary.  
Fast, simple, and does not require training.  

Intuition: Find documents that explicitly contain the query terms.

### Only Semantic Search (Embeddings)

Best for capturing meaning and contextual similarity.  
Handles synonyms, paraphrases, and natural language effectively.  
More flexible than keyword-based approaches.  

Intuition: Find documents that express similar ideas, even if wording differs.

### Hybrid Search (Lexical + Semantic)

Combines the strengths of lexical and semantic retrieval.  
Lexical ensures exact matches are not missed.  
Semantic captures conceptual similarity.  
Improves overall recall significantly.  

Intuition: Retrieve anything relevant, whether through exact terms or meaning.

### Query Rewriting (Pre-retrieval)

Improves the quality of the input query.  
Adds missing context, expands terms, or resolves ambiguity.  
Leads to better retrieval results across all methods.  

Intuition: Ask a clearer and more complete question before searching.

### Re-ranking (Cross-encoder)

Refines the ordering of retrieved candidates.  
Evaluates query and document jointly for deeper relevance.  
Reduces subtle mismatches and improves final context selection.  

Intuition: From a good set of candidates, select the most relevant ones.  
**In this repo:** implemented in `src/rerank.py`, wired from `src/hybrid.py` (`retrieve_with_expansion`) and the Streamlit app.

### Big Picture

Lexical search provides precision on exact terms.  
Semantic search captures meaning.  
Hybrid search improves recall.  
Query rewriting improves the input.  
Re-ranking improves the final output.  

Overall approach: retrieve broadly, then refine to select the best context.

## Contributors

- **Eli Gonzalez**
- **Jackson Lu**
