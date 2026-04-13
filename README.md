# Smart Amazon Product Query Assistant (Video Games)

Welcome! This repository is part of a UBC MDS project that builds a retrieval style assistant for **Amazon video game products**. The goal is to combine review text and product metadata so that queries can be answered with relevant, grounded context. The pipeline uses query expansion, lexical search with BM25, dense search with FAISS, and reciprocal rank fusion to merge results in a sensible way.

![Pipeline Architecture](img/architecture.png)

## Objective

We are building a RAG oriented workflow: users makes questions about video games, we retrieve supporting passages from processed reviews and metadata, and later stages can use that context with a language model. This milestone focuses on data preparation, offline indices, hybrid retrieval, and the app MVP. The scope is **only the Video Games category** from the public Amazon Reviews 2023 release.

## Data you need

Download the **reviews** and **metadata** JSONL files for the category you care about from the official site:

[https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)

For this project, use the **Video Games** files and place them here:

* `data/raw/Video_Games.jsonl` (reviews)
* `data/raw/meta_Video_Games.jsonl` (metadata)

File names must match what `src/preprocess.py` expects. If you are new to the dataset, the site also links the paper and Hugging Face options for loading data.

## Faster runs for testing

ETL can take a long time on the full reviews file. In `src/preprocess.py`, the flag `USE_SMALLER_SAMPLE` is available. When it is `True`, reviews are filtered to timestamps **after** `START_DATE` (see the same file), which keeps runs smaller and faster for quick experiments. Set `USE_SMALLER_SAMPLE` to `False` when you want the full filtered pipeline for serious indexing.

## Conda environment

Create and activate the environment from the repo root:

```bash
conda env create -f environment.yml
conda activate 575_jleg_rag_env
```

This installs Python 3.12, pandas, PyArrow, PyTorch (conda), and pip packages for LangChain, FAISS, sentence transformers, BM25, Anthropic, DuckDB, Jupyter, and more.

-> Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`.

## Where we are today

**Done or in place**

* ETL from raw JSONL to `data/processed/merged_reviews.parquet` with `doc_id`, `text_bm25`, and `text_faiss`
* LangChain document helpers in `src/documents.py`
* BM25 index build and pickle in `src/bm25.py`
* FAISS index build in `src/semantic.py`
* Query expansion with Claude Haiku in `src/query_expansion.py`
* Hybrid retrieval with RRF in `src/hybrid.py` (BM25 and FAISS lists fused per query, then across expanded queries)

**What are we planning for next milestone**

* Cross encoder reranking
* A single `retrieve_context` bundle API module (planned next)
* Final answer generation with a LLM

## Local test: hybrid search

After preprocessing and building both indices, run a hybrid retrieval from the project root. This uses the merged table to attach the canonical `text_faiss` passage to each hit.

1. Process data (adjust `USE_SMALLER_SAMPLE` if needed):

   ```bash
   python src/preprocess.py
   ```

2. Build BM25 and FAISS (order can be either):

   ```bash
   python src/bm25.py
   python src/semantic.py
   ```

3. Run the hybrid CLI (example without query expansion so no API key is required):

   ```bash
   python src/hybrid.py "best Nintendo console under 300" --mode hybrid --top-k 5 --no-expand
   ```

   If you want Claude to rewrite the query first, run without the flag that skips expansion and put your API key in `.env`.

You should see printed queries used, ranks, scores, and short **content** snippets from retrieval.

## Exploration and EDA

For charts, schema notes, and exploratory context on the merged data, take a look to `notebooks/milestone1_exploration.ipynb`. It is a good companion when you interpret retrieval behavior or tune preprocessing.

## Dependencies (summary)

Core stack: **Python 3.12**, **pandas**, **PyArrow**, **NumPy**, **SciPy**, **PyTorch** (conda), **matplotlib**, **seaborn**, **spaCy**, **JupyterLab**.

Retrieval and NLP (pip): **langchain community**, **langchain core**, **langchain text splitters**, **langchain huggingface**, **rank_bm25**, **sentence transformers**, **transformers**, **faiss cpu**, **anthropic**, **duckdb**, **nltk**, **datasets**, **python dotenv**.

## Next milestone (preview)

The next steps aim to add **cross encoder reranking** on a bounded candidate pool, then a small **pipeline** module that returns a structured context bundle (original query, expanded queries, top passages with metadata) ready for an answer model. Final answer generation may remain out of scope until a later milestone.

## Contributors

* **Eli Gonzalez**
* **Jackson Lu**
