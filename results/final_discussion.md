# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

- **Number of products used:** 137,269 unique products
- **Sampling strategy:** No sampling, the full Amazon Video Games metadata catalogue is used. The raw dataset contains 4.6 million customer reviews across those 137K products. Preprocessing aggregates the top-10 most helpful reviews per product (by `helpful_vote`) into a single row, so the index operates at product granularity rather than review granularity.
- **Index rebuild:** Both the BM25 index (`data/context_store/bm25_retriever.pkl`) and the FAISS index (`data/context_store/faiss_index/`) were built over the full 137K-product parquet. No changes to the indexing scripts were required, the pipeline already processed the entire dataset from Milestone 1.

---

### LLM Experiment

Our default model till now has been Claude Haiku, Anthropic's fastest and most cost effective model so far. We've heard so many wonders about Opus that we wanted to put it to the test. Claude Opus is Anthropic’s most intelligent model, designed for complex reasoning, coding, and high level analysis.

So, basically we want to know:

- Will Opus live up to the hype?
- For this task, is there a significant improvement on results when using Opus instead of Haiku?
- How much slower will the response be? Since this is a RAG with multiple steps that take time, we want to test if adding a heavy thinking model will increase latency even more

We'll use the same queries we documented in `results/milestone2_discussion.md`.

**Models compared:**

| Model | Family | Size/tier | Role |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | Small / fast | Current default in pipeline |
| `claude-opus-4-7` | Claude Opus 4.7 | High complexity reasoning / slow | Challenger |
|  |  |  |  |

#### Tests

1. **`wireless PS5 controller`**

**Retrieval snapshot (titles or notes):**

- Wireless Keyboard for PS5 Controller, arVin Rechargeable Bluetooth Keypad Gamepad Online Gaming Live Chat Message Chatpad Keyboard with Built-in Speaker & 3.5MM Audio Jack for Playstation 5 Dualsense
- Custom UN-MODDED Wireless PRO Controller compatible with PS5 DualSense Exclusive Unique Design (Mask)
- SADES PS5 Controller Charging Station Wireless Charging Receiver Adapter Compatible With PS5 Controller Gamepad,Fast Charger PS5 Wireless Handle Double Seat Charge

**Haiku's Answer:**

*I found a couple of options for you! If you're looking for an actual PS5 controller, there's a Custom UN-MODDED Wireless PRO Controller compatible with PS5 DualSense that has an excellent rating of 4.6/5 stars. Alternatively, if you need charging solutions for your PS5 controller, the SADES PS5 Controller Charging Station offers wireless charging capability with a double seat design for convenient charging.*

**Opus's Answer:**

*Here are some great wireless options for your PS5 controller! The Custom UN-MODDED Wireless PRO Controller (4.6/5) is a standout DualSense-compatible controller with an exclusive Mask design. You might also like the arVin Wireless Bluetooth Keypad for live chat and messaging, or the SADES Wireless Charging Station to keep your controllers powered up.*

**Comments**  

I kind of see the difference. Opus looks like taking the role more seriously, like being more convincing and appealing. At the same time, it made use of all the retrieved options, whereas Haiku only suggested two of them. In this case I would prefer Opus's answer.

---

**2. `best god of war`**

**Retrieval snapshot (titles or notes):**

- God of War: Collection Full Game - PS Vita [Digital Code]
- God of War 3 Favoritos - Spanish/English Edition - PlayStation 3
- God of War 2 - PlayStation 2

**Haiku's Answer:**

*Based on the available options, God of War 2 for PlayStation 2 stands out as an excellent choice with a 4.5/5 rating—the highest among these titles. It continues Kratos's epic saga as he sits atop the Olympus throne and battles untold horrors across the far reaches of the earth. If you're looking for a complete collection experience, the God of War: Collection for PS Vita offers both the original God of War and God of War II in one package, bringing the classic adventures of ancient Greece to a portable system.*

**Opus's Answer:**

*Based on ratings, God of War 2 for PlayStation 2 is the top pick, scoring an impressive 4.5/5. In this epic adventure, Kratos takes his throne as the new God of War and embarks on a brutal journey to the far reaches of the earth to end his torment. It's a must-play for fans of action-packed mythological storytelling!*

**Comments**  

I liked that Haiku suggested more options to complement the purchase. In a way, is like giving the user a wider view of what they could prefer or find. On the other hand, Opus focused on exactly what was searched. In this case I guess that each model wins on different aspects: Haiku gave a more complete answer, offering different products. Opus focused on the best option only.

---

**3. `party game for Switch`**

**Retrieval snapshot (titles or notes):**

- Vegas Party (Nintendo Switch)
- Sports Party - Nintendo Switch Standard Edition
- 30 In 1 Game Collection Vol 1 (Nintendo Switch)

**Haiku's Answer:**

*Great choice! For party games on Switch, I'd recommend Sports Party (rated 4.3/5), which lets you challenge friends and family to fun sports like Frisbee, basketball, golf, and beach tennis. If you want more variety, the 30 In 1 Game Collection Vol 1 (also 4.3/5) offers 30 different pick-up-and-play games perfect for groups, including classics and skill-based challenges. Both are excellent for multiplayer fun!*

**Opus's Answer:**

*Here are some great party games for your Nintendo Switch! Sports Party (4.3/5) is a top pick, offering six sports like Frisbee, basketball, and beach tennis that you can enjoy with family and friends. For variety, the 30 In 1 Game Collection Vol 1 (4.3/5) packs tons of quick pick-up-and-play mini-games, while Vegas Party brings a fun, competitive Las Vegas-themed race with 10 gambling minigames.*

**Comments**  

In this case the answer is actually very very similar. The difference is mainly at the end. Since the output is basically indifferent, then I would prefer Haiku's.

---

**4. `RPG under $30`**

**Retrieval snapshot (titles or notes):**

- Dark Cloud (Renewed) ($30)
- Codex: Death Guard (Hb) (English) ($40)
- Advanced Dungeons & Dragons: Masterpiece Collection

**Haiku's Answer:**

*Based on the products available, I'd recommend Dark Cloud (Renewed) for an RPG under $30. This PlayStation 2 game combines RPG action with real-time strategy and features an innovative Georama system that lets you build customized worlds. It has excellent ratings (4.1/5) and offers imaginative, vibrant gameplay at an affordable price point.*

**Opus's Answer:**

*For an RPG under $30, I'd recommend Dark Cloud (Renewed) for PlayStation 2. It's a highly-rated (4.1/5) cross-genre RPG featuring real-time strategy, an engaging storyline, and the innovative Georama world-building feature that lets you customize and interact with your environments.*

**Comments**  

I like how Opus phrases the pitch. Instead of talking about the "available products" it simply talks about the products and the reviews. The answer was also more concise again.

### 

---

**5. `PS5 controller drift`**

**Retrieval snapshot (titles or notes):**

- 1 Pair Controller Analog Stick Drift Fix Mod for Xbox One PS4 PS5 Switch Pro Gamepad Game Handle Joystick Drift Repair Board (Black)
- 2 x Analog Stick Drift Fix Mod Handle Rocker Adjusting Plate Game Handle Joystick Drift Repair Board for PS4 PS5 Xbox One Xbox Series X S Controller (Black)
- 10pcs Gamepad Analog Stick Drift Fix Mod, Universal 3D Analog Joystick Drift Fix Mod, Black Easy Using Gamepad Joystick Drift Repair Board, for Game Controllers

**Haiku's Answer:**

*I found several great options to fix PS5 controller drift! You can choose from analog stick drift fix mod boards that are compatible with PS5 controllers. The most highly-rated option is the 10pcs Gamepad Analog Stick Drift Fix Mod (4.5/5 stars) by SHYEKYO, which offers universal compatibility and easy installation. Alternatively, there are 2-piece repair kits like those from LightFairy and Limentea (both 3.3/5 stars) that include replacement joystick sticks and adjustment plates to correct the drift issue.*

**Opus's Answer:**

**Evaluation:**

So, it's an interesting answer. Claude understood the query in a very good way. The query is related to issues with a product, and Claude offered products that help with that issue. I think Claude might be performing as an actual salesman.

#### Conclusions

In short, for this task, Opus gives slightly better answers. It felt more convincing, more focused, and more concise. Latency was pretty similar, nothing noticeably different.

That said, the gap isn’t big enough to justify choosing Opus. It’s better, sure, but not by much. And since Haiku is significantly cheaper, it makes more sense to go with Haiku.

Honestly, for this kind of task, using a model as heavy as Opus feels like overkill. Its full potential just isn’t really being put to use here.

---

## Step 2: Additional Feature — Option 3: Scale to >100k Products

### What We Implemented

**We chose Option 3 (Scale to >100,000 Products).** Our pipeline processes the full Amazon Video Games catalogue: **137,269 unique products** derived from 4.6 million customer reviews.

**Engineering approach:**

The key challenge at this scale is avoiding slow row-by-row operations in Python. We solved this by moving all ETL into a single DuckDB SQL statement (`src/preprocess.py`). The query reads both raw JSONL files (reviews and metadata), joins them on `parent_asin`, aggregates the top-10 most helpful reviews per product using a window function, builds two purpose-built text fields (`text_bm25` and `text_faiss`) via string concatenation, and writes the result to a Parquet file — all without loading 4.6M rows into pandas memory.

```python
# src/preprocess.py — single-pass DuckDB ETL over 4.6M reviews → 137K product parquet
con.execute("""
    COPY (
        SELECT
            m.parent_asin,
            m.title   AS product_title,
            ...
            string_agg(r.text, ' | ' ORDER BY r.helpful_vote DESC) FILTER (
                WHERE r.rn <= 10
            ) AS review_texts,
            ...
        FROM meta m
        LEFT JOIN (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY parent_asin ORDER BY helpful_vote DESC
            ) AS rn FROM reviews
        ) r ON m.parent_asin = r.parent_asin
        GROUP BY ...
    ) TO 'data/processed/merged_reviews.parquet' (FORMAT PARQUET)
""")
```

**Index sizes at 137K scale:**

| Index | Format | Notes |
|---|---|---|
| FAISS | `IndexFlatL2`, `all-MiniLM-L6-v2` embeddings (384-dim) | ~200MB on disk |
| BM25 | LangChain `BM25Retriever`, pickled | ~150MB on disk |

**Why naive approaches break down at this scale:**

- Pandas `apply(axis=1)` for text construction over 4.6M rows would take ~30 min; DuckDB does it in seconds.
- Loading all 4.6M review texts into memory at once would require >16GB RAM; DuckDB streams from disk.
- Embedding all 137K products with `sentence-transformers` requires batched GPU/CPU inference — handled by `semantic.py` via the model's `encode(..., batch_size=64)` call.

All code is in `src/preprocess.py`, `src/bm25.py`, and `src/semantic.py`. The Streamlit app (`app/app.py`) loads both indices once at startup via `@st.cache_resource` and serves all queries from memory.

---

## Step 3: Improve Documentation and Code Quality

### Documentation Update

**README improvements for the final submission:**

- Added full pipeline architecture overview with component descriptions
- Added detailed Design Decisions section (8 decisions, each with trade-off discussion): product-level indexing, dual text columns, DuckDB ETL, hybrid RRF retrieval, Claude Haiku query expansion, metadata embedded in documents, cross-encoder re-ranking, optional grounded answer
- Added RAG Retrieval Strategies Reference section summarising intuitions for each retrieval method
- Added Contributors section
- Setup instructions cover all three phases: data download, environment, index building, app launch, and CLI smoke tests

### Code Quality Changes

**Checklist status:**

| Item | Status |
|---|---|
| No hardcoded file paths (use `pathlib.Path`) | Done — all paths use `Path` constants (e.g. `BM25_INDEX_PATH`, `FAISS_INDEX_DIR` in each module) |
| No API keys in source code | Done — `ANTHROPIC_API_KEY` loaded from `.env` via `python-dotenv`; `.env.example` provided |
| All functions include docstrings | Done — every public function has at least a one-line docstring |
| Environment file is up-to-date | Done — `environment.yml` pins all dependencies |
| No temporary files / large artifacts | Done — `data/raw/`, `data/processed/`, `data/context_store/` all listed in `.gitignore`; no large binaries tracked |

**Additional cleanups:**

- Removed unused imports across `src/` modules
- `generate_answer` and `expand_query` accept an optional pre-built `client: Anthropic` to avoid repeated key lookups when called in a loop (e.g. query expansion + answer generation in the same request)
- Streamlit app uses `@st.cache_resource` for all heavy objects (FAISS index, BM25 retriever, cross-encoder) so they are loaded once per worker process, not on every user interaction

---

## Step 4: Cloud Deployment Plan

### Overview

This plan describes deploying the Amazon Video Games RAG search app on **AWS**. The app is a stateless Streamlit front-end backed by in-memory FAISS and BM25 indices, a cross-encoder re-ranker loaded from HuggingFace, and the Anthropic API for query expansion and answer generation.

---

### Data Storage

| Artifact | AWS Service | Rationale |
|---|---|---|
| Raw JSONL (reviews + metadata, ~5GB) | **S3** (Standard-IA) | Write-once; only accessed during preprocessing. Infrequent Access tier reduces cost. |
| Processed Parquet (`merged_reviews.parquet`, ~500MB) | **S3** (Standard) | Used by index-build jobs; moderate access frequency. |
| FAISS index (`faiss_index/`, ~200MB) | **S3** (Standard) | Downloaded to container at startup. Small enough to load into RAM. |
| BM25 index (`bm25_retriever.pkl`, ~150MB) | **S3** (Standard) | Same as FAISS — downloaded at container startup. |
| Cross-encoder model weights | **S3** or HuggingFace Hub cache | Loaded once per container lifecycle. |
| User feedback CSV (optional) | **S3** or **DynamoDB** | Append-only log; DynamoDB if high write concurrency is expected. |

S3 bucket versioning is enabled on the index prefix so that a bad re-index can be rolled back by pointing the app at the previous object version.

---

### Compute

**App hosting: AWS App Runner (recommended for low-to-medium traffic)**

For a small user base (up to ~20 concurrent users), **AWS App Runner** is the right choice over ECS + ALB. App Runner takes a container image directly from ECR, handles TLS, routing, and scaling automatically, and charges only for active compute time — it scales to near-zero when idle, so you do not pay for an always-on instance during off-hours.

- Package the Streamlit app as a Docker image and push to **Amazon ECR**.
- Create an App Runner service pointing at that ECR image.
- Each instance downloads the FAISS and BM25 indices from S3 on startup, then serves queries from memory.
- **Instance sizing:** `2 vCPU / 4 GB RAM` is sufficient (FAISS + BM25 + cross-encoder need ~3 GB; App Runner pauses billing when no requests are in flight).
- No ALB, no VPC configuration, no Auto Scaling Group — App Runner handles all of this internally.

**When to add an ALB (scale-up path):**

An ALB only makes sense if you outgrow App Runner (>100 concurrent users, custom routing rules, WebSocket support, or need fine-grained VPC control). At that point, migrate to ECS Fargate behind an ALB with a minimum of 2 tasks across two Availability Zones. For this project an ALB is unnecessary overhead.

**Handling multiple concurrent users:**

- App Runner automatically scales instances based on concurrent requests (configurable: e.g. 1 new instance per 10 concurrent requests).
- Because each instance holds the full index in RAM and is stateless, horizontal scaling works without shared state or sticky sessions.
- For a course project with ~5 users, a single instance is sufficient.

**LLM inference:**

- Anthropic API is called externally (no self-hosted model). No GPU infrastructure needed; cost scales with usage.
- The API key is stored in **AWS Secrets Manager** and injected into the container as an environment variable at deploy time — never baked into the Docker image.
- If LLM cost becomes a concern, query expansion can be disabled in the UI (already a toggle), or replaced with a smaller self-hosted model on a GPU EC2 instance.

---

### Streaming / Updates

**Incorporating new products:**

New product data arrives as additional JSONL files (or delta exports from the Amazon Reviews dataset). The update pipeline is:

1. Upload new JSONL files to the raw S3 prefix.
2. An **S3 Event Notification** triggers a **Lambda function** (or an **ECS scheduled task**) that runs the preprocessing job (`src/preprocess.py`), then rebuilds both indices (`src/bm25.py`, `src/semantic.py`).
3. New index files are written to a staging S3 prefix (e.g. `context_store/staging/`).
4. After a smoke-test Lambda confirms index integrity, the staging prefix is promoted to production (S3 object copy / rename).
5. ECS tasks are recycled (rolling update) so they pick up the new indices from S3 on their next startup.

This gives a **blue-green index swap** with no downtime: old containers keep serving from the old in-memory index while new containers start up with the updated index.

**Keeping the pipeline up to date:**

- **Index rebuild schedule:** Run a nightly or weekly ECS scheduled task (via **EventBridge Scheduler**) to incorporate any new review data.
- **Model updates:** The cross-encoder and embedding model are pinned by version in `environment.yml`. Upgrading them requires rebuilding the Docker image (CI/CD pipeline on **GitHub Actions** → ECR push → ECS rolling deploy).
- **LLM model updates:** The Anthropic model ID is an environment variable (`DEFAULT_MODEL`). Updating it requires only a new ECS task definition revision — no re-index needed.

---

### Architecture Diagram (Summary)

```
User browser
    │
    ▼
AWS App Runner (auto-TLS, auto-scaling, no ALB needed)
  ├── Streamlit app (app/app.py)
  ├── FAISS index  ◄── S3 (loaded at startup)
  ├── BM25 index   ◄── S3 (loaded at startup)
  ├── Cross-encoder (HuggingFace / S3 cache)
  └── Anthropic API calls (query expansion, answer gen)
         │
         ▼
    Anthropic Claude API (external)

Index update pipeline:
S3 event → Lambda → ECS task (preprocess + index rebuild) → S3 staging → promote → App Runner redeploy

Secrets: ANTHROPIC_API_KEY in AWS Secrets Manager → injected as env var
Monitoring: CloudWatch Logs + App Runner metrics
```

---

### Cost Estimate (rough)

**Small scale (~5 users, course project):**

| Component | Estimated monthly cost |
|---|---|
| App Runner (1 instance × 2vCPU/4GB, pay-per-use) | ~$15–30 (idles when no traffic) |
| S3 (10 GB data + requests) | ~$5 |
| Secrets Manager | ~$1 |
| Anthropic API (depends on usage) | variable |
| **Total (excluding LLM)** | **~$20–35/month** |

**Production scale (~100 concurrent users, ALB + ECS Fargate):**

| Component | Estimated monthly cost |
|---|---|
| ECS Fargate (2 tasks × 4vCPU/8GB, ~730 hr) | ~$120 |
| ALB | ~$20 |
| S3 + Secrets Manager | ~$6 |
| Anthropic API | variable |
| **Total (excluding LLM)** | **~$150/month** |

App Runner's pay-per-request model makes it the clear choice for a demo or course project — you pay only when the app is actually serving requests.
