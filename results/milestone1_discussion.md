# Milestone 1 — Query Set

A set of 10 queries spanning different difficulty levels to evaluate BM25, semantic.

## Query Set

| # | Query | Difficulty |
|---|---|---|
| 1 | `wireless PS5 controller` | Easy |
| 2 | `Nintendo Switch OLED 64GB` | Easy |
| 3 | `Xbox Series X HDMI cable` | Easy |
| 4 | `best god of war` | Medium |
| 5 | `controller that doesn't cause hand fatigue` | Medium |
| 6 | `game for a family to play together on TV` | Medium |
| 7 | `headset with clear voice chat for online gaming` | Medium |
| 8 | `best Nintendo console under 300` | Complex |
| 9 | `best open world RPG under $30 with good story` | Complex |
| 10 | `what do reviewers say about drift issues on Switch controllers` | Complex |

## Observations

<details>
<summary>BM25</summary>

#### 1. `wireless PS5 controller`

<img src="img/q1_bm25.png" alt="wireless PS5 controller — BM25" width="50%">

#### 2. `Nintendo Switch OLED 64GB`

<img src="img/q2_bm25.png" alt="Nintendo Switch OLED 64GB — BM25" width="50%">

#### 3. `Xbox Series X HDMI cable`

<img src="img/q3_bm25.png" alt="Xbox Series X HDMI cable — BM25" width="50%">

#### 4. `best god of war`

<img src="img/q4_bm25.png" alt="best god of war — BM25" width="50%">

#### 5. `controller that doesn't cause hand fatigue`

<img src="img/q5_bm25.png" alt="controller that doesn't cause hand fatigue — BM25" width="50%">

#### 6. `game for a family to play together on TV`

<img src="img/q6_bm25.png" alt="game for a family to play together on TV — BM25" width="50%">

#### 7. `headset with clear voice chat for online gaming`

<img src="img/q7_bm25.png" alt="headset with clear voice chat for online gaming — BM25" width="50%">

#### 8. `best Nintendo console under 300`

<img src="img/q8_bm25.png" alt="best Nintendo console under 300 — BM25" width="50%">

#### 9. `best open world RPG under $30 with good story`

<img src="img/q9_bm25.png" alt="best open world RPG under $30 with good story — BM25" width="50%">

#### 10. `what do reviewers say about drift issues on Switch controllers`

<img src="img/q10_bm25.png" alt="what do reviewers say about drift issues on Switch controllers — BM25" width="50%">

</details>

---

<details>
<summary>Semantic Search</summary>

#### 1. `wireless PS5 controller`

<img src="img/q1_semantic.png" alt="wireless PS5 controller — Semantic" width="50%">

#### 2. `Nintendo Switch OLED 64GB`

<img src="img/q2_semantic.png" alt="Nintendo Switch OLED 64GB — Semantic" width="50%">

#### 3. `Xbox Series X HDMI cable`

<img src="img/q3_semantic.png" alt="Xbox Series X HDMI cable — Semantic" width="50%">

#### 4. `best god of war`

<img src="img/q4_semantic.png" alt="best god of war — Semantic" width="50%">

#### 5. `controller that doesn't cause hand fatigue`

<img src="img/q5_semantic.png" alt="controller that doesn't cause hand fatigue — Semantic" width="50%">

#### 6. `game for a family to play together on TV`

<img src="img/q6_semantic.png" alt="game for a family to play together on TV — Semantic" width="50%">

#### 7. `headset with clear voice chat for online gaming`

<img src="img/q7_semantic.png" alt="headset with clear voice chat for online gaming — Semantic" width="50%">

#### 8. `best Nintendo console under 300`

<img src="img/q8_semantic.png" alt="best Nintendo console under 300 — Semantic" width="50%">

#### 9. `best open world RPG under $30 with good story`

<img src="img/q9_semantic.png" alt="best open world RPG under $30 with good story — Semantic" width="50%">

#### 10. `what do reviewers say about drift issues on Switch controllers`

<img src="img/q10_semantic.png" alt="what do reviewers say about drift issues on Switch controllers — Semantic" width="50%">

</details>

---

## Compare Results

| # | Query | Difficulty | Winner |
|---|---|---|---|
| 1 | `wireless PS5 controller` | Easy | BM25 |
| 2 | `Nintendo Switch OLED 64GB` | Easy | BM25 |
| 3 | `Xbox Series X HDMI cable` | Easy | BM25 |
| 4 | `best god of war` | Medium | Semantic |
| 5 | `controller that doesn't cause hand fatigue` | Medium | BM25 |
| 6 | `game for a family to play together on TV` | Medium | Semantic |
| 7 | `headset with clear voice chat for online gaming` | Medium | Semantic |
| 8 | `best Nintendo console under 300` | Complex | Semantic |
| 9 | `best open world RPG under $30 with good story` | Complex | BM25 |
| 10 | `what do reviewers say about drift issues on Switch controllers` | Complex | BM25 |

### Discussion

**Which method performs better for this query? Why?**

BM25 dominates on keyword-heavy queries (1–3) where product titles contain exact matching terms. Semantic search wins on queries built from proper nouns made of common words (query 4: "God of War") or concept-based descriptions (query 7). For complex queries (8–10) neither retriever produces fully satisfying results.

**Are there cases where BM25 fails but semantic search succeeds?**

Yes, query 4 (`best god of war`) is the clearest example. BM25 reduces "God of War" to the token "war" (low IDF on "god", "of", "best") and returns WarCraft III and Spartan: Total Warrior. Semantic search recognises "God of War" as a PlayStation franchise and returns the correct titles directly.

**Are there cases where semantic search fails?**

Yes — query 8 (`best Nintendo console under 300`). The price constraint "under 300" cannot be encoded into a single vector, so semantic search ignores it. Nintendo hardware is also underrepresented in the dataset (Nintendo limits third-party Amazon listings), so neither retriever can surface the expected products regardless of method.

**Are the top results actually useful for the user's intent?**

For easy queries (1–3) results are relevant and actionable. For medium queries (4–7), semantic search finds the right product category but may miss specific constraints. For complex queries (8–10) results are largely not useful without reranking or structured filtering, as both retrievers fail to handle multiple constraints or opinionated queries.

**How does performance vary across query types?**

| Query Type | BM25 | Semantic |
|---|---|---|
| Keyword (exact terms) | Strong | Adequate |
| Proper noun / franchise | Weak | Strong |
| Intent / concept | Weak | Adequate |
| Multi-constraint / complex | Weak | Weak |

---

## Summarize Insights

### Strengths and weaknesses of each method

**BM25**
- Strengths: fast, interpretable, excellent on queries with specific and unique keywords (product model numbers, accessory names, brand + product type). Requires no GPU and no model loading.
- Weaknesses: treats queries as a bag of words with no understanding of meaning. Common words score near zero due to low IDF, which breaks queries like "God of War" where every token is frequent. Cannot handle synonyms, paraphrases, or intent, "hand fatigue" will not match "ergonomic grip".

**Semantic Search (FAISS)**
- Strengths: understands meaning, synonyms, and intent. Handles proper nouns as concepts (recognises "God of War" as a franchise), concept-based queries ("game for families"), and paraphrased descriptions.
- Weaknesses: price constraints, numerical filters, and multi-hop reasoning cannot be encoded in a single vector. Also sensitive to dataset coverage — if a product category is sparse (e.g. Nintendo hardware), no embedding can compensate for missing data.

---

### Query types that are challenging for both methods

- **Price-constrained queries** — "under $300", "under $30": both retrievers treat numbers as tokens and cannot filter by price. A structured post-retrieval filter is needed.
- **Comparative queries** — "worth the price compared to alternatives": requires reading and synthesising multiple reviews, which neither retriever can do alone.
- **Opinion-mining queries** — "what do reviewers say about drift issues": the answer is distributed across thousands of reviews. No single document contains it; synthesis is required.
- **Sparse category queries** — when the target product type has few listings in the dataset, retrieval quality is bounded by data coverage regardless of method.

---

### Where advanced methods would help

| Problem | Advanced Method |
|---|---|
| Results are relevant but wrong order | **Reranking** — a cross-encoder reads (query, document) pairs together and re-scores the top-N candidates with much higher precision |
| Price / attribute constraints ignored | **Structured filtering** — post-retrieval filter on parquet columns (`price`, `average_rating`) before returning results |
| Opinion and comparison queries | **RAG** — retrieve top passages, then use a language model to synthesise an answer across multiple documents |
| Vocabulary mismatch between query and product text | **Query expansion** — already implemented; Claude Haiku rewrites the query into paraphrases to improve recall |
