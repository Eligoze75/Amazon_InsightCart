"""Anthropic-based query rewriting for retrieval (Claude Haiku).

Used by retrieval pipelines and the Streamlit app. Set ``ANTHROPIC_API_KEY``
(see ``.env.example``). Set ``DEFAULT_MODEL`` to a Haiku model id your key can
call (e.g. ``claude-3-5-haiku-latest`` or a dated snapshot).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_NUM_VARIANTS = 3
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You help retrieve Amazon video game product reviews. "
    "Given a user question, produce diverse paraphrases that could match "
    "different wording in reviews or product text. Stay on topic; do not answer the question."
)


@dataclass(frozen=True)
class QueryExpansionResult:
    """Original query plus LLM paraphrases for multi-query retrieval."""

    original: str
    variants: tuple[str, ...]

    def all_queries(self) -> list[str]:
        """Original first, then unique variants (order preserved, non-empty)."""
        seen: set[str] = set()
        out: list[str] = []
        for q in (self.original, *self.variants):
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out


def _get_api_key() -> str:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or the environment."
        )
    return key


def _parse_json_array(text: str) -> list[str]:
    """Extracts a JSON array of strings from model output (allows markdown fences)."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("No JSON array found in model response")
    arr = json.loads(raw[start : end + 1])
    if not isinstance(arr, list):
        raise ValueError("Expected JSON array")
    return [str(x).strip() for x in arr if str(x).strip()]


def expand_query(
    query: str,
    *,
    num_variants: int = DEFAULT_NUM_VARIANTS,
    model: str = DEFAULT_MODEL,
    client: Anthropic | None = None,
) -> QueryExpansionResult:
    """Returns the original query and Haiku-generated paraphrases."""
    q = query.strip()
    if not q:
        return QueryExpansionResult(original="", variants=())

    if client is None:
        client = Anthropic(api_key=_get_api_key())

    user_prompt = (
        f"User query:\n{q}\n\n"
        f"Return ONLY a JSON array of exactly {num_variants} strings. "
        "Each string must be a different paraphrase of the user query for search. "
        "No other text or markdown."
    )

    msg = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = ""
    for block in msg.content:
        if block.type == "text":
            text += block.text

    variants_raw = _parse_json_array(text)
    variants = tuple(variants_raw[:num_variants])
    if len(variants) < num_variants:
        LOGGER.warning(
            "Expected %s variants, got %s; padding with empty skipped",
            num_variants,
            len(variants),
        )

    return QueryExpansionResult(original=q, variants=variants)


RAG_SYSTEM_PROMPT = (
    "You are a helpful Amazon Video Games product assistant. "
    "Answer the user's question using only the product information provided. "
    "Be concise and specific."
)


def generate_answer(
    query: str,
    results: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    client: Anthropic | None = None,
) -> str:
    """Synthesize a natural language answer from retrieved product results.

    Args:
        query: The original user query.
        results: Top-k result dicts from hybrid search (must have ``product_title``
                 and either ``content`` or ``review_texts``).
        model: Claude model to use.
        client: Optional pre-built Anthropic client (avoids repeated key lookup).

    Returns:
        A 2-3 sentence answer grounded in the retrieved products.
    """
    if client is None:
        client = Anthropic(api_key=_get_api_key())

    context_parts: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("product_title") or "Unknown product"
        rating = r.get("average_rating")
        rating_str = f" (rated {rating:.1f}/5)" if rating is not None else ""
        text = (r.get("content") or r.get("review_texts") or "").strip()
        snippet = text[:500] + ("…" if len(text) > 500 else "")
        context_parts.append(f"{i}. {title}{rating_str}\n{snippet}")

    context = "\n\n".join(context_parts)
    user_prompt = (
        f"User query: {query}\n\n"
        f"Top retrieved products:\n{context}\n\n"
        "Based only on the products above, answer the user's query in 2-3 sentences."
    )

    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=RAG_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return msg.content[0].text.strip()


def main() -> None:
    """Smoke test: expand a sample query (requires ANTHROPIC_API_KEY)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sample = "What is the best Nintendo console under 300 dollars?"
    result = expand_query(sample, num_variants=3)
    print("Original:", result.original)
    print("Variants:", list(result.variants))
    print("All queries:", result.all_queries())


if __name__ == "__main__":
    main()
