"""
Stage 1 of parsing: decide which category a query belongs to.

This stage does one thing only — pick one of VALID_QUERY_TYPES — so the prompt
stays small and the model is not also being asked to pull fields out. The
extraction stage then gets a prompt written specifically for that one category.

There is deliberately no regex "safety net" correcting the model here. An
earlier version had one, and measured against the full question set it changed
a correct answer into a wrong one 7 times out of 250: phrasings its patterns
did not anticipate ("Of X and Y, which is within 100 km of Z?", "what towns
fall within 40 km") were forced into the wrong category. A pattern list can
only override the model, never improve it, so the model's answer stands.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Tuple

import openai

from .prompt_loader import CLASSIFY_SYSTEM
from .query_params import VALID_QUERY_TYPES

log = logging.getLogger("agent2.pipeline.classifier")

# Overridable per deployment via .env (see agent2/.env.example) — same
# os.getenv(NAME, default) convention as DB_HOST etc. in database.py. A
# single request can further override this via UserQuery.model, threaded
# through parse_query() -> QueryClassifier(model=...); this env var only sets
# what's used when a request doesn't ask for a specific model.
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gpt-4o-mini")

# The prompt itself lives in config/prompts/classify.yaml, not here — see
# prompt_loader.py's load_classify_system(). Editing the wording, categories,
# or examples is a YAML edit, not a code change.


def extract_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort JSON parse, kept as a safety net even though the classify
    call itself now requests response_format={"type": "json_object"}. Strips
    markdown fences and falls back to the first {...} block if the model
    wrapped its answer in prose regardless."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}


class QueryClassifier:
    """Picks the one category that best describes a query."""

    def __init__(self, client: openai.OpenAI | None = None, model: str = CLASSIFY_MODEL):
        """Reuse a passed-in OpenAI client, or make one from the env key.
        `model` defaults to CLASSIFY_MODEL but can be overridden per instance."""
        self._client = client or openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def classify(self, query: str) -> Tuple[str, int]:
        """Return (query_type, tokens_used).

        An unrecognised answer falls back to DIRECT_LOOKUP, which is the only
        category that can answer something without further structure."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=20,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        raw = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        log.info("       | STEP 1a | Classify (%s): %s  (tokens=%d)", self.model, raw, tokens)

        data = extract_json_object(raw)
        if not data:
            log.error("       | STEP 1a | Could not parse JSON from classify response - raw=%r", raw)

        query_type = data.get("query_type", "DIRECT_LOOKUP")
        if query_type not in VALID_QUERY_TYPES:
            log.warning("       | STEP 1a | Unknown query_type %r - defaulting to DIRECT_LOOKUP",
                        query_type)
            query_type = "DIRECT_LOOKUP"
        return query_type, tokens
