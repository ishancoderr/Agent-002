"""
Stage 2 of parsing: pull the fields out of a query, given its category.

The category is already settled by the time this runs, so the model gets one
prompt written for that category alone and only has to fill in its schema —
"copy what you see" rather than "work out what this is". gpt-4o-mini is enough
for that, which is why the expensive model is confined to stage 1.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any, Dict, Tuple

import openai

from .prompt_loader import EXTRACT_TEMPLATES

log = logging.getLogger("agent2.pipeline.extractor")

# Overridable per deployment via .env (see agent2/.env.example) — same
# os.getenv(NAME, default) convention as DB_HOST etc. in database.py. A
# single request can further override this via UserQuery.model, threaded
# through parse_query() -> QueryExtractor(model=...); this env var only sets
# what's used when a request doesn't ask for a specific model.
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-4o-mini")
MAX_TOKENS = 600      # an "all states" geometry answer needs the room


class QueryExtractor:
    """Fills in one category's fields using that category's own prompt."""

    def __init__(self, client: openai.OpenAI | None = None, model: str = EXTRACT_MODEL):
        """Reuse a passed-in OpenAI client, or make one from the env key.
        `model` defaults to EXTRACT_MODEL but can be overridden per instance."""
        self._client = client or openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def extract(self, query: str, query_type: str) -> Tuple[Dict[str, Any], int]:
        """Return (fields, tokens_used).

        `query_type` reaches here already validated by QueryClassifier — it's
        always one of VALID_QUERY_TYPES, and UNRELATED never reaches
        extraction (parse_query()/pipeline_main.run() short-circuit on it
        first) — so every value here should already have a real template.
        If one doesn't, that's a real bug (a category with no matching
        config/prompts/*.yaml file), and guessing DIRECT_LOOKUP's prompt
        would silently extract the wrong shape instead of surfacing it."""
        if query_type not in EXTRACT_TEMPLATES:
            raise ValueError(
                f"No extraction template for query_type={query_type!r} — "
                f"add one to config/prompts/ (see prompt_loader.py)."
            )
        system_prompt = EXTRACT_TEMPLATES[query_type]
        # Resolved fresh on every call rather than baked into the static
        # template text, so "this year" always means the actual current year,
        # not whatever year happened to be hardcoded when the prompt was written.
        system_prompt += f"\n\nCURRENT_YEAR = {date.today().year}"
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        log.info("       | STEP 1b | Extract (%s, template=%s): %s  (tokens=%d)",
                 self.model, query_type, raw, tokens)

        try:
            return json.loads(raw), tokens
        except json.JSONDecodeError as exc:
            # response_format=json_object makes this very unlikely, but a bad
            # parse must not take the whole query down: an empty dict lets the
            # caller apply its defaults instead.
            log.error("       | STEP 1b | JSON parse failed: %s - raw=%r", exc, raw)
            return {}, tokens
