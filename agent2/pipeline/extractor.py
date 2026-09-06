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
from typing import Any, Dict, Tuple

import openai

from .extraction_templates import EXTRACT_TEMPLATES, EXTRACT_DIRECT_LOOKUP

log = logging.getLogger("agent2.pipeline.extractor")

EXTRACT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 600      # an "all states" geometry answer needs the room


class QueryExtractor:
    """Fills in one category's fields using that category's own prompt."""

    def __init__(self, client: openai.OpenAI | None = None, model: str = EXTRACT_MODEL):
        self._client = client or openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def extract(self, query: str, query_type: str) -> Tuple[Dict[str, Any], int]:
        """Return (fields, tokens_used).

        A category with no template of its own falls back to the DIRECT_LOOKUP
        schema, which is the shape the rest of the pipeline can always use."""
        system_prompt = EXTRACT_TEMPLATES.get(query_type, EXTRACT_DIRECT_LOOKUP)
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
