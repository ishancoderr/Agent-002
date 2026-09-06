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

from .query_params import VALID_QUERY_TYPES

log = logging.getLogger("agent2.pipeline.classifier")

CLASSIFY_MODEL = "gpt-4"


CLASSIFY_SYSTEM = """\
You are a query classifier for a geospatial statistics system about German federal states.
Your ONLY job is to decide which ONE category best describes the user's question. Do not
extract any other details — that happens in a separate step.

Categories:

- DIRECT_LOOKUP: asks for population / marriages / live_births data for one or more named
  states (or "all" states). No spatial relationship, geometry, or shape combination is
  involved — a plain (state, year, attribute) lookup.

- SPATIAL_ADJACENCY: asks which state(s) border / touch / share a border with one or more
  named states. A verdict about whether two boundaries meet, not a shape.

- SPATIAL_DIRECTION: asks which states lie north / south / east / west of a named state.
  A verdict about relative position, not a shape.

- SPATIAL_DISTANCE: asks which STATES are within some distance (km) of a named city or
  state, or whether one named state is within that distance of another. A verdict/list,
  not a shape.

- GEOMETRY_LOOKUP: asks for the geometry, shape, boundary, centroid, WKT, or coordinates
  of one or more named cities or states, as-is — NOT demographic data, and no combination
  of shapes is requested.

- SPATIAL_OPERATION: asks for a NEW shape built out of two or more NAMED cities/states —
  their union (one combined boundary), intersection (the area they share), difference
  (one boundary with another's area removed from it), or symmetric difference (the area
  in one or the other but not both). ALSO covers a distance/buffer test where every
  candidate is explicitly NAMED in the question ("Which of X and Y lie within N km of
  Z?") — every entity involved, reference and candidates alike, is named in the query.

- SPATIAL_RELATIONSHIP_BUFFER: asks which members of an OPEN-ENDED, un-enumerable type —
  in this system, cities — lie within some distance of a named reference. The candidates
  are NOT named in the query; naming them is impossible because they are the answer being
  asked for, not an input. The distinguishing test: could every entity in the question be
  written down in advance? If yes (a fixed list of named states) → SPATIAL_OPERATION or
  SPATIAL_DISTANCE. If no (the question only says "cities", not which ones) →
  SPATIAL_RELATIONSHIP_BUFFER.

- UNRELATED: the question does not name a German federal state or German city, and does
  not ask for demographic data (population/marriages/live_births), geometry/shape, or a
  spatial relationship/operation over German states/cities. This includes general-knowledge
  questions, small talk, questions about other countries, or anything else outside this
  system's scope. When in doubt because the question names no recognizable German state or
  city and matches none of the categories above, choose UNRELATED rather than forcing it
  into DIRECT_LOOKUP.

The grammatical form of the question does not matter. A statement ("States north of Bayern")
and a question asking the same thing ("Which states are north of Bayern?", "Is any state
north of Bayern?") belong to the SAME category. Do not classify based on sentence shape —
classify based on what is actually being asked for.

Return ONLY valid JSON: {"query_type": "<one of the eight categories above>"}

EXAMPLES:

"Give me population for all German states in 2021"
→ {"query_type":"DIRECT_LOOKUP"}

"Give me married and live birth data for Berlin in 2020"
→ {"query_type":"DIRECT_LOOKUP"}

"Which state borders both Hessen and Hamburg?"
→ {"query_type":"SPATIAL_ADJACENCY"}

"Do Bayern and Sachsen share a border?"
→ {"query_type":"SPATIAL_ADJACENCY"}

"States north of Bayern, population and marriages 2020-2021"
→ {"query_type":"SPATIAL_DIRECTION"}

"Which states are north of Bayern?"
→ {"query_type":"SPATIAL_DIRECTION"}

"Does Sachsen lie north of Bayern?"
→ {"query_type":"SPATIAL_DIRECTION"}

"States within 100 km of Munich, population in 2021"
→ {"query_type":"SPATIAL_DISTANCE"}

"Which states are within 100 km of Dortmund?"
→ {"query_type":"SPATIAL_DISTANCE"}

"Is Sachsen within 100 km of Dortmund?"
→ {"query_type":"SPATIAL_DISTANCE"}

"What is the geometry of Munich city?"
→ {"query_type":"GEOMETRY_LOOKUP"}

"Show me the boundary of Bayern state"
→ {"query_type":"GEOMETRY_LOOKUP"}

"What are the geometries of München and Bayern?"
→ {"query_type":"GEOMETRY_LOOKUP"}

"Show shapes of all German states"
→ {"query_type":"GEOMETRY_LOOKUP"}

"Give me one combined boundary for Bayern and Sachsen"
→ {"query_type":"SPATIAL_OPERATION"}

"Which area do Bayern and Sachsen share?"
→ {"query_type":"SPATIAL_OPERATION"}

"Give me Brandenburg with Berlin removed"
→ {"query_type":"SPATIAL_OPERATION"}

"Which areas lie in Bayern or Sachsen but not in both?"
→ {"query_type":"SPATIAL_OPERATION"}

"Which of Nordrhein-Westfalen and Niedersachsen lie within 100 km of Dortmund?"
→ {"query_type":"SPATIAL_OPERATION"}

"Which cities lie within 100 km of Dortmund?"
→ {"query_type":"SPATIAL_RELATIONSHIP_BUFFER"}

"What cities are within 50 km of München?"
→ {"query_type":"SPATIAL_RELATIONSHIP_BUFFER"}

"What is the capital of France?"
→ {"query_type":"UNRELATED"}

"What's the weather like today?"
→ {"query_type":"UNRELATED"}

"Who won the World Cup in 2022?"
→ {"query_type":"UNRELATED"}

"Tell me a joke"
→ {"query_type":"UNRELATED"}

"What is the population of Paris?"
→ {"query_type":"UNRELATED"}
"""


def extract_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort JSON parse for models that do not support
    response_format={"type": "json_object"} — classic gpt-4 among them. Strips
    markdown fences and falls back to the first {...} block if the model
    wrapped its answer in prose."""
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
