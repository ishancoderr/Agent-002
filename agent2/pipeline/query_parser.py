"""
Step 1 — Natural language → structured QueryParams via GPT-4o mini.

Supported query types:
  DIRECT_LOOKUP      – "population of Bayern in 2021"
  SPATIAL_ADJACENCY  – "state that borders both Hessen and Hamburg"
  SPATIAL_DIRECTION  – "states north of Bayern"
  SPATIAL_DISTANCE   – "states within 100 km of München"
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger("agent2.pipeline.parser")

GERMAN_STATES = [
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen",
    "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen",
    "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Sachsen",
    "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen",
]

_SYSTEM = """\
You are a query parser for a geospatial statistics database about German federal states.
Extract structured query parameters and return ONLY valid JSON (no markdown fences).

German state names – use exactly these spellings:
Baden-Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen,
Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz,
Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein, Thüringen

Valid attributes: population, marriages, live_births

Return JSON:
{
  "query_type": "DIRECT_LOOKUP" | "SPATIAL_ADJACENCY" | "SPATIAL_DIRECTION" | "SPATIAL_DISTANCE",
  "spatial": ["Bayern"] or "all",
  "temporal": [2020, 2021],
  "attributes": ["population"],
  "spatial_relationship": {
    "type": "adjacency" | "north_of" | "south_of" | "east_of" | "west_of" | "distance",
    "refs": ["Hessen", "Hamburg"],
    "distance_km": 100
  } or null
}

Examples:
"Give me population for all German states in 2021"
→ {"query_type":"DIRECT_LOOKUP","spatial":"all","temporal":[2021],"attributes":["population"],"spatial_relationship":null}

"Give me marriages and live_births for Bayern from 2019 to 2023"
→ {"query_type":"DIRECT_LOOKUP","spatial":["Bayern"],"temporal":[2019,2020,2021,2022,2023],"attributes":["marriages","live_births"],"spatial_relationship":null}

"Which state borders both Hessen and Hamburg? Show population 2015-2024"
→ {"query_type":"SPATIAL_ADJACENCY","spatial":"all","temporal":[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],"attributes":["population"],"spatial_relationship":{"type":"adjacency","refs":["Hessen","Hamburg"],"distance_km":null}}

"States north of Bayern for 2020 and 2021, population and marriages"
→ {"query_type":"SPATIAL_DIRECTION","spatial":"all","temporal":[2020,2021],"attributes":["population","marriages"],"spatial_relationship":{"type":"north_of","refs":["Bayern"],"distance_km":null}}

"Which states are within 100 km of München, population in 2021"
→ {"query_type":"SPATIAL_DISTANCE","spatial":"all","temporal":[2021],"attributes":["population"],"spatial_relationship":{"type":"distance","refs":["München"],"distance_km":100}}
"""


@dataclass
class SpatialRelationship:
    type: str
    refs: List[str] = field(default_factory=list)
    distance_km: Optional[float] = None


@dataclass
class QueryParams:
    query_type: str
    spatial: List[str]
    temporal: List[int]
    attributes: List[str]
    spatial_relationship: Optional[SpatialRelationship] = None
    raw_query: str = ""


def parse_query(query: str) -> tuple:
    """Returns (QueryParams, tokens_consumed: int)."""
    log.info("       │ Sending to GPT-4o mini ...")
    log.info("       │ Input : %r", query)

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": query},
        ],
    )

    raw            = response.choices[0].message.content.strip()
    if response.usage is None:
        raise RuntimeError("OpenAI response missing usage/token data — cannot track token consumption.")
    tokens_consumed = response.usage.total_tokens

    log.info("       │ LLM response: %s", raw)
    log.info("       │ Tokens used : %d", tokens_consumed)

    data = json.loads(raw)

    spatial = data.get("spatial", "all")
    spatial = ["all"] if spatial == "all" else (
        [spatial] if isinstance(spatial, str) else spatial
    )

    rel_data = data.get("spatial_relationship")
    spatial_rel = None
    if rel_data:
        spatial_rel = SpatialRelationship(
            type=rel_data.get("type", ""),
            refs=rel_data.get("refs", []),
            distance_km=rel_data.get("distance_km"),
        )

    params = QueryParams(
        query_type           = data.get("query_type", "DIRECT_LOOKUP"),
        spatial              = spatial,
        temporal             = data.get("temporal", []),
        attributes           = data.get("attributes", ["population"]),
        spatial_relationship = spatial_rel,
        raw_query            = query,
    )

    return params, tokens_consumed
