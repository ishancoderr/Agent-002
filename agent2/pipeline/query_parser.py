"""
Step 1 — Natural language → structured QueryParams via GPT-4o mini.

Supported query types:
  DIRECT_LOOKUP      – "population of Bayern in 2021"
  SPATIAL_ADJACENCY  – "state that borders both Hessen and Hamburg"
  SPATIAL_DIRECTION  – "states north of Bayern"
  SPATIAL_DISTANCE   – "states within 100 km of München"
  GEOMETRY_LOOKUP    – "geometry of Munich city" / "shapes of Bayern and Köln"
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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

VALID_QUERY_TYPES  = {"DIRECT_LOOKUP", "SPATIAL_ADJACENCY", "SPATIAL_DIRECTION",
                      "SPATIAL_DISTANCE", "GEOMETRY_LOOKUP"}
VALID_ATTRS        = {"population", "marriages", "live_births"}
VALID_ENTITY_TYPES = {"city", "state"}
MIN_YEAR           = 1990
MAX_YEAR           = 2030

_SYSTEM = """\
You are a query parser for a geospatial statistics database about German federal states.
Extract structured query parameters and return ONLY valid JSON (no markdown fences).

German state names – use exactly these spellings:
Baden-Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen,
Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz,
Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein, Thüringen

Valid attributes: population, marriages, live_births
  - "married", "marriage", "marriages" → always output "marriages"
  - "live birth", "births", "live_birth" → always output "live_births"

For temporal ranges always output temporal_start and temporal_end as integers, NOT an array.
The system will expand the range into individual years.
If no year is mentioned default to temporal_start=2021, temporal_end=2021.

City names – always use the German spelling with umlauts:
  Munich / München   → "München"
  Cologne / Köln     → "Köln"
  Nuremberg / Nürnberg → "Nürnberg"
  Dusseldorf         → "Düsseldorf"
  Frankfurt          → "Frankfurt"
  Stuttgart          → "Stuttgart"
  Hamburg            → "Hamburg"

ENTITY TYPE RULES:
  - If the name is one of the 16 German federal states → entity_type = "state"
  - If the name is a city → entity_type = "city"
  - "Berlin" is BOTH a city and a state. For GEOMETRY_LOOKUP default to entity_type = "state"
    unless the user explicitly says "city of Berlin".
  - "Hamburg" and "Bremen" are also both cities and states — same rule, default to "state".

GEOMETRY QUERIES:
  If the user asks for geometry, shape, boundary, centroid, WKT, coordinates, or spatial
  extent of ANY number of cities or states, output query_type = "GEOMETRY_LOOKUP".
  List ALL requested entities. Do NOT include attributes or temporal fields.

UNKNOWN / UNANSWERABLE QUERIES:
  If the query is completely unrelated to German geospatial statistics or geometry,
  still output a valid JSON with query_type = "DIRECT_LOOKUP", spatial = [], attributes = ["population"],
  temporal_start = 2021, temporal_end = 2021. Never return an error or non-JSON.

─────────────────────────────────────────────────────────────
DEMOGRAPHICS schema:
{
  "query_type": "DIRECT_LOOKUP" | "SPATIAL_ADJACENCY" | "SPATIAL_DIRECTION" | "SPATIAL_DISTANCE",
  "spatial": ["Bayern"] or "all",
  "temporal_start": 2020,
  "temporal_end": 2021,
  "attributes": ["population"],
  "spatial_relationship": {
    "type": "adjacency" | "north_of" | "south_of" | "east_of" | "west_of" | "distance",
    "refs": ["Hessen", "Hamburg"],
    "distance_km": 100
  } or null
}

GEOMETRY schema:
{
  "query_type": "GEOMETRY_LOOKUP",
  "entities": [
    {"entity_name": "München", "entity_type": "city"},
    {"entity_name": "Bayern",  "entity_type": "state"}
  ]
}
─────────────────────────────────────────────────────────────
EXAMPLES:

"Give me population for all German states in 2021"
→ {"query_type":"DIRECT_LOOKUP","spatial":"all","temporal_start":2021,"temporal_end":2021,"attributes":["population"],"spatial_relationship":null}

"Give me marriages and live_births for Bayern from 2019 to 2023"
→ {"query_type":"DIRECT_LOOKUP","spatial":["Bayern"],"temporal_start":2019,"temporal_end":2023,"attributes":["marriages","live_births"],"spatial_relationship":null}

"Give me married and live birth data for Berlin in 2020"
→ {"query_type":"DIRECT_LOOKUP","spatial":["Berlin"],"temporal_start":2020,"temporal_end":2020,"attributes":["marriages","live_births"],"spatial_relationship":null}

"Which state borders both Hessen and Hamburg? Show population 2015-2024"
→ {"query_type":"SPATIAL_ADJACENCY","spatial":"all","temporal_start":2015,"temporal_end":2024,"attributes":["population"],"spatial_relationship":{"type":"adjacency","refs":["Hessen","Hamburg"],"distance_km":null}}

"States north of Bayern, population and marriages 2020-2021"
→ {"query_type":"SPATIAL_DIRECTION","spatial":"all","temporal_start":2020,"temporal_end":2021,"attributes":["population","marriages"],"spatial_relationship":{"type":"north_of","refs":["Bayern"],"distance_km":null}}

"States within 100 km of Munich, population in 2021"
→ {"query_type":"SPATIAL_DISTANCE","spatial":"all","temporal_start":2021,"temporal_end":2021,"attributes":["population"],"spatial_relationship":{"type":"distance","refs":["München"],"distance_km":100}}

"What is the geometry of Munich city?"
→ {"query_type":"GEOMETRY_LOOKUP","entities":[{"entity_name":"München","entity_type":"city"}]}

"Show me the boundary of Bayern state"
→ {"query_type":"GEOMETRY_LOOKUP","entities":[{"entity_name":"Bayern","entity_type":"state"}]}

"What are the geometries of München and Bayern?"
→ {"query_type":"GEOMETRY_LOOKUP","entities":[{"entity_name":"München","entity_type":"city"},{"entity_name":"Bayern","entity_type":"state"}]}

"Give me geometries for Berlin, Hamburg, Köln and Frankfurt"
→ {"query_type":"GEOMETRY_LOOKUP","entities":[{"entity_name":"Berlin","entity_type":"state"},{"entity_name":"Hamburg","entity_type":"state"},{"entity_name":"Köln","entity_type":"city"},{"entity_name":"Frankfurt","entity_type":"city"}]}
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
    # GEOMETRY_LOOKUP only
    entities: Optional[List[Dict[str, str]]] = None


def _sanitize_entities(raw: Any) -> List[Dict[str, str]]:
    """Validate and clean the entities list from GPT output."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name  = str(item.get("entity_name", "")).strip()
        etype = str(item.get("entity_type", "city")).strip().lower()
        if not name:
            continue
        if etype not in VALID_ENTITY_TYPES:
            etype = "state" if name in GERMAN_STATES else "city"
            log.warning("       │ Unknown entity_type for %r — inferred as %s", name, etype)
        result.append({"entity_name": name, "entity_type": etype})
    return result


def parse_query(query: str) -> tuple:
    """Returns (QueryParams, tokens_consumed: int)."""
    log.info("       │ Sending to GPT-4o mini ...")
    log.info("       │ Input : %r", query)

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": query},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if response.usage is None:
        raise RuntimeError("OpenAI response missing usage/token data — cannot track token consumption.")
    tokens_consumed = response.usage.total_tokens

    log.info("       │ LLM response: %s", raw)
    log.info("       │ Tokens used : %d", tokens_consumed)

    data = json.loads(raw)

    # ── Geometry lookup — short-circuit before demographics parsing ───────────
    if data.get("query_type") == "GEOMETRY_LOOKUP":
        entities = _sanitize_entities(data.get("entities", []))
        log.info("       │ GEOMETRY_LOOKUP : %d entity/entities", len(entities))
        for e in entities:
            log.info("       │   %s (%s)", e["entity_name"], e["entity_type"])
        if not entities:
            raise ValueError("GEOMETRY_LOOKUP query must include at least one entity")
        params = QueryParams(
            query_type = "GEOMETRY_LOOKUP",
            spatial    = [],
            temporal   = [],
            attributes = [],
            raw_query  = query,
            entities   = entities,
        )
        return params, tokens_consumed

    # Expand temporal_start / temporal_end into a list of years
    t_start = int(data.get("temporal_start", data.get("temporal_end", 2021)))
    t_end   = int(data.get("temporal_end",   t_start))
    t_start = max(t_start, MIN_YEAR)
    t_end   = min(t_end,   MAX_YEAR)
    temporal = list(range(t_start, t_end + 1))

    # Validate attributes — drop unknowns, default to population if empty
    raw_attrs  = data.get("attributes", ["population"])
    attributes = [a for a in raw_attrs if a in VALID_ATTRS]
    invalid    = [a for a in raw_attrs if a not in VALID_ATTRS]
    if invalid:
        log.warning("       │ Dropped unknown attributes: %s", invalid)
    if not attributes:
        log.warning("       │ No valid attributes in GPT output %s — defaulting to population", raw_attrs)
        attributes = ["population"]

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

    log.info("       │ Temporal    : %d → %d (%d years)", t_start, t_end, len(temporal))
    log.info("       │ Attributes  : %s", attributes)

    params = QueryParams(
        query_type           = data.get("query_type", "DIRECT_LOOKUP"),
        spatial              = spatial,
        temporal             = temporal,
        attributes           = attributes,
        spatial_relationship = spatial_rel,
        raw_query            = query,
    )

    return params, tokens_consumed
