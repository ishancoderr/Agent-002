"""
Step 1 — Natural language → structured QueryParams, via two separate LLM calls:
classify (GPT-4, which category is this?) then extract (gpt-4o-mini, using a
prompt template written specifically for that category).

The category taxonomy mirrors the document's own mathematics, not just a
convenient split of examples:

  DIRECT_LOOKUP
      Section 1. A query names a subset of each dimension, Q = G'xT'xA', and
      the gap is what's missing from it: gap_i = Q \\ D_i. Every DIRECT_LOOKUP
      exchange sends/receives :missing-slots / :found-slots.

  SPATIAL_ADJACENCY, SPATIAL_DIRECTION, SPATIAL_DISTANCE
      Section 4's "verdict" relationships (topological / direction / metric),
      generalized from a single named pair to "give me every state for which
      this verdict is true". Resolved locally via PostGIS once any missing
      geometries are filled in (Scenarios 17-19's pattern).

  GEOMETRY_LOOKUP
      Section 2. A name in G presupposes nothing about whether its shape is
      held; the gap is a set difference over names, gap = G' \\ H_i. Every
      GEOMETRY_LOOKUP exchange sends/receives :missing-geometries /
      :found-geometries, regardless of dimension (point or polygon).

  SPATIAL_OPERATION
      Section 3 (Scenarios 13-16: Union / Intersection / Difference /
      SymDifference — the only four ways to combine two shapes without
      running off to infinity, per the region-counting argument) PLUS
      Scenario 20 (a Buffer+Within test where every target is NAMED). All of
      these first resolve named-entity geometry gaps exactly like
      GEOMETRY_LOOKUP, then run a local computation once every shape has
      arrived — the operation itself is never missing (Section 3.5), only an
      input can be.

  SPATIAL_RELATIONSHIP_BUFFER
      Scenario 21 only: a Buffer+Within test where the targets CANNOT be
      named in advance (an open/unenumerable type — cities), so the targets
      are the answer rather than the input. This is the one case in the
      whole system where a constructed shape, not a named-entity request,
      crosses the wire (:spatial-query rather than :missing-geometries).
      Distinguishing this from SPATIAL_OPERATION's named-target buffer test
      (Scenario 20) is exactly the "cannot enumerate the target type" check
      from Section 4's discussion of Scenario 21.

  UNRELATED
      Not in the document at all. The whole taxonomy above is built on Q =
      G'xT'xA' over German states/cities — a query that doesn't name a German
      state/city and doesn't ask for demographic data, geometry, or a spatial
      relationship/operation over them has no gap to compute and no category
      to fall into. Rejected immediately after classification, before the
      extraction call even runs — there is nothing for it to extract.
"""
from __future__ import annotations

import json
import logging
import os
import re
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

VALID_QUERY_TYPES = {
    "DIRECT_LOOKUP", "SPATIAL_ADJACENCY", "SPATIAL_DIRECTION", "SPATIAL_DISTANCE",
    "GEOMETRY_LOOKUP", "SPATIAL_OPERATION", "SPATIAL_RELATIONSHIP_BUFFER", "UNRELATED",
}
# The three verdict relationships share one extraction template and one
# output shape (a spatial_relationship object).
_RELATIONSHIP_TYPES = {"SPATIAL_ADJACENCY", "SPATIAL_DIRECTION", "SPATIAL_DISTANCE"}

VALID_OPERATIONS = {"Union", "Intersection", "Difference", "SymDifference", "BufferWithin"}
VALID_ATTRS        = {"population", "marriages", "live_births"}
VALID_ENTITY_TYPES = {"city", "state"}
MIN_YEAR           = 1990
MAX_YEAR           = 2030

CLASSIFY_MODEL = "gpt-4"
EXTRACT_MODEL  = "gpt-4o-mini"


# ── Stage 1: classification (GPT-4) ─────────────────────────────────────────
# Small, focused prompt whose only job is picking a category. Includes both
# declarative and interrogative phrasings so the model isn't just pattern
# matching a single grammatical form.

_CLASSIFY_SYSTEM = """\
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

"What are the geometries of Munich and Bayern?"
→ {"query_type":"GEOMETRY_LOOKUP"}

"Give me geometries for Berlin, Hamburg, Cologne and Frankfurt"
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

"What cities are within 50 km of Munich?"
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


# ── Stage 2: extraction — one template per category ──────────────────────────
# Each template is self-contained: it only carries the naming/normalization
# rules and the output schema its own category needs, so the model never sees
# instructions for a category it wasn't classified into.

_EXTRACT_DIRECT_LOOKUP = """\
You are extracting the details of a DIRECT_LOOKUP query for a geospatial statistics
database about German federal states. Return ONLY valid JSON (no markdown fences).

German state names – use exactly these spellings:
Baden-Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen,
Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz,
Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein, Thüringen

Valid attributes: population, marriages, live_births
  - "married", "marriage", "marriages" → always output "marriages"
  - "live birth", "births", "live_birth" → always output "live_births"

For temporal ranges always output temporal_start and temporal_end as integers, NOT an
array. The system will expand the range into individual years.
If no year is mentioned default to temporal_start=2021, temporal_end=2021.

Output exactly this shape:
{
  "spatial": ["Bayern"] or "all",
  "temporal_start": 2020,
  "temporal_end": 2021,
  "attributes": ["population"]
}
"spatial" is the named state(s), or "all" if none/every state is named. "attributes"
lists every attribute the query asks for; if none is named, still include "population"
since a data lookup needs at least one attribute to answer.

EXAMPLES:

Query: Give me population for all German states in 2021
→ {"spatial":"all","temporal_start":2021,"temporal_end":2021,"attributes":["population"]}

Query: Give me marriages and live_births for Bayern from 2019 to 2023
→ {"spatial":["Bayern"],"temporal_start":2019,"temporal_end":2023,"attributes":["marriages","live_births"]}

Query: Give me married and live birth data for Berlin in 2020
→ {"spatial":["Berlin"],"temporal_start":2020,"temporal_end":2020,"attributes":["marriages","live_births"]}
"""

_EXTRACT_SPATIAL_RELATIONSHIP = """\
You are extracting the details of a verdict-style spatial relationship query (adjacency,
direction, or distance between German federal states) for a geospatial statistics
database. Return ONLY valid JSON (no markdown fences).

German state names – use exactly these spellings:
Baden-Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen,
Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz,
Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein, Thüringen

City names – always use the English spelling (no umlauts):
  Munich / München   → "Munich"
  Cologne / Köln     → "Cologne"
  Nuremberg / Nürnberg → "Nuremberg"
  Dusseldorf / Düsseldorf → "Dusseldorf"
  Frankfurt          → "Frankfurt"
  Stuttgart          → "Stuttgart"
  Hamburg            → "Hamburg"

Valid attributes: population, marriages, live_births
  - "married", "marriage", "marriages" → always output "marriages"
  - "live birth", "births", "live_birth" → always output "live_births"
  - Only include an attribute the query EXPLICITLY names. A pure spatial question
    ("Which states are within 100 km of Dortmund?", "Which states touch Hessen?") asks
    for the state list itself, not data — output "attributes": [] for it. Never invent
    population or a year the user did not mention.

For temporal ranges always output temporal_start and temporal_end as integers, NOT an
array. If no year is mentioned default to temporal_start=2021, temporal_end=2021 — this
default is harmless when "attributes" is empty, since no data lookup happens in that case.

Output exactly this shape:
{
  "spatial": "all",
  "temporal_start": 2020,
  "temporal_end": 2021,
  "attributes": [],
  "spatial_relationship": {
    "type": "adjacency" | "north_of" | "south_of" | "east_of" | "west_of" | "distance",
    "refs": ["Hessen", "Hamburg"],
    "distance_km": 100
  }
}
"spatial" is always "all" — the actual state list gets resolved from the relationship, not
this field. "refs" holds the state/city name(s) the relationship is measured against.
"distance_km" is only set when the relationship type is "distance".

EXAMPLES:

Query: Which state borders both Hessen and Hamburg? Show population 2015-2024
→ {"spatial":"all","temporal_start":2015,"temporal_end":2024,"attributes":["population"],"spatial_relationship":{"type":"adjacency","refs":["Hessen","Hamburg"],"distance_km":null}}

Query: Which states are north of Bayern?
→ {"spatial":"all","temporal_start":2021,"temporal_end":2021,"attributes":[],"spatial_relationship":{"type":"north_of","refs":["Bayern"],"distance_km":null}}

Query: States north of Bayern, population and marriages 2020-2021
→ {"spatial":"all","temporal_start":2020,"temporal_end":2021,"attributes":["population","marriages"],"spatial_relationship":{"type":"north_of","refs":["Bayern"],"distance_km":null}}

Query: Which states are within 100 km of Dortmund?
→ {"spatial":"all","temporal_start":2021,"temporal_end":2021,"attributes":[],"spatial_relationship":{"type":"distance","refs":["Dortmund"],"distance_km":100}}
"""

_EXTRACT_GEOMETRY_LOOKUP = """\
You are extracting the details of a GEOMETRY_LOOKUP query for a geospatial statistics
database about German federal states. Return ONLY valid JSON (no markdown fences).

German state names – use exactly these spellings:
Baden-Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen,
Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz,
Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein, Thüringen

City names – always use the English spelling (no umlauts):
  Munich / München   → "Munich"
  Cologne / Köln     → "Cologne"
  Nuremberg / Nürnberg → "Nuremberg"
  Dusseldorf / Düsseldorf → "Dusseldorf"
  Frankfurt          → "Frankfurt"
  Stuttgart          → "Stuttgart"
  Hamburg            → "Hamburg"

ENTITY TYPE RULES:
  - If the name is one of the 16 German federal states → entity_type = "state"
  - If the name is a city → entity_type = "city"
  - "Berlin" is BOTH a city and a state. Default to entity_type = "state" unless the user
    explicitly says "city of Berlin".
  - "Hamburg" and "Bremen" are also both cities and states — same rule, default to "state".

Output exactly this shape:
{
  "entities": [
    {"entity_name": "Munich",  "entity_type": "city"},
    {"entity_name": "Bayern",  "entity_type": "state"}
  ]
}
List every entity the query asks about. Do not include attributes or temporal fields.

EXAMPLES:

Query: What is the geometry of Munich city?
→ {"entities":[{"entity_name":"Munich","entity_type":"city"}]}

Query: Show me the boundary of Bayern state
→ {"entities":[{"entity_name":"Bayern","entity_type":"state"}]}

Query: What are the geometries of Munich and Bayern?
→ {"entities":[{"entity_name":"Munich","entity_type":"city"},{"entity_name":"Bayern","entity_type":"state"}]}

Query: Give me geometries for Berlin, Hamburg, Cologne and Frankfurt
→ {"entities":[{"entity_name":"Berlin","entity_type":"state"},{"entity_name":"Hamburg","entity_type":"state"},{"entity_name":"Cologne","entity_type":"city"},{"entity_name":"Frankfurt","entity_type":"city"}]}
"""

_EXTRACT_SPATIAL_OPERATION = """\
You are extracting the details of a SPATIAL_OPERATION query for a geospatial statistics
system about German states/cities. This category builds a NEW shape out of two or more
NAMED entities, using exactly one of five operations. Return ONLY valid JSON (no markdown
fences).

THE FIVE OPERATIONS (there are only four ways to combine two shapes without the result
running off to infinity, plus the named-target buffer test):
  - "Union": one shape covering everything in either input. Order doesn't matter.
    Trigger words: union, combine, combined boundary, merge, dissolve.
  - "Intersection": the shape covering only what both inputs share. Order doesn't matter.
    Trigger words: intersection, shared, overlap, in common.
  - "Difference": the first named shape with the second one's area removed from it.
    ORDER MATTERS: "spatial":["Brandenburg","Berlin"] means Brandenburg minus Berlin, not
    the other way around. Trigger words: difference, removed, minus, without, less.
  - "SymDifference": the area covered by exactly one input, not both. Order doesn't matter.
    Trigger words: symmetric difference, in one or the other but not both, either...or...
    but not both.
  - "BufferWithin": tests which of several NAMED candidate entities lie within a distance
    of a NAMED reference entity. Every candidate must be explicitly named in the query —
    if the query asks about an un-named, open-ended set (e.g. plain "cities"), this is the
    WRONG category; that case is classified differently upstream and never reaches this
    template.

German state names – use exactly these spellings:
Baden-Württemberg, Bayern, Berlin, Brandenburg, Bremen, Hamburg, Hessen,
Mecklenburg-Vorpommern, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz,
Saarland, Sachsen, Sachsen-Anhalt, Schleswig-Holstein, Thüringen

City names – always use the English spelling (no umlauts):
  Munich / München → "Munich", Cologne / Köln → "Cologne", Dortmund → "Dortmund"

Output exactly this shape:
{
  "operation": "Union" | "Intersection" | "Difference" | "SymDifference" | "BufferWithin",
  "spatial": ["Bayern", "Sachsen"],
  "entity_type": "state",
  "distance_km": null
}
"spatial" lists every named entity involved, in the order they appear in the query (this
order is significant for "Difference": the first entry is the shape being reduced). For
"BufferWithin", "spatial"[0] is the reference entity and the remaining entries are the
named candidates to test; set "distance_km" to the distance named in the query (default
100 if the operation is BufferWithin and none is given). For the other four operations
leave "distance_km" as null. "entity_type" is "state" unless every named entity is a city.

EXAMPLES:

Query: Give me one combined boundary for Bayern and Sachsen
→ {"operation":"Union","spatial":["Bayern","Sachsen"],"entity_type":"state","distance_km":null}

Query: Which area do Bayern and Sachsen share?
→ {"operation":"Intersection","spatial":["Bayern","Sachsen"],"entity_type":"state","distance_km":null}

Query: Give me Brandenburg with Berlin removed
→ {"operation":"Difference","spatial":["Brandenburg","Berlin"],"entity_type":"state","distance_km":null}

Query: Which areas lie in Bayern or Sachsen but not in both?
→ {"operation":"SymDifference","spatial":["Bayern","Sachsen"],"entity_type":"state","distance_km":null}

Query: Which of Nordrhein-Westfalen and Niedersachsen lie within 100 km of Dortmund?
→ {"operation":"BufferWithin","spatial":["Dortmund","Nordrhein-Westfalen","Niedersachsen"],"entity_type":"state","distance_km":100}
"""

_EXTRACT_SPATIAL_RELATIONSHIP_BUFFER = """\
You are extracting the details of a SPATIAL_RELATIONSHIP_BUFFER query for a geospatial
statistics system about German cities. This category is a buffer-and-test query over an
OPEN-ENDED target type: it asks which CITIES lie within some distance of a named
reference city, where the target cities cannot be named in advance since they are the
answer. Return ONLY valid JSON (no markdown fences).

City names – always use the English spelling (no umlauts):
  Munich / München   → "Munich"
  Cologne / Köln     → "Cologne"
  Nuremberg / Nürnberg → "Nuremberg"
  Dusseldorf / Düsseldorf → "Dusseldorf"
  Frankfurt          → "Frankfurt"
  Stuttgart          → "Stuttgart"
  Hamburg            → "Hamburg"
  Dortmund           → "Dortmund"

Output exactly this shape:
{
  "spatial": ["Dortmund"],
  "operations": ["Buffer", "Within"],
  "distance_km": 100,
  "target_entity": "city"
}
"spatial" holds the single reference city the distance is measured from. "operations" is
always ["Buffer", "Within"] for this category. "distance_km" is the distance in
kilometres named in the query (default 100 if unspecified). "target_entity" is always
"city". Do not include attributes or temporal fields.

EXAMPLES:

Query: Which cities lie within 100 km of Dortmund?
→ {"spatial":["Dortmund"],"operations":["Buffer","Within"],"distance_km":100,"target_entity":"city"}

Query: What cities are within 50 km of Munich?
→ {"spatial":["Munich"],"operations":["Buffer","Within"],"distance_km":50,"target_entity":"city"}
"""

_EXTRACT_TEMPLATES = {
    "DIRECT_LOOKUP": _EXTRACT_DIRECT_LOOKUP,
    "SPATIAL_ADJACENCY": _EXTRACT_SPATIAL_RELATIONSHIP,
    "SPATIAL_DIRECTION": _EXTRACT_SPATIAL_RELATIONSHIP,
    "SPATIAL_DISTANCE": _EXTRACT_SPATIAL_RELATIONSHIP,
    "GEOMETRY_LOOKUP": _EXTRACT_GEOMETRY_LOOKUP,
    "SPATIAL_OPERATION": _EXTRACT_SPATIAL_OPERATION,
    "SPATIAL_RELATIONSHIP_BUFFER": _EXTRACT_SPATIAL_RELATIONSHIP_BUFFER,
}


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
    # SPATIAL_OPERATION / SPATIAL_RELATIONSHIP_BUFFER
    distance_km: Optional[float] = None
    target_entity: Optional[str] = None
    # SPATIAL_OPERATION only
    operation: Optional[str] = None
    entity_type: Optional[str] = None
    # Step-by-step trace, for evaluation logging — see metrics_logger.py
    classify_tokens: int = 0
    extract_tokens: int = 0
    extracted_data: Dict[str, Any] = field(default_factory=dict)


# ── Deterministic classification safety net ─────────────────────────────────
# Even a classification-only prompt can misfire on a phrasing its examples
# don't cover. These patterns are unambiguous enough to detect and correct
# deterministically, applied right after stage 1 so stage 2 always gets the
# right category to extract for.
_DISTANCE_RE = re.compile(
    r"within\s+(\d+(?:\.\d+)?)\s*km\s+of\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-\s]*?)\s*[\?\.]?\s*$",
    re.IGNORECASE,
)
_DIRECTION_RE = re.compile(
    r"\b(north|south|east|west)\s+of\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-\s]*?)\s*[\?\.]?\s*$",
    re.IGNORECASE,
)
_CITY_KEYWORD_RE = re.compile(r"\bcit(y|ies)\b", re.IGNORECASE)
# Whether the sentence names its candidates ("of X and Y") vs asks about an
# open type ("which cities/states...") is exactly Scenario 20 vs 21's test.
_NAMED_LIST_RE = re.compile(r"\bwhich\s+of\b", re.IGNORECASE)

_OPERATION_KEYWORDS = {
    "Union": re.compile(r"\b(union|combine[d]?|dissolve)\b|combined\s+boundary", re.IGNORECASE),
    "Intersection": re.compile(r"\b(intersection|overlap|in\s+common)\b|\bshare[sd]?\b.*\bboth\b", re.IGNORECASE),
    "SymDifference": re.compile(r"symmetric\s+difference|but\s+not\s+(in\s+)?both", re.IGNORECASE),
    "Difference": re.compile(r"\b(difference|removed|minus|without)\b", re.IGNORECASE),
}


def _deterministic_query_type_override(query: str, query_type: str) -> str:
    """Force the unambiguous categories when the raw text clearly asks for one,
    overriding whatever stage 1 classified it as."""
    m = _DISTANCE_RE.search(query)
    if m:
        # "cities within N km of X" (open target) is SPATIAL_RELATIONSHIP_BUFFER
        # (Scenario 21); "which of X and Y ... within N km" (named targets) or a
        # bare "states within N km" is SPATIAL_DISTANCE/SPATIAL_OPERATION.
        if _CITY_KEYWORD_RE.search(query) and not _NAMED_LIST_RE.search(query):
            wanted = "SPATIAL_RELATIONSHIP_BUFFER"
        elif _NAMED_LIST_RE.search(query):
            wanted = "SPATIAL_OPERATION"
        else:
            wanted = "SPATIAL_DISTANCE"
        if query_type != wanted:
            log.warning("       │ Deterministic override: distance pattern matched "
                        "but classifier returned query_type=%s — forcing %s", query_type, wanted)
            return wanted
        return query_type

    if _DIRECTION_RE.search(query) and query_type != "SPATIAL_DIRECTION":
        log.warning("       │ Deterministic override: direction pattern matched "
                    "but classifier returned query_type=%s — forcing SPATIAL_DIRECTION", query_type)
        return "SPATIAL_DIRECTION"

    # Constructive-operation keywords are a lighter safety net: only apply when
    # the classifier didn't already land on a geometry/operation category, to
    # avoid overriding a correct DIRECT_LOOKUP that happens to contain "without".
    if query_type not in ("SPATIAL_OPERATION", "GEOMETRY_LOOKUP"):
        for op, pattern in _OPERATION_KEYWORDS.items():
            if pattern.search(query):
                log.warning("       │ Deterministic override: %r keyword matched "
                            "but classifier returned query_type=%s — forcing SPATIAL_OPERATION",
                            op, query_type)
                return "SPATIAL_OPERATION"

    return query_type


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


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort JSON parse for models (e.g. classic gpt-4) that don't support
    response_format={"type": "json_object"} — strips markdown fences and pulls
    out the first {...} block if the model wrapped its answer in prose."""
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


def _classify(query: str, client: openai.OpenAI) -> tuple:
    """Stage 1 (GPT-4): which category is this? Returns (query_type, tokens_consumed).

    Classic gpt-4 does not support response_format={"type": "json_object"} (that's
    only on gpt-4-turbo/gpt-4o/mini), so JSON compliance here relies on the prompt's
    "Return ONLY valid JSON" instruction plus a tolerant parser as a safety net."""
    response = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        max_tokens=20,
        temperature=0,
        messages=[
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user",   "content": query},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if response.usage is None:
        raise RuntimeError("OpenAI response missing usage/token data — cannot track token consumption.")
    tokens = response.usage.total_tokens
    log.info("       │ STEP 1a │ Classify response (%s): %s  (tokens=%d)", CLASSIFY_MODEL, raw, tokens)

    data = _extract_json_object(raw)
    if not data:
        log.error("       │ STEP 1a │ Could not parse JSON from classify response — raw=%r", raw)
    query_type = data.get("query_type", "DIRECT_LOOKUP")
    if query_type not in VALID_QUERY_TYPES:
        log.warning("       │ STEP 1a │ Unknown query_type %r — defaulting to DIRECT_LOOKUP", query_type)
        query_type = "DIRECT_LOOKUP"

    query_type = _deterministic_query_type_override(query, query_type)
    return query_type, tokens


def _extract(query: str, query_type: str, client: openai.OpenAI) -> tuple:
    """Stage 2 (gpt-4o-mini): given the category, pull out its fields using the
    template written specifically for that category. Returns (data, tokens_consumed)."""
    system_prompt = _EXTRACT_TEMPLATES.get(query_type, _EXTRACT_DIRECT_LOOKUP)
    response = client.chat.completions.create(
        model=EXTRACT_MODEL,
        max_tokens=300,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Query: {query}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if response.usage is None:
        raise RuntimeError("OpenAI response missing usage/token data — cannot track token consumption.")
    tokens = response.usage.total_tokens
    log.info("       │ STEP 1b │ Extract response (%s, template=%s): %s  (tokens=%d)",
             EXTRACT_MODEL, query_type, raw, tokens)

    data = json.loads(raw)
    return data, tokens


def parse_query(query: str) -> tuple:
    """Returns (QueryParams, tokens_consumed: int).

    Two calls: _classify() (GPT-4) decides the category, then _extract()
    (gpt-4o-mini) fills in that category's fields using its own dedicated
    template. The category then drives which KQML message shape the rest of
    the pipeline builds:
      DIRECT_LOOKUP / SPATIAL_ADJACENCY / SPATIAL_DIRECTION / SPATIAL_DISTANCE
        (with attributes)         -> :missing-slots / :found-slots
      GEOMETRY_LOOKUP, SPATIAL_OPERATION
        (named-entity geometry gaps) -> :missing-geometries / :found-geometries
      SPATIAL_RELATIONSHIP_BUFFER
        (target cannot be named)  -> :spatial-query (buffer-and-test)
    — see query_controller.py's dispatch on params.query_type."""
    log.info("       │ Input : %r", query)
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    query_type, tokens_classify = _classify(query, client)
    log.info("       │ STEP 1a │ Classified as: %s", query_type)

    # ── Unrelated — reject immediately, no extraction call needed ─────────────
    if query_type == "UNRELATED":
        log.info("       │ UNRELATED query — rejecting without an extraction call")
        return QueryParams(
            query_type = "UNRELATED",
            spatial    = [],
            temporal   = [],
            attributes = [],
            raw_query  = query,
            classify_tokens = tokens_classify,
            extract_tokens  = 0,
            extracted_data  = {},
        ), tokens_classify

    # ── Geometry lookup — short-circuit before demographics parsing ───────────
    if query_type == "GEOMETRY_LOOKUP":
        data, tokens_extract = _extract(query, query_type, client)
        tokens_consumed = tokens_classify + tokens_extract
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
            classify_tokens = tokens_classify,
            extract_tokens  = tokens_extract,
            extracted_data  = data,
        )
        return params, tokens_consumed

    # ── Spatial operation (Scenarios 13-16, 20) — named entities only ─────────
    if query_type == "SPATIAL_OPERATION":
        data, tokens_extract = _extract(query, query_type, client)
        tokens_consumed = tokens_classify + tokens_extract
        operation = data.get("operation", "")
        raw_spatial = data.get("spatial", [])
        names = [raw_spatial] if isinstance(raw_spatial, str) else list(raw_spatial or [])
        names = [n for n in names if n]
        entity_type = data.get("entity_type", "state") or "state"

        if operation not in VALID_OPERATIONS or len(names) < 2:
            log.warning("       │ SPATIAL_OPERATION malformed (operation=%r, spatial=%s) — "
                        "falling back to DIRECT_LOOKUP", operation, names)
            query_type = "DIRECT_LOOKUP"
            data, tokens_extract2 = _extract(query, query_type, client)
            tokens_consumed += tokens_extract2
        else:
            try:
                distance_km = float(data["distance_km"]) if data.get("distance_km") is not None else None
            except (TypeError, ValueError):
                distance_km = None
            if operation == "BufferWithin" and distance_km is None:
                distance_km = 100.0
            log.info("       │ SPATIAL_OPERATION : op=%s  spatial=%s  entity_type=%s  distance_km=%s",
                     operation, names, entity_type, distance_km)
            return QueryParams(
                query_type  = "SPATIAL_OPERATION",
                spatial     = names,
                temporal    = [],
                attributes  = [],
                raw_query   = query,
                operation   = operation,
                entity_type = entity_type,
                distance_km = distance_km,
                classify_tokens = tokens_classify,
                extract_tokens  = tokens_consumed - tokens_classify,
                extracted_data  = data,
            ), tokens_consumed

    # ── Buffer over an open/un-enumerable target type (Scenario 21) ───────────
    if query_type == "SPATIAL_RELATIONSHIP_BUFFER":
        data, tokens_extract = _extract(query, query_type, client)
        tokens_consumed = tokens_classify + tokens_extract
        raw_spatial = data.get("spatial", [])
        ref_cities  = [raw_spatial] if isinstance(raw_spatial, str) else list(raw_spatial or [])
        ref_cities  = [c for c in ref_cities if c]
        try:
            distance_km = float(data.get("distance_km", 100))
        except (TypeError, ValueError):
            distance_km = 100.0

        if not ref_cities:
            log.warning("       │ SPATIAL_RELATIONSHIP_BUFFER with no reference city — "
                        "falling back to DIRECT_LOOKUP")
            query_type = "DIRECT_LOOKUP"
            data, tokens_extract2 = _extract(query, query_type, client)
            tokens_consumed += tokens_extract2
        else:
            log.info("       │ SPATIAL_RELATIONSHIP_BUFFER : ref=%s  distance_km=%s  target=%s",
                     ref_cities, distance_km, data.get("target_entity", "city"))
            return QueryParams(
                query_type    = "SPATIAL_RELATIONSHIP_BUFFER",
                spatial       = ref_cities,
                temporal      = [],
                attributes    = [],
                raw_query     = query,
                distance_km   = distance_km,
                target_entity = data.get("target_entity", "city"),
                classify_tokens = tokens_classify,
                extract_tokens  = tokens_consumed - tokens_classify,
                extracted_data  = data,
            ), tokens_consumed

    else:
        data, tokens_extract = _extract(query, query_type, client)
        tokens_consumed = tokens_classify + tokens_extract

    # Expand temporal_start / temporal_end into a list of years
    t_start = int(data.get("temporal_start", data.get("temporal_end", 2021)))
    t_end   = int(data.get("temporal_end",   t_start))
    t_start = max(t_start, MIN_YEAR)
    t_end   = min(t_end,   MAX_YEAR)
    temporal = list(range(t_start, t_end + 1))

    # Validate attributes — drop unknowns, default to population only for DIRECT_LOOKUP
    raw_attrs  = data.get("attributes", [])
    attributes = [a for a in raw_attrs if a in VALID_ATTRS]
    invalid    = [a for a in raw_attrs if a not in VALID_ATTRS]
    if invalid:
        log.warning("       │ Dropped unknown attributes: %s", invalid)
    if not attributes:
        if query_type == "DIRECT_LOOKUP":
            # A data lookup with no attribute named still needs one to answer.
            log.warning("       │ No valid attributes in GPT output %s — defaulting to population", raw_attrs)
            attributes = ["population"]
        else:
            # A pure spatial question ("which states ...") asks for the state
            # list itself — do not invent population/year the user never asked for.
            log.info("       │ Pure spatial question — no data attributes requested")

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

    if query_type in _RELATIONSHIP_TYPES and spatial_rel is None:
        # Extraction failed to produce a relationship for a relationship category —
        # fall back to whatever the deterministic pattern captured, if any.
        m = _DIRECTION_RE.search(query) if query_type == "SPATIAL_DIRECTION" else None
        m = _DISTANCE_RE.search(query) if query_type == "SPATIAL_DISTANCE" else m
        if m:
            if query_type == "SPATIAL_DISTANCE":
                spatial_rel = SpatialRelationship(type="distance", refs=[m.group(2).strip()],
                                                   distance_km=float(m.group(1)))
            else:
                spatial_rel = SpatialRelationship(type=f"{m.group(1).lower()}_of", refs=[m.group(2).strip()])
            log.warning("       │ Extraction produced no spatial_relationship for %s — "
                        "filled from regex fallback: %s", query_type, spatial_rel)

    if spatial_rel is not None:
        spatial = ["all"]  # SPATIAL_DIRECTION/DISTANCE/ADJACENCY resolve `spatial` from the relationship

    log.info("       │ Temporal    : %d → %d (%d years)", t_start, t_end, len(temporal))
    log.info("       │ Attributes  : %s", attributes)
    log.info("       │ Tokens      : classify=%d  extract=%d  total=%d",
             tokens_classify, tokens_consumed - tokens_classify, tokens_consumed)

    params = QueryParams(
        query_type           = query_type,
        spatial              = spatial,
        temporal             = temporal,
        attributes           = attributes,
        spatial_relationship = spatial_rel,
        raw_query            = query,
        classify_tokens      = tokens_classify,
        extract_tokens       = tokens_consumed - tokens_classify,
        extracted_data       = data,
    )

    return params, tokens_consumed
