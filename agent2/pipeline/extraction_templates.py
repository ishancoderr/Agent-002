"""
The stage-2 extraction prompts, one per category.

These are data, not logic: each template carries only the naming rules and the
output schema its own category needs, so the model never sees instructions for
a category it was not classified into. The state and city name blocks are
pulled from the gazetteer so a spelling is defined in exactly one place.
"""
from __future__ import annotations

from .gazetteer import STATE_NAMES_BLOCK, CITY_NAMES_BLOCK

EXTRACT_DIRECT_LOOKUP = ("""\
You are extracting the details of a DIRECT_LOOKUP query for a geospatial statistics
database about German federal states. Return ONLY valid JSON (no markdown fences).

""" + STATE_NAMES_BLOCK + """

Valid attributes: population, marriages, live_births
  - "married", "marriage", "marriages" → always output "marriages"
  - "live birth", "births", "live_birth" → always output "live_births"

"temporal" is always a JSON array of every individual year being asked about, listed
out in full — NOT a start/end pair. A range like "2019 to 2023" must be expanded into
every year in that range: [2019,2020,2021,2022,2023]. A single year is still a
one-element array: [2021]. If no year is mentioned default to [2021].

Output exactly this shape:
{
  "spatial": ["Bayern"] or "all",
  "temporal": [2021],
  "attributes": ["population"]
}
"spatial" is the named state(s), or "all" if none/every state is named. "attributes"
lists every attribute the query asks for; if none is named, still include "population"
since a data lookup needs at least one attribute to answer.

EXAMPLES:

Query: Give me population for all German states in 2021
→ {"spatial":"all","temporal":[2021],"attributes":["population"]}

Query: Give me marriages and live_births for Bayern from 2019 to 2023
→ {"spatial":["Bayern"],"temporal":[2019,2020,2021,2022,2023],"attributes":["marriages","live_births"]}

Query: Give me married and live birth data for Berlin in 2020
→ {"spatial":["Berlin"],"temporal":[2020],"attributes":["marriages","live_births"]}
""")

EXTRACT_SPATIAL_RELATIONSHIP = ("""\
You are extracting the details of a verdict-style spatial relationship query (adjacency,
direction, or distance between German federal states) for a geospatial statistics
database. Return ONLY valid JSON (no markdown fences).

""" + STATE_NAMES_BLOCK + """

""" + CITY_NAMES_BLOCK + """

Valid attributes: population, marriages, live_births
  - "married", "marriage", "marriages" → always output "marriages"
  - "live birth", "births", "live_birth" → always output "live_births"
  - Only include an attribute the query EXPLICITLY names. A pure spatial question
    ("Which states are within 100 km of Dortmund?", "Which states touch Hessen?") asks
    for the state list itself, not data — output "attributes": [] for it. Never invent
    population or a year the user did not mention.

"temporal" is always a JSON array of every individual year being asked about, listed
out in full — NOT a start/end pair. A range like "2015-2024" must be expanded into every
year in that range. A single year is still a one-element array: [2021]. If no year is
mentioned default to [2021] — this default is harmless when "attributes" is empty, since
no data lookup happens in that case.

Output exactly this shape:
{
  "spatial": "all",
  "temporal": [2021],
  "attributes": [],
  "spatial_relationship": {
    "type": "adjacency" | "north_of" | "south_of" | "east_of" | "west_of" | "distance",
    "subject": "Sachsen" or null,
    "refs": ["Hessen", "Hamburg"],
    "distance_km": 100
  }
}
"spatial" is always "all" — the actual state list gets resolved from the relationship, not
this field. "refs" holds the state/city name(s) the relationship is measured against.
"distance_km" is only set when the relationship type is "distance".

"subject" is the one thing the question asks ABOUT, and it decides what kind of answer is
wanted:
  - A yes/no question names one thing and asks whether it stands in the relationship to
    another. Put that thing in "subject", and what it is measured against in "refs":
      "Does Sachsen lie north of Bayern?"      subject "Sachsen", refs ["Bayern"]
      "Do Bayern and Sachsen share a border?"  subject "Bayern",  refs ["Sachsen"]
      "Is Sachsen within 100 km of Dortmund?"  subject "Sachsen", refs ["Dortmund"]
  - A question asking WHICH states qualify wants the list, not a yes/no. Set "subject"
    to null and put every named reference in "refs":
      "Which states are north of Bayern?"            subject null, refs ["Bayern"]
      "Which state borders both Hessen and Hamburg?" subject null, refs ["Hessen","Hamburg"]
Two names in "refs" with a null subject asks which states relate to ALL of them, which is
a different question from a yes/no about a pair.

EXAMPLES:

Query: Which state borders both Hessen and Hamburg? Show population 2015-2024
→ {"spatial":"all","temporal":[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],"attributes":["population"],"spatial_relationship":{"type":"adjacency","subject":null,"refs":["Hessen","Hamburg"],"distance_km":null}}

Query: Do Bayern and Sachsen share a border?
→ {"spatial":"all","temporal":[2021],"attributes":[],"spatial_relationship":{"type":"adjacency","subject":"Bayern","refs":["Sachsen"],"distance_km":null}}

Query: Which states are north of Bayern?
→ {"spatial":"all","temporal":[2021],"attributes":[],"spatial_relationship":{"type":"north_of","subject":null,"refs":["Bayern"],"distance_km":null}}

Query: Does Sachsen lie north of Bayern?
→ {"spatial":"all","temporal":[2021],"attributes":[],"spatial_relationship":{"type":"north_of","subject":"Sachsen","refs":["Bayern"],"distance_km":null}}

Query: States north of Bayern, population and marriages 2020-2021
→ {"spatial":"all","temporal":[2020,2021],"attributes":["population","marriages"],"spatial_relationship":{"type":"north_of","subject":null,"refs":["Bayern"],"distance_km":null}}

Query: Which states are within 100 km of Dortmund?
→ {"spatial":"all","temporal":[2021],"attributes":[],"spatial_relationship":{"type":"distance","subject":null,"refs":["Dortmund"],"distance_km":100}}

Query: Is Sachsen within 100 km of Dortmund?
→ {"spatial":"all","temporal":[2021],"attributes":[],"spatial_relationship":{"type":"distance","subject":"Sachsen","refs":["Dortmund"],"distance_km":100}}
""")

EXTRACT_GEOMETRY_LOOKUP = ("""\
You are extracting the details of a GEOMETRY_LOOKUP query for a geospatial statistics
database about German federal states. Return ONLY valid JSON (no markdown fences).

""" + STATE_NAMES_BLOCK + """

""" + CITY_NAMES_BLOCK + """

ENTITY TYPE RULES:
  - If the name is one of the 16 German federal states → entity_type = "state"
  - If the name is a city → entity_type = "city"
  - "Berlin" is BOTH a city and a state. Default to entity_type = "state" unless the user
    explicitly says "city of Berlin".
  - "Hamburg" and "Bremen" are also both cities and states — same rule, default to "state".

Output exactly this shape:
{
  "entities": [
    {"entity_name": "München", "entity_type": "city"},
    {"entity_name": "Bayern",  "entity_type": "state"}
  ]
}
List every entity the query asks about. Do not include attributes or temporal fields.

EXAMPLES:

Query: What is the geometry of Munich city?
→ {"entities":[{"entity_name":"München","entity_type":"city"}]}

Query: Show me the boundary of Bayern state
→ {"entities":[{"entity_name":"Bayern","entity_type":"state"}]}

Query: What are the geometries of München and Bayern?
→ {"entities":[{"entity_name":"München","entity_type":"city"},{"entity_name":"Bayern","entity_type":"state"}]}

Query: Give me geometries for Berlin, Hamburg, Köln and Frankfurt
→ {"entities":[{"entity_name":"Berlin","entity_type":"state"},{"entity_name":"Hamburg","entity_type":"state"},{"entity_name":"Köln","entity_type":"city"},{"entity_name":"Frankfurt","entity_type":"city"}]}

Query: Show shapes of all German states
→ {"entities":[{"entity_name":"Baden-Württemberg","entity_type":"state"},{"entity_name":"Bayern","entity_type":"state"},{"entity_name":"Berlin","entity_type":"state"},{"entity_name":"Brandenburg","entity_type":"state"},{"entity_name":"Bremen","entity_type":"state"},{"entity_name":"Hamburg","entity_type":"state"},{"entity_name":"Hessen","entity_type":"state"},{"entity_name":"Mecklenburg-Vorpommern","entity_type":"state"},{"entity_name":"Niedersachsen","entity_type":"state"},{"entity_name":"Nordrhein-Westfalen","entity_type":"state"},{"entity_name":"Rheinland-Pfalz","entity_type":"state"},{"entity_name":"Saarland","entity_type":"state"},{"entity_name":"Sachsen","entity_type":"state"},{"entity_name":"Sachsen-Anhalt","entity_type":"state"},{"entity_name":"Schleswig-Holstein","entity_type":"state"},{"entity_name":"Thüringen","entity_type":"state"}]}
""")

EXTRACT_SPATIAL_OPERATION = ("""\
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

""" + STATE_NAMES_BLOCK + """

""" + CITY_NAMES_BLOCK + """

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
""")

EXTRACT_SPATIAL_RELATIONSHIP_BUFFER = ("""\
You are extracting the details of a SPATIAL_RELATIONSHIP_BUFFER query for a geospatial
statistics system about German cities. This category is a buffer-and-test query over an
OPEN-ENDED target type: it asks which CITIES lie within some distance of a named
reference city, where the target cities cannot be named in advance since they are the
answer. Return ONLY valid JSON (no markdown fences).

""" + CITY_NAMES_BLOCK + """

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

Query: What cities are within 50 km of München?
→ {"spatial":["München"],"operations":["Buffer","Within"],"distance_km":50,"target_entity":"city"}
""")

EXTRACT_TEMPLATES = {
    "DIRECT_LOOKUP": EXTRACT_DIRECT_LOOKUP,
    "SPATIAL_ADJACENCY": EXTRACT_SPATIAL_RELATIONSHIP,
    "SPATIAL_DIRECTION": EXTRACT_SPATIAL_RELATIONSHIP,
    "SPATIAL_DISTANCE": EXTRACT_SPATIAL_RELATIONSHIP,
    "GEOMETRY_LOOKUP": EXTRACT_GEOMETRY_LOOKUP,
    "SPATIAL_OPERATION": EXTRACT_SPATIAL_OPERATION,
    "SPATIAL_RELATIONSHIP_BUFFER": EXTRACT_SPATIAL_RELATIONSHIP_BUFFER,
}
