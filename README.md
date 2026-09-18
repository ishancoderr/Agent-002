# Multi-Agent Geospatial Missing Data System — Agent 2

A KQML-based multi-agent system for handling missing data in a geospatial statistics
dataset. Two autonomous agents (this repo, and its sibling **Agent-001**) each hold their
own partition of the data and collaborate over KQML to answer a query neither one can
fully answer alone.

Everything in this system — which entities exist, what data is tracked about them, and
what SQL gets run — is driven by config files and an LLM, not hand-written Python. See
**`EXTENDING.md`** for how to add or remove an entity type (a river, a hospital, whatever)
or edit the prompts. This file is the "how does it work" overview.

---

## Overview

When a user submits a natural-language query such as:

> *"Give me population data for Hessen from 2022 to 2025"*

the system:

1. **Classifies** the query into one of eight categories with an LLM call
2. **Extracts** the structured fields (which entity, which years, which attributes) with a
   second, category-specific LLM call
3. **Resolves** spatial relationships (adjacency, direction, distance) using PostGIS, if
   the category needs it
4. **Looks up** local data — via SQL the LLM writes and a validator checks, not
   hand-written SQL — and classifies every missing cell as one of three gap types
5. **Asks** the peer agent over KQML to fill those gaps
6. **Merges** the combined result and returns it

---

## End-to-end request flow

```
POST /query  {"query": "Give me population data for Hessen from 2022 to 2025"}
       │
       ▼
controller/query_controller.py  ← orchestrates every step, logs each one
       │
       │  STEP 1 ── pipeline/pipeline_main.py's run()
       │              CleanQuery         rewrites names to their DB spelling
       │                │
       │                ▼  gpt-4o-mini (config/prompts/classify.yaml)
       │              QueryClassifier    → query_type = "DIRECT_LOOKUP"
       │                │
       │                ▼  gpt-4o-mini (config/prompts/direct_lookup.yaml)
       │              QueryExtractor     → {spatial: ["Hessen"], temporal: [2022..2025],
       │                                     attributes: ["population"], entity_type: "state"}
       │                │
       │                ▼
       │              QueryParams        the one object every later step reads
       │
       │  STEP 2 ── pipeline/spatial_validator.py
       │              DIRECT_LOOKUP → skipped entirely (this step only runs for the
       │              three verdict-style relationship categories — adjacency, direction,
       │              distance — which resolve state names via PostGIS ST_Intersects /
       │              ST_Azimuth / ST_DWithin before anything else can run)
       │
       │  STEP 3-5 ── retrieval/local_store.py's answer_query()
       │              │
       │              ├─ LocalStore.lookup(params)
       │              │    retrieval/sql_generator.py's SqlGenerator writes a validated
       │              │    SELECT (partition_check, then demographics), run against this
       │              │    agent's own DB. Classifies each (entity, year) as:
       │              │      found                      → DataRecord
       │              │      no row at all for the year  → temporal gap
       │              │      row exists, value is NULL   → attribute gap
       │              │      entity not in this partition → spatial gap
       │              │
       │              ├─ if gaps: messaging/peer_client.py's ask_data(gaps)
       │              │    GapSlot → kqml_messaging.MissingSlot (carries entity_type) →
       │              │    AskMessage → POST http://<peer>/kqml/receive
       │              │    Peer replies with a TellMessage; retried up to 3× on failure
       │              │
       │              └─ result/merger.py's merge_results()
       │                   this agent's DataRecords + the peer's flat dicts, merged and
       │                   sorted by (entity, year), each row tagged with its source
       │
       ▼
QueryResponse
  .status  = "complete" | "partial-complete" | "not-found"
  .query   = {type, spatial, temporal, attributes, entity_type}
  .data    = {complete: [...], partial: [...], missing: [...]}
  .provenance = {kqml_turns, records_from_agent_2, records_from_agent_1, ...}
```

`GEOMETRY_LOOKUP` and `SPATIAL_OPERATION` (shape combination — union, intersection,
difference, symmetric difference) and `SPATIAL_RELATIONSHIP_BUFFER` (an open-ended "which
cities are within N km of X" query, where the targets can't be named in advance) each
follow their own handler in `query_controller.py`, using `retrieval/geometry_resolver.py`
instead of `local_store.py` — but the same "resolve locally first, ask the peer only for
what's missing" shape applies throughout.

---

## Repository structure

```
Agent-002/
├── requirements.txt
├── README.md                        ← this file
├── EXTENDING.md                     ← how to add/remove entities, edit prompts
│
├── config/
│   ├── schema/
│   │   ├── entities.yaml            ← every entity type: table, columns, geometry kind
│   │   ├── attributes.yaml          ← every attribute table hanging off an entity
│   │   └── name_aliases.yaml        ← name spelling variants (Munich/München, ...)
│   └── prompts/
│       ├── classify.yaml            ← stage-1: which of the 8 categories is this?
│       ├── direct_lookup.yaml       ← stage-2 + SQL-generation examples, per category
│       ├── geometry_lookup.yaml
│       ├── spatial_operation.yaml
│       ├── spatial_relationship.yaml         (adjacency/direction/distance, shared)
│       ├── spatial_relationship_buffer.yaml
│       └── sql_generation.yaml      ← SQL-generation rules shared by every category
│
└── agent2/
    ├── main.py                      ← FastAPI app + startup kqml-messaging version check
    ├── database.py                  ← SQLAlchemy engine + SessionLocal
    ├── .env                         ← DB credentials, AGENT1_URL, OPENAI_API_KEY, model overrides
    │
    ├── controller/
    │   ├── query_controller.py      ← POST /query   GET /health
    │   └── kqml_controller.py       ← POST /kqml/receive (answering a peer's ask)
    │
    ├── pipeline/                    ← natural language → QueryParams
    │   ├── pipeline_main.py         ← run(): the one function query_controller.py calls
    │   ├── clean_query.py           ← rewrites raw query text to canonical DB spellings
    │   ├── query_classifier.py      ← stage 1: LLM call using classify.yaml
    │   ├── query_extractor.py       ← stage 2: LLM call using the per-category template
    │   ├── query_parser.py          ← turns raw extraction JSON into a validated QueryParams
    │   ├── query_params.py          ← QueryParams/SpatialRelationship + config-derived constants
    │   ├── prompt_loader.py         ← assembles config/prompts/*.yaml into the text sent to the LLM
    │   ├── gazetteer.py             ← name normalization, built from name_aliases.yaml
    │   └── spatial_validator.py     ← PostGIS adjacency/direction/distance resolution
    │
    ├── retrieval/                   ← local DB lookup, LLM-generated SQL, gap detection
    │   ├── sql_generator.py         ← SqlGenerator + validate_sql() (the injection-safety layer)
    │   ├── local_store.py           ← partition check, demographics lookup, gap classification,
    │   │                              answer_query() (the "detect gaps, ask peer, merge" entry point)
    │   └── geometry_resolver.py     ← the same, for GEOMETRY_LOOKUP/SPATIAL_OPERATION shapes
    │
    ├── messaging/                   ← KQML agent-to-agent communication
    │   ├── agent_registry.py        ← N-agent URL registry, read from .env
    │   ├── kqml_client.py           ← thin wrapper: send a demographic-data ask
    │   ├── kqml_geometry_client.py  ← thin wrapper: send a geometry ask
    │   └── peer_client.py           ← builds and sends every KQML message this agent asks with
    │
    ├── result/
    │   └── merger.py                ← combine this agent's + the peer's records, sorted
    │
    ├── evaluation/
    │   └── metrics_logger.py        ← writes evaluation_metrics.log (full ask/tell trace per request)
    │
    └── utils/
        └── show_sql.py              ← dev tool: print SQL for a query without running it
```

The shared package **`kqml-messaging`** (a separate repo,
[github.com/ishancoderr/kqml-geo](https://github.com/ishancoderr/kqml-geo), installed
editable) defines the actual KQML message types — `MessageFactory`, `MissingSlot`,
`FoundSlot`, `MissingGeometrySlot`, `FoundGeometrySlot`, `AskMessage`, `TellMessage`,
`JSONSerializer` — used identically by both agents. `main.py` checks the installed version
against a minimum at startup and refuses to start against a stale copy.

---

## Config-driven design, in one paragraph

`config/schema/entities.yaml` is the single source of truth for what kinds of named,
located things this system knows about (state, city, ...) — table name, key/id/geometry
columns, whether the set is small and fixed ("enumerable") or open-ended.
`config/schema/attributes.yaml` is the single source of truth for what's measured about
each one (population, marriages, ...), including the aliases a user might type for each
column. Both are read by `agent2/pipeline/query_params.py` (to build the valid
entity-type/attribute sets), `agent2/pipeline/gazetteer.py` (to recognize names in a
query), `agent2/retrieval/sql_generator.py` (to build the SQL allowlist and the schema
description the LLM is shown), and `agent2/pipeline/prompt_loader.py` (to generate the
prompt text). Adding a new entity or attribute table is a YAML edit plus a database
migration — see `EXTENDING.md`.

---

## How SQL actually gets written

Every SQL statement that runs against this agent's own database — the demographic lookup,
a geometry lookup, a spatial-relationship query — is written by an LLM call
(`agent2/retrieval/sql_generator.py`'s `SqlGenerator`), not hand-coded. The model is shown
only the schema declared in `entities.yaml`/`attributes.yaml` and a handful of worked
examples (from each category's own `config/prompts/*.yaml` file, under its
`sql_generation:` key). Before anything runs, `validate_sql()` parses the statement with a
real SQL parser (`sqlglot`, not regex) and rejects it unless it's a single, read-only
`SELECT` that only references allowlisted tables/columns/functions — values are always
bound parameters, never written into the SQL text by the model. Statements whose shape
never changes (a named-entity lookup, a buffer, a binary geometry operation) are generated
once per process and reused, not regenerated on every call.

---

## Missing-data types

| Type | Meaning | What happens |
|---|---|---|
| **Spatial gap** | Entity has zero rows in this agent's DB at all | Whole entity → sent to the peer |
| **Temporal gap** | Entity exists but has no row for that year | Missing years → sent to the peer |
| **Attribute gap** | Row exists but the column value is `NULL` | Affected years → sent to the peer |

---

## KQML message exchange

```
Agent-2                                           Agent-1
   │  POST /kqml/receive                             │
   │ ─────────────────────────────────────────────► │
   │  {"performative": "ask", "sender": "Agent-2",   │
   │   "receiver": "Agent-1", "reply_with": "req-1", │
   │   "content": {"missing_slots": [{               │
   │     "spatial": "Hessen", "temporal": [2024,2025],│
   │     "attributes": ["population"],               │
   │     "entity_type": "state"                      │
   │   }]}}                                           │
   │                                                 │
   │  HTTP 200                                        │
   │ ◄───────────────────────────────────────────── │
   │  {"performative": "tell", "in_reply_to": "req-1",│
   │   "content": {"found_slots": [{                 │
   │     "spatial": "Hessen", "temporal": [2024,2025],│
   │     "attributes": ["population"], "entity_type": "state",
   │     "data": [{"year":2024,"population":6510000},│
   │               {"year":2025,"population":6580010}]│
   │   }], "missing_slots": []}}                      │
```

`entity_type` on the slot is what lets the peer know *which table to look in* — without
it, a request for a non-state entity would silently be checked against the state
partition and come back as a false "missing" even when the peer genuinely holds it. Every
entity type is a plain string here, not a fixed enum — the wire format itself never needs
to change when a new entity type is added on either side.

---

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ with the **PostGIS** extension
- An OpenAI API key

---

## Setup

### 1. Create the database and load the schema/data

```sql
CREATE DATABASE geostats_agent2;
\c geostats_agent2
CREATE EXTENSION postgis;
```

Then run whatever `entities.yaml`/`attributes.yaml` describe against it (see
`EXTENDING.md` §1 for the table-creation pattern used for every entity in this system).

### 2. Configure environment (`agent2/.env`)

```env
DB_HOST=localhost
DB_PORT=5433
DB_USER=postgres
DB_PASS=your_password
DB_NAME=geostats_agent2

AGENT1_URL=http://localhost:8000

OPENAI_API_KEY=sk-proj-...

# optional per-stage model overrides — default to gpt-4o-mini if unset
# CLASSIFY_MODEL=gpt-4o-mini
# EXTRACT_MODEL=gpt-4o-mini
# SQL_MODEL=gpt-4o-mini
```

To add a third or fourth agent later, just add `AGENT3_URL=...` / `AGENT4_URL=...` — no
code change needed.

### 3. Install dependencies and run

```bash
pip install -r requirements.txt
uvicorn agent2.main:app --port 8001 --reload
```

---

## API endpoints

### `POST /query`

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Give me population data for Hessen from 2022 to 2025"}'
```

| Status | Meaning |
|---|---|
| `complete` | Everything requested was found, across both agents |
| `partial-complete` | Some found, some still missing everywhere |
| `not-found` | Nothing found anywhere |
| `rejected` | The query was classified `UNRELATED`, or named an attribute with no year given |

### `POST /kqml/receive`

Receives a KQML `ask` from the peer agent, answers it against this agent's own database,
returns a KQML `tell`.

### `GET /health`

```json
{"status": "ok", "agent": "Agent-2"}
```

---

## Supported query categories

| Category | Example |
|---|---|
| `DIRECT_LOOKUP` | `"Population of Bayern in 2021"` |
| `SPATIAL_ADJACENCY` | `"Which state borders both Hessen and Hamburg? Show population 2015-2024"` |
| `SPATIAL_DIRECTION` | `"States north of Bayern, population 2020-2021"` |
| `SPATIAL_DISTANCE` | `"States within 100 km of München, population in 2021"` |
| `GEOMETRY_LOOKUP` | `"What is the geometry of Bayern?"` |
| `SPATIAL_OPERATION` | `"Give me one combined boundary for Bayern and Sachsen"` |
| `SPATIAL_RELATIONSHIP_BUFFER` | `"Which cities lie within 100 km of Dortmund?"` |
| `UNRELATED` | anything outside this system's scope — rejected, not guessed at |

---

## Developer tools

Print the SQL a query would run, without hitting the database:

```bash
python -m agent2.utils.show_sql "Give me population for Hessen from 2022 to 2025"
```

Every request also writes a full step-by-step trace — including the exact KQML `ask`/`tell`
JSON exchanged — to `evaluation_metrics.log`, useful for debugging exactly what a peer was
asked and what it answered.

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `sqlalchemy` | SQL toolkit — every statement still runs through bound parameters |
| `psycopg2-binary` | PostgreSQL driver |
| `httpx` | HTTP client for KQML inter-agent calls |
| `openai` | LLM calls for classification, extraction, and SQL generation |
| `sqlglot` | Real SQL parser — powers `validate_sql()`'s injection-safety checks |
| `pyyaml` | Reads `config/schema/*.yaml` and `config/prompts/*.yaml` |
| `python-dotenv` | `.env` file loading |
| `pydantic` | Request/response validation |
| `kqml-messaging` | Shared KQML message models — [github.com/ishancoderr/kqml-geo](https://github.com/ishancoderr/kqml-geo) |

---

## License

MIT
