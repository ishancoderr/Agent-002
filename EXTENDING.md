# Extending this system

This is a practical how-to for adding or removing an entity type (a "kind of thing" the
system knows about, like a state, a city, or a river), adding or removing the data
attached to one, and editing the prompts that control how the LLM reads a query.

If you just want to understand how the system works end to end, read `README.md` first —
this file assumes you already know the shape of the pipeline.

Almost everything in this system is **config-driven**: the two files under
`config/schema/` describe what entities and attribute tables exist, and the system reads
them at runtime. Adding a new kind of data is mostly a YAML edit and a database migration
— not a code change. The one place that still needs a hand-written edit is the wording in
two prompt files, explained in its own section below.

---

## 1. Adding a new entity type (e.g. a hospital, a river, a lake)

An "entity type" is anything with a name and (usually) a location — states and cities are
the two that ship with this system. Here's the full recipe, using a hypothetical
`hospital` entity as the example (point location, one attribute table tracking yearly
patient counts).

### Step 1 — Create the database tables

Every entity type needs one table for the entity itself, and (optionally) one or more
attribute tables holding the data measured about it.

```sql
CREATE TABLE hospitals (
    hospital_id   SERIAL PRIMARY KEY,
    hospital_name VARCHAR NOT NULL UNIQUE,
    location      geometry(Point, 4326)   -- a hospital is a point; use LineString for a
                                           -- river, Polygon/MultiPolygon for an area
);

CREATE TABLE hospital_stats (
    stat_id           SERIAL PRIMARY KEY,
    hospital_id       INTEGER NOT NULL REFERENCES hospitals(hospital_id),
    stat_year         INTEGER NOT NULL,
    patients_treated  INTEGER,
    UNIQUE (hospital_id, stat_year)
);
```

Run this against **each agent's own database** (they're separate databases — the whole
point of the system is that each agent holds its own partition of the data).

### Step 2 — Declare the entity in `config/schema/entities.yaml`

```yaml
hospital:
  enabled:         true
  label:           "hospital"
  label_plural:    "hospitals"
  table:           hospitals
  key_column:      hospital_name
  id_column:       hospital_id
  geometry_column: location
  geometry_kind:   point          # point | linestring | polygon
  enumerable:      false          # true only for a small, fixed, known list (like the
                                   # 16 German states); false for an open-ended set
  gazetteer_block: >-
    Hospital names: any reasonable spelling is fine — names are normalized against the
    database automatically after extraction.
```

That's it — nothing else needs to know this entity exists. `VALID_ENTITY_TYPES`, the SQL
safety validator's table/column allowlist, the gazetteer, and the KQML message format all
read this file and pick the new entity up automatically.

### Step 3 — Declare its attribute table in `config/schema/attributes.yaml`

```yaml
- table:         hospital_stats
  entity:        hospital
  join_column:   hospital_id
  period_column: stat_year        # the year column; use `null` only if the value never
                                   # varies by year (see the comment in the file itself —
                                   # this currently has one real limitation, explained there)
  columns:
    patients_treated:
      label:   "patients treated"
      type:    integer
      aliases: [patients, "patient count", "how many patients"]
```

`aliases` are the words a user might actually type ("how many patients were treated") that
should all resolve to the real column name (`patients_treated`). List every phrasing you
can think of — this is what teaches the extraction prompt to recognize the question
without you writing a prompt rule by hand.

An entity can have more than one attribute table (e.g. `hospital_stats` for patient counts
and a separate `hospital_beds` for bed capacity) — just add another entry with the same
`entity: hospital`.

### Step 4 — Add name aliases, if needed (`config/schema/name_aliases.yaml`)

Only needed if some names have a genuinely different English/German form worth
normalizing (the way `Cologne`/`Köln` or `Rhine`/`Rhein` do). Most entities — most cities,
most hospitals — don't need this at all; an unrecognized name just falls through to a
case-insensitive database search instead of failing.

```yaml
hospital:
  Charité: [charite]   # only if a query might reasonably drop the accent
```

### Step 5 — Update the two hand-written prompt files

This is the one place that isn't automatic, because these two files contain hand-written
English sentences describing the categories, not generated lists.

**`config/prompts/classify.yaml`** — this decides *which kind of question* a query is
before anything else runs. Its `DIRECT_LOOKUP`, `GEOMETRY_LOOKUP`, `SPATIAL_OPERATION`, and
`UNRELATED` category descriptions currently say "German federal states" / "cities/states" —
update the wording to also mention the new entity, e.g.:

```yaml
- 'DIRECT_LOOKUP: asks for a data value this system tracks (e.g. population for a state;
   patients treated for a hospital) for one or more specifically named entities ...'
```

**`config/prompts/geometry_lookup.yaml`** — its `entity_type_rules` list tells the model
how to tell entity types apart by name. Add a bullet:

```yaml
entity_type_rules:
  - 'If the name is one of the 16 German federal states → entity_type = "state"'
  - 'If the name is a hospital → entity_type = "hospital"'
  - 'Otherwise, if the name is a city → entity_type = "city"'
  ...
```

Everything else in `config/prompts/` (`direct_lookup.yaml`'s attribute/gazetteer blocks,
`sql_generation.yaml`'s shared rules) is generated dynamically from `entities.yaml` and
`attributes.yaml` — no edit needed there. See §3 below for how that generation actually
works, if you want to understand why these two files are the exception.

### Step 6 — Restart and test

```bash
uvicorn agent2.main:app --port 8001 --reload
```

Try a few queries covering each shape:
- `"How many patients did Charité treat in 2024?"` (DIRECT_LOOKUP)
- `"What is the location of Charité?"` (GEOMETRY_LOOKUP)
- A query naming a hospital only the *other* agent holds — confirm it comes back
  correctly via the KQML exchange rather than "not found" (see README.md's KQML section
  for what a healthy exchange looks like in the logs).

**Do this on both agents.** They don't have to hold the *same* entities — in fact, testing
the missing-data behavior properly means giving each agent a different subset, the same
way the states/cities partition already works.

---

## 2. Removing an entity type

The reverse of the above, in the same order:

1. **Drop the tables**: `DROP TABLE IF EXISTS hospital_stats; DROP TABLE IF EXISTS hospitals;`
   (drop the attribute table(s) first — they have the foreign key).
2. **Remove the entry from `entities.yaml`.**
3. **Remove the entry from `attributes.yaml`.**
4. **Remove the entry from `name_aliases.yaml`**, if you added one.
5. **Revert the hand-written sentences** in `classify.yaml` and `geometry_lookup.yaml`
   back to not mentioning the entity.
6. Restart and confirm: a query naming the removed entity should now either get rejected
   as `UNRELATED` or come back honestly empty — never a stale/fabricated answer.

A quick way to check you got everything: `grep -rn "hospital" config/` should turn up
nothing except, at most, an explanatory comment (the "how to add an entity type" example
in `entities.yaml`/`attributes.yaml` intentionally keeps a worked example in a comment —
that's documentation, not live support, and is fine to leave).

---

## 3. Editing prompts — how the pieces fit together

Every prompt the system sends to the LLM is assembled at startup from
`config/prompts/*.yaml`, not hand-written as a Python string. `agent2/pipeline/prompt_loader.py`
does the assembling — it's worth skimming once, since its own docstring explains the design.

**Two kinds of prompt file:**

- `classify.yaml` — the *only* prompt with no `query_type` field. It's the one that
  decides which of the categories (`DIRECT_LOOKUP`, `GEOMETRY_LOOKUP`,
  `SPATIAL_ADJACENCY`, ...) a query belongs to. Read before anything else runs.
- Everything else (`direct_lookup.yaml`, `geometry_lookup.yaml`,
  `spatial_operation.yaml`, `spatial_relationship.yaml`,
  `spatial_relationship_buffer.yaml`) — one per category (or, for the three verdict-style
  relationship categories, one file shared by all three via a `query_type: [...]` list).
  These extract the actual fields (which entity, which year, which attribute) once the
  category is already known.

**Each file is a `sections` list plus the content for each section:**

```yaml
sections: [intro, gazetteer, attributes, temporal_rules, output_schema, output_notes, examples]
```

Each name in `sections` maps to a renderer function in `prompt_loader.py`'s `_RENDERERS`
dict. Adding a new *kind* of content (not just new data for an existing kind) means adding
one renderer function there — but for everything covered in §1 above, the existing
renderers already handle it.

**What's generated vs. hand-written, concretely:**

| Section | Source | Hand-written? |
|---|---|---|
| `gazetteer` | `entities.yaml` + `name_aliases.yaml`, via `gazetteer.py`'s `get_names_block()` | No |
| `attributes` | `attributes.yaml`, via `_load_attribute_rules()` | No |
| `output_schema`, `examples` | the YAML file itself | Yes — these are the actual shape/worked-examples you're teaching the model |
| `classify.yaml`'s category descriptions | the YAML file itself | Yes (see §1, Step 5) |

**A special case**: `direct_lookup.yaml` sets `attributes_entity: all` and
`gazetteer: attributes` instead of naming one entity explicitly. This is what makes
`DIRECT_LOOKUP` multi-entity — it shows the model *every* entity that has an attribute
table (not just state), each under its own "`<Entity>` attributes:" heading, generated
fresh every time `attributes.yaml` changes. `geometry_lookup.yaml` similarly uses
`gazetteer: all` (every entity with a geometry, which today is every entity — geometry
isn't optional the way attribute data is). Both mechanisms are why §1's Steps 2–4 alone
are enough to make a new entity askable for `DIRECT_LOOKUP`/`GEOMETRY_LOOKUP` — no template
edit needed for those two files specifically.

**SQL generation prompts live *inside* each category's own file too** — look for the
`sql_generation:` key at the bottom of `direct_lookup.yaml`, `geometry_lookup.yaml`, etc.
These are worked examples of the SQL shape `agent2/retrieval/sql_generator.py` should
produce for that category; the safety rules every one of them has to follow (SELECT-only,
bound parameters, no comments, ...) live once in `config/prompts/sql_generation.yaml` and
are shared across every category rather than repeated. You will not normally need to touch
these when adding an entity — the existing examples already teach the model the general
pattern (name a table, alias its columns) generically enough to extend to a new one.

**Testing a prompt change**: there's no separate "compile" step — edit the YAML, restart
the server (prompts are assembled once at import time), and send a test query. If you want
to see the exact assembled prompt text without spending an API call, you can import
`agent2.pipeline.prompt_loader` directly:

```python
from agent2.pipeline.prompt_loader import EXTRACT_TEMPLATES, CLASSIFY_SYSTEM
print(EXTRACT_TEMPLATES["DIRECT_LOOKUP"])
```

---

## 4. A safety note

Whatever SQL the model generates is never trusted outright — `agent2/retrieval/sql_generator.py`'s
`validate_sql()` parses it with a real SQL parser (not regex) and rejects anything that
isn't a single, read-only `SELECT` referencing only tables/columns declared in
`entities.yaml`/`attributes.yaml`, with values always passed as bound parameters, never
written into the SQL text. Adding a new entity type never weakens this: the allowlist is
built from the same two config files, so a new table is only ever *addable* to what the
model may query, never a way to bypass the check.
