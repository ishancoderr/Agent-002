"""
Loads the stage-1 classify prompt and the stage-2 extraction prompts from
config/prompts/*.yaml and assembles each into the system-prompt text
query_classifier.py / query_extractor.py send to the model.

Adding a category is a YAML file, not a code change: drop a new file into
config/prompts/ with its own `query_type` (a string, or a list of strings if
several categories share one template — see spatial_relationship.yaml) and a
`sections` list naming, in order, which of the renderers below to use. If an
existing renderer covers everything the new prompt needs, nothing here has
to change at all — same "copy a block, no Python changes" philosophy as
config/schema/entities.yaml. A genuinely new kind of content (not prose, a
list, the output schema, or examples) needs one new renderer function and one
line added to _RENDERERS; it does not need changes anywhere else.

classify.yaml is the one file in this directory with no `query_type` — it's
not a per-category template, it's the prompt that decides the category — so
_load_all() (which builds EXTRACT_TEMPLATES) skips it, and
load_classify_system() reads it explicitly instead. It reuses the same
`sections`/_RENDERERS mechanism, plus two renderers (`categories`,
`classify_examples`) specific to its own shape.

The attribute list and alias rules ("married" -> "marriages") are never
authored by hand in a prompt file. They are generated here from
config/schema/attributes.yaml for whichever entity a template names via
`attributes_entity`, so a column's name and aliases are spelled once
regardless of how many prompt categories need them. State and city name
spellings are generated the same way, from the gazetteer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

from . import gazetteer

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_DIR = _REPO_ROOT / "config" / "prompts"
_ATTRIBUTES_PATH = _REPO_ROOT / "config" / "schema" / "attributes.yaml"


def _load_attribute_rules(entity: str) -> str:
    """Build the "Valid attributes: ..." block for `entity` from
    config/schema/attributes.yaml: a header naming every column, then one
    bullet per column that declares aliases, mapping every recognized phrase
    to the canonical name the extractor must output. Collects every
    attribute_tables entry for `entity`, not just the first one found — an
    entity can have more than one attribute table."""
    data = yaml.safe_load(_ATTRIBUTES_PATH.read_text(encoding="utf-8"))
    tables = [t for t in data["attribute_tables"] if t["entity"] == entity]
    if not tables:
        raise KeyError(f"No attribute_tables entry for entity={entity!r} in {_ATTRIBUTES_PATH}")
    lines: List[str] = []
    for table in tables:
        columns = table["columns"]
        lines.append(f"Valid attributes: {', '.join(columns)}")
        for canonical, spec in columns.items():
            aliases = spec.get("aliases") or []
            if not aliases:
                continue
            quoted = ", ".join(f'"{a}"' for a in aliases)
            lines.append(f'  - {quoted} → always output "{canonical}"')
    return "\n".join(lines)


def _all_attribute_entities() -> List[str]:
    """Every entity type with at least one attribute_tables entry, in
    config/schema/attributes.yaml's own order — what `attributes_entity:
    all` iterates over, so a template listing every attribute-bearing
    entity's valid columns (direct_lookup.yaml: a plain data query could be
    about any of them) never needs a prompt or Python change when a new one
    is added; it only needs its own attributes.yaml entry."""
    data = yaml.safe_load(_ATTRIBUTES_PATH.read_text(encoding="utf-8"))
    seen: List[str] = []
    for table in data["attribute_tables"]:
        if table["entity"] not in seen:
            seen.append(table["entity"])
    return seen


# ── Section renderers ─────────────────────────────────────────────────────
# Each takes the parsed YAML doc and returns a list of text blocks to splice
# into the assembled prompt (usually one block; gazetteer can return several).

def _render_plain(field: str) -> Callable[[Dict[str, Any]], List[str]]:
    """Build a renderer for a single string field: look it up in the doc at
    render time and pass it through unchanged, or emit nothing if absent."""
    def render(doc: Dict[str, Any]) -> List[str]:
        """Return doc[field] as a one-item list, or [] if it's missing/empty."""
        value = doc.get(field)
        return [value] if value else []
    return render


def _render_gazetteer(doc: Dict[str, Any]) -> List[str]:
    """Render one names block per entity type in doc["gazetteer"]: an
    explicit list (e.g. ["state", "city"]) when a category is only ever
    about those specific types — spatial_relationship_buffer.yaml's ["city"]
    is a fact about what that category means, not something a new entity
    type should widen — or one of two sentinels:

    - "all": every enabled entity type in entities.yaml, for a category
      that can genuinely name any of them (a GEOMETRY_LOOKUP can ask about
      any entity that has a shape).
    - "attributes": every entity type that has at least one attribute table
      in attributes.yaml (direct_lookup.yaml's own case) — deliberately
      narrower than "all": a plain data query can only be about an entity
      that actually has data, so an entity that exists (e.g. today, city)
      but has nothing to look up never appears here, which would otherwise
      let the model extract an entity_type with no attributes.yaml table
      to answer it from.

    Which entity types exist, and their prompt wording, live in
    config/schema/entities.yaml (via gazetteer.get_names_block) — not a
    dict here, so a new entity type there needs no change in this file."""
    spec = doc.get("gazetteer", [])
    if spec == "all":
        kinds = gazetteer.ENABLED_ENTITY_TYPES
    elif spec == "attributes":
        kinds = _all_attribute_entities()
    else:
        kinds = spec
    return [gazetteer.get_names_block(kind) for kind in kinds]


def _render_attributes(doc: Dict[str, Any]) -> List[str]:
    """Render the "Valid attributes" block(s) for doc["attributes_entity"]:

    - a single entity name (e.g. spatial_relationship.yaml's "state") when
      the category is inherently about one specific entity — a fact about
      what that category means (adjacency/direction/distance only ever
      resolve states), not something a new attribute table should change.
    - the literal "all" (direct_lookup.yaml) when the category is a plain
      data lookup that could be about any attribute-bearing entity: every
      one of them is rendered, each under its own "<Entity> attributes:"
      header once there's more than one, so a new entity with its own
      attributes.yaml table appears here automatically.

    doc["attribute_rules"] (if any) is tacked on as one more bullet, once,
    after every entity's block."""
    entity = doc.get("attributes_entity")
    if not entity:
        return []
    entities = _all_attribute_entities() if entity == "all" else [entity]
    blocks = []
    for e in entities:
        block = _load_attribute_rules(e)
        if len(entities) > 1:
            block = f"{e.capitalize()} attributes:\n{block}"
        blocks.append(block)
    combined = "\n\n".join(blocks)
    extra = doc.get("attribute_rules")
    if extra:
        combined += "\n  - " + extra
    return [combined]


def _render_operations(doc: Dict[str, Any]) -> List[str]:
    """Render doc["operations_intro"] followed by one bullet per entry in
    doc["operations"], as a single block (used by SPATIAL_OPERATION)."""
    parts = []
    if doc.get("operations_intro"):
        parts.append(doc["operations_intro"])
    parts.append("\n".join(f"  - {op}" for op in doc.get("operations", [])))
    return ["\n".join(parts)]


def _render_entity_type_rules(doc: Dict[str, Any]) -> List[str]:
    """Render doc["entity_type_rules_intro"] followed by one bullet per
    entry in doc["entity_type_rules"], as a single block."""
    parts = []
    if doc.get("entity_type_rules_intro"):
        parts.append(doc["entity_type_rules_intro"])
    parts.append("\n".join(f"  - {r}" for r in doc.get("entity_type_rules", [])))
    return ["\n".join(parts)]


def _render_output_schema(doc: Dict[str, Any]) -> List[str]:
    """Render doc["output_schema"] under an "Output exactly this shape:"
    header.

    output_schema is stored as literal text, not structured YAML fed through
    json.dumps — these blocks are illustrative pseudo-JSON ("adjacency" |
    "north_of" | ..., ["Bayern"] or "all"), not real JSON, so a generic
    serializer would either reject them or (as a first pass here did)
    silently drop the enum/"or" alternatives if they were tucked into YAML
    comments. Keeping the block as text is what actually preserves them."""
    schema = doc.get("output_schema")
    if not schema:
        return []
    return ["Output exactly this shape:\n" + schema.rstrip()]


def _render_examples(doc: Dict[str, Any]) -> List[str]:
    """Render doc["examples"] as "Query: ...\n→ {json}" pairs under an
    EXAMPLES: header — the format every extraction template's examples use."""
    examples = doc.get("examples")
    if not examples:
        return []
    lines = ["EXAMPLES:"]
    for ex in examples:
        lines.append("")
        lines.append(f"Query: {ex['query']}")
        lines.append("→ " + json.dumps(ex["output"], ensure_ascii=False, separators=(",", ":")))
    return ["\n".join(lines)]


def _render_categories(doc: Dict[str, Any]) -> List[str]:
    """Render doc["categories_intro"] followed by every entry in
    doc["categories"] as a bullet, classify.yaml's own section.

    Unlike entity_type_rules/operations, each category bullet in the
    original prompt has a blank line between it and the next one (they're
    denser, multi-sentence definitions) — so this joins with "\n\n", not
    the single "\n" the other bullet-list renderers use."""
    parts = []
    if doc.get("categories_intro"):
        parts.append(doc["categories_intro"])
    bullets = "\n\n".join(f"- {c}" for c in doc.get("categories", []))
    parts.append(bullets)
    return ["\n\n".join(parts)]


def _render_classify_examples(doc: Dict[str, Any]) -> List[str]:
    """Render doc["examples"] as '"query text"\n→ {json}' pairs under an
    EXAMPLES: header — classify.yaml's own examples format.

    classify.yaml's examples quote the query directly ("query text") rather
    than prefixing it with "Query: " — a different original format from the
    extraction templates' examples, so this can't reuse _render_examples."""
    examples = doc.get("examples")
    if not examples:
        return []
    lines = ["EXAMPLES:"]
    for ex in examples:
        lines.append("")
        lines.append(f'"{ex["query"]}"')
        lines.append("→ " + json.dumps(ex["output"], ensure_ascii=False, separators=(",", ":")))
    return ["\n".join(lines)]


_RENDERERS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "intro": _render_plain("intro"),
    "gazetteer": _render_gazetteer,
    "attributes": _render_attributes,
    "operations": _render_operations,
    "entity_type_rules": _render_entity_type_rules,
    "temporal_rules": _render_plain("temporal_rules"),
    "subject_rules": _render_plain("subject_rules"),
    "output_schema": _render_output_schema,
    "output_notes": _render_plain("output_notes"),
    "examples": _render_examples,
    "categories": _render_categories,
    "grammatical_note": _render_plain("grammatical_note"),
    "output_instruction": _render_plain("output_instruction"),
    "classify_examples": _render_classify_examples,
}


def _assemble(doc: Dict[str, Any]) -> str:
    """Run doc["sections"] through _RENDERERS in order and join the results
    into the final prompt text, blank-line-separated with a trailing newline."""
    parts: List[str] = []
    for section in doc["sections"]:
        renderer = _RENDERERS.get(section)
        if renderer is None:
            raise ValueError(
                f"config/prompts file for query_type={doc.get('query_type')!r} names "
                f"unknown section {section!r} — add a renderer to prompt_loader.py's "
                f"_RENDERERS, or fix the typo."
            )
        parts.extend(renderer(doc))
    return "\n\n".join(parts) + "\n"


def _load_all() -> Dict[str, str]:
    """Assemble every config/prompts/*.yaml extraction template and return
    {query_type: assembled prompt text}, covering every query_type it names."""
    templates: Dict[str, str] = {}
    for path in sorted(_PROMPTS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "query_type" not in doc:
            # Not a per-category extraction template (e.g. classify.yaml,
            # which decides the category rather than filling one in) —
            # nothing here to add to EXTRACT_TEMPLATES.
            continue
        prompt = _assemble(doc)
        query_types = doc["query_type"]
        if isinstance(query_types, str):
            query_types = [query_types]
        for qt in query_types:
            templates[qt] = prompt
    return templates


def load_classify_system() -> str:
    """Assemble config/prompts/classify.yaml into the stage-1 classify
    system prompt. Separate from _load_all() since this file isn't keyed by
    query_type — it's the one prompt that decides the query_type."""
    path = _PROMPTS_DIR / "classify.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _assemble(doc)


# Built once at import time — every *.yaml in config/prompts/ is picked up
# automatically, so a new category needs no change here. query_extractor.py
# treats a query_type missing from this dict as an error, not something to
# paper over with a default template — see QueryExtractor.extract().
EXTRACT_TEMPLATES: Dict[str, str] = _load_all()

# The stage-1 classify prompt query_classifier.py sends as its system message.
CLASSIFY_SYSTEM: str = load_classify_system()
