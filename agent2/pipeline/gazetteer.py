"""
Canonical entity-name gazetteer — the single source of truth for German state
and city name spellings used across the extraction prompts, and for cleaning
whatever the LLM extracts to match what is actually stored in the database.

The actual name/alias data lives in config/schema/name_aliases.yaml, not in
this file — this module only loads it and exposes the lookup functions.
Editing a spelling or adding a new alias is a YAML edit, not a code change.

Verified directly against Agent-2's `states.state_name` and `cities.city_name`
columns (2026-09-02), not assumed:
  - The 16 state names are stored with full umlauts (Baden-Württemberg, ...).
  - Most cities are stored in their English form: Munich, Cologne, Nuremberg.
  - Düsseldorf is the one exception — it keeps its German umlaut spelling in
    the DB, unlike Munich/Cologne/Nuremberg.

Lookup keys are normalized (lowercased, diacritics stripped, punctuation and
whitespace removed) before matching, so name_aliases.yaml only needs to list
genuinely different words (Munich vs München) — not every case/hyphenation/
umlaut-vs-plain-vowel variant of the same word (München / MÜNCHEN / Munchen
all fold to the same key without a separate entry). This is also what lets
the extraction prompts stop insisting on one exact spelling: whatever form
the LLM outputs gets folded to the same key as the database's own spelling.

get_names_block() (used by agent2/pipeline/prompt_loader.py to build the
extraction prompts) generates the gazetteer paragraph for any entity type
straight from config/schema/entities.yaml plus this file's own name data,
rather than a hardcoded per-type dict — a new enumerable entity type just
needs a "state"-shaped flat list added under its own key in
name_aliases.yaml and a `gazetteer_block` in entities.yaml; nothing here has
to change.

The canonical-spelling lookups (lookup_canonical, normalize_entity_name) are
built the same way: _CANONICAL_BY_TYPE has one entry per entity type declared
in entities.yaml, built from that type's own name_aliases.yaml data — not two
hardcoded "state"/"city" dicts. A name that could belong to more than one
type (Berlin, Hamburg, Bremen are each both) is resolved using entities.yaml's
own `ambiguous_name_default`, not a hardcoded priority order.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "schema"

_DATA_PATH = _SCHEMA_DIR / "name_aliases.yaml"
_data = yaml.safe_load(_DATA_PATH.read_text(encoding="utf-8"))

_ENTITIES_PATH = _SCHEMA_DIR / "entities.yaml"
_ENTITIES_DOC = yaml.safe_load(_ENTITIES_PATH.read_text(encoding="utf-8"))
_ENTITIES = _ENTITIES_DOC["entities"]
_AMBIGUOUS_DEFAULT = _ENTITIES_DOC.get("ambiguous_name_default")

GERMAN_STATES: list[str] = _data["state"]

# Every enabled entity type, in entities.yaml's own declared order — what
# `gazetteer: all` (config/prompts/direct_lookup.yaml, via
# prompt_loader.py's _render_gazetteer()) iterates over, so a newly-added
# entity type starts appearing there with no prompt change. Same "enabled"
# filter agent2/pipeline/query_params.py's VALID_ENTITY_TYPES uses.
ENABLED_ENTITY_TYPES: list[str] = [
    name for name, spec in _ENTITIES.items() if spec.get("enabled", True)
]


def _normalize_key(text: str) -> str:
    """Fold a name down to a bare matching key: lowercase, strip diacritics
    (München -> munchen), fold ß -> ss, fold the ue/oe/ae ASCII transliteration
    of an umlaut down to its plain vowel (Wuerttemberg -> Wurttemberg, same key
    as Württemberg), and drop everything that isn't a letter or digit (spaces,
    hyphens, punctuation). Two names that differ only in case, hyphenation, or
    umlaut spelling fold to the same key — that's what keeps name_aliases.yaml
    from having to enumerate every spelling variant of the same word."""
    text = text.replace("ß", "ss")
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_folded = re.sub(r"ue", "u", ascii_folded, flags=re.IGNORECASE)
    ascii_folded = re.sub(r"oe", "o", ascii_folded, flags=re.IGNORECASE)
    ascii_folded = re.sub(r"ae", "a", ascii_folded, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", ascii_folded.lower())


def _build_canonical_lookup(raw: "list[str] | dict[str, list[str]]") -> dict[str, str]:
    """Build one entity type's normalized-key -> canonical-spelling lookup
    from its name_aliases.yaml entry. A flat list (e.g. `state`) has no
    aliases, so each name indexes only itself. A dict (e.g. `city`, keyed by
    what the database actually stores, e.g. Munich: [münchen, muenchen,
    munich]) indexes the canonical spelling against itself too, so a
    perfectly-spelled input still matches without needing its own alias
    entry, then indexes every alias alongside it."""
    lookup: dict[str, str] = {}
    if isinstance(raw, dict):
        for canonical, aliases in raw.items():
            lookup[_normalize_key(canonical)] = canonical
            for alias in aliases:
                lookup[_normalize_key(alias)] = canonical
    else:
        for canonical in raw:
            lookup[_normalize_key(canonical)] = canonical
    return lookup


# One lookup per entity type declared in entities.yaml — not two hardcoded
# "state"/"city" dicts, so a third enumerable or aliased type needs only its
# own name_aliases.yaml entry, nothing here.
_CANONICAL_BY_TYPE: dict[str, dict[str, str]] = {
    entity_type: _build_canonical_lookup(_data[entity_type])
    for entity_type in _ENTITIES
    if entity_type in _data
}

# Order to search when the caller doesn't know a name's entity type: try
# entities.yaml's own `ambiguous_name_default` first — the type a name is
# resolved to when, like Berlin/Hamburg/Bremen, it could be more than one —
# then every other declared type, in the order entities.yaml lists them.
_LOOKUP_ORDER: list[str] = (
    ([_AMBIGUOUS_DEFAULT] if _AMBIGUOUS_DEFAULT in _CANONICAL_BY_TYPE else [])
    + [t for t in _CANONICAL_BY_TYPE if t != _AMBIGUOUS_DEFAULT]
)

# Longest name any gazetteer entry can spell out as separate words (e.g. a
# hyphen-free "Nordrhein Westfalen") — CleanQuery uses this to size its
# word-window scan without having to duplicate this data itself.
MAX_NAME_WORDS: int = max(
    len(canonical.replace("-", " ").split())
    for lookup in _CANONICAL_BY_TYPE.values()
    for canonical in lookup.values()
)


def get_names_block(entity_type: str) -> str:
    """Build the gazetteer paragraph for `entity_type` that the stage-2
    extraction prompts show the model (see config/prompts/*.yaml's
    `gazetteer:` list). The static wording comes from entities.yaml's
    `gazetteer_block`; for an enumerable type (a closed set — see
    entities.yaml's own `enumerable` comment) the full comma-joined name
    list is appended here, since that list lives in name_aliases.yaml and
    would go stale if it were copied into entities.yaml's prose too."""
    try:
        entity = _ENTITIES[entity_type]
    except KeyError:
        raise KeyError(
            f"No entity_type={entity_type!r} in config/schema/entities.yaml — "
            f"a config/prompts/*.yaml file's `gazetteer:` list named a type "
            f"that isn't declared there."
        ) from None
    header = entity["gazetteer_block"]
    if not entity.get("enumerable"):
        return header
    names = _data[entity_type]
    return header + "\n" + ", ".join(names)


def lookup_canonical(text: str) -> str | None:
    """Return the canonical DB spelling if `text`, taken as a whole phrase,
    matches a known name/alias of any entity type after normalization — else
    None. Used by CleanQuery to test word-windows of a raw query against the
    gazetteer without needing to know entity_type up front. Checked in
    _LOOKUP_ORDER (entities.yaml's `ambiguous_name_default` first)."""
    if not text:
        return None
    key = _normalize_key(text)
    for entity_type in _LOOKUP_ORDER:
        canonical = _CANONICAL_BY_TYPE[entity_type].get(key)
        if canonical:
            return canonical
    return None


def normalize_entity_name(name: str, entity_type: str = "") -> str:
    """Map any recognized alias/variant of an entity name to the exact
    spelling stored in the database. Unrecognized names are returned
    unchanged — the DB layer's own ILIKE fallback still applies to those.

    `entity_type` narrows the lookup to one declared in entities.yaml (e.g.
    "state" or "city") when known; pass "" (or an unrecognized value) to
    check every type, in _LOOKUP_ORDER."""
    if not name:
        return name
    if entity_type in _CANONICAL_BY_TYPE:
        key = _normalize_key(name)
        return _CANONICAL_BY_TYPE[entity_type].get(key, name)
    return lookup_canonical(name) or name
