"""
Canonical entity-name gazetteer — the single source of truth for German state
and city name spellings used across the extraction prompts, and for cleaning
whatever the LLM extracts to match what is actually stored in the database.

The actual name/alias data lives in name_aliases.json, not in this file — this
module only loads it and exposes the lookup functions. Editing a spelling or
adding a new alias is a JSON edit, not a code change.

Verified directly against Agent-2's `states.state_name` and `cities.city_name`
columns (2026-09-02), not assumed — and confirmed identical to Agent-1's data:
  - The 16 state names are stored with full umlauts (Baden-Württemberg, ...).
  - Most cities are stored in their English form: Munich, Cologne, Nuremberg.
  - Düsseldorf is the one exception — it keeps its German umlaut spelling in
    the DB, unlike Munich/Cologne/Nuremberg.

Lookup keys are normalized (lowercased, diacritics stripped, punctuation and
whitespace removed) before matching, so the JSON only needs to list genuinely
different words (Munich vs München) — not every case/hyphenation/umlaut-vs-
plain-vowel variant of the same word (München / MÜNCHEN / Munchen all fold to
the same key without a separate JSON entry). This is also what lets the
extraction prompts stop insisting on one exact spelling: whatever form the
LLM outputs gets folded to the same key as the database's own spelling.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "name_aliases.json"
_data = json.loads(_DATA_PATH.read_text(encoding="utf-8"))

GERMAN_STATES: list[str] = _data["states"]


def _normalize_key(text: str) -> str:
    """Fold a name down to a bare matching key: lowercase, strip diacritics
    (München -> munchen), fold ß -> ss, fold the ue/oe/ae ASCII transliteration
    of an umlaut down to its plain vowel (Wuerttemberg -> Wurttemberg, same key
    as Württemberg), and drop everything that isn't a letter or digit (spaces,
    hyphens, punctuation). Two names that differ only in case, hyphenation, or
    umlaut spelling fold to the same key — that's what keeps name_aliases.json
    from having to enumerate every spelling variant of the same word."""
    text = text.replace("ß", "ss")
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_folded = re.sub(r"ue", "u", ascii_folded, flags=re.IGNORECASE)
    ascii_folded = re.sub(r"oe", "o", ascii_folded, flags=re.IGNORECASE)
    ascii_folded = re.sub(r"ae", "a", ascii_folded, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", ascii_folded.lower())


# name_aliases.json is keyed by what the database actually stores — e.g.
# "Munich": ["münchen", "muenchen", "munich"] — so the file itself documents
# ground truth first. Build the normalized-alias -> canonical lookup used at
# runtime; the canonical DB spelling is also indexed against itself so a
# perfectly-spelled input still matches without needing its own JSON entry.
_CITY_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in _data["cities"].items():
    _CITY_CANONICAL[_normalize_key(_canonical)] = _canonical
    for _alias in _aliases:
        _CITY_CANONICAL[_normalize_key(_alias)] = _canonical

_STATE_CANONICAL: dict[str, str] = {_normalize_key(s): s for s in GERMAN_STATES}

# Longest name any gazetteer entry can spell out as separate words (e.g. a
# hyphen-free "Nordrhein Westfalen") — CleanQuery uses this to size its
# word-window scan without having to duplicate this data itself.
MAX_NAME_WORDS: int = max(
    max(len(s.replace("-", " ").split()) for s in GERMAN_STATES),
    max(len(c.replace("-", " ").split()) for c in _data["cities"]),
)

STATE_NAMES_BLOCK = (
    "German states (any reasonable spelling/capitalization is fine — names are "
    "normalized against the database automatically):\n"
    + ", ".join(GERMAN_STATES)
)

CITY_NAMES_BLOCK = (
    "City names: any reasonable spelling, capitalization, or umlaut-vs-plain-vowel "
    "form is fine (e.g. München/Muenchen/Munich all resolve the same way) — names "
    "are normalized against the database automatically after extraction."
)


def lookup_canonical(text: str) -> str | None:
    """Return the canonical DB spelling if `text`, taken as a whole phrase,
    matches a known state or city name/alias after normalization — else None.
    Used by CleanQuery to test word-windows of a raw query against the
    gazetteer without needing to know entity_type up front."""
    if not text:
        return None
    key = _normalize_key(text)
    return _STATE_CANONICAL.get(key) or _CITY_CANONICAL.get(key)


def normalize_entity_name(name: str, entity_type: str = "") -> str:
    """Map any recognized alias/variant of a state or city name to the exact
    spelling stored in the database. Unrecognized names are returned
    unchanged — the DB layer's own lookup fallback still applies to those.

    `entity_type` narrows the lookup to "state" or "city" when known; pass ""
    (or an unrecognized value) to check both tables."""
    if not name:
        return name
    key = _normalize_key(name)
    if entity_type == "state":
        return _STATE_CANONICAL.get(key, name)
    if entity_type == "city":
        return _CITY_CANONICAL.get(key, name)
    return _STATE_CANONICAL.get(key) or _CITY_CANONICAL.get(key) or name
