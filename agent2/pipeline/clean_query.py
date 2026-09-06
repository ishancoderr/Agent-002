"""
CleanQuery — normalizes a raw user query BEFORE it ever reaches the
classifier or extractor. Known state/city name mentions (in any alias,
spelling, capitalization, or hyphenation form) are rewritten to their exact
database spelling directly in the query text, using the gazetteer built from
name_aliases.json.

Doing this once, up front, means _classify() and _extract() only ever see
canonical names — the LLM's job becomes "copy this name" rather than
"invent the right spelling", which is far more reliable. It also means
parse_query() does not need to repeat entity-name cleanup after every
extraction branch: clean the query once, here, and every downstream step
inherits clean names.
"""
from __future__ import annotations

import re

from .gazetteer import MAX_NAME_WORDS, lookup_canonical


class CleanQuery:
    """Wraps the cleaning of a single raw query string.

    Usage:
        cleaned = CleanQuery(raw_query).cleaned
    """

    def __init__(self, raw_query: str):
        self.raw_query = raw_query
        self.cleaned = self._clean(raw_query)

    @staticmethod
    def _clean(text: str) -> str:
        """Collapse whitespace, then scan the query word by word, replacing
        the longest recognized state/city phrase at each position with its
        canonical DB spelling. Greedy longest-match-first so a two-word name
        (e.g. "Nordrhein Westfalen") is caught before its first word alone
        could be mistaken for something else."""
        text = re.sub(r"\s+", " ", text.strip())
        words = text.split(" ")
        n = len(words)
        out: list[str] = []
        i = 0
        while i < n:
            replaced = False
            for size in range(min(MAX_NAME_WORDS, n - i), 0, -1):
                window = words[i:i + size]
                candidate = " ".join(window)
                canonical = lookup_canonical(candidate)
                if canonical:
                    # Preserve trailing punctuation from the last word in the
                    # window (e.g. "Bayern?" -> "Bayern" + "?").
                    trailing = re.search(r"[^\w]+$", window[-1])
                    out.append(canonical + (trailing.group(0) if trailing else ""))
                    i += size
                    replaced = True
                    break
            if not replaced:
                out.append(words[i])
                i += 1
        return " ".join(out)
