"""
Compatibility surface for the local store.

The lookup and gap classification now live in LocalStore (local_store.py),
which builds its statements through SqlBuilder. This module keeps the original
function names so the controllers and the KQML layer are unaffected by that
move.
"""
from __future__ import annotations

from typing import Any, Dict

from .local_store import (ALL_STATES, DataRecord, GapSlot, LocalResult,  # noqa: F401
                          LocalStore)
from ..pipeline.query_params import QueryParams

_store = LocalStore()


def execute_local_lookup(params: QueryParams) -> LocalResult:
    """Answer a parsed query from this agent's own partition."""
    return _store.lookup(params)


def execute_local_lookup_from_slots(slot) -> Dict[str, Any]:
    """Answer a peer's KQML ask against this agent's own partition."""
    return _store.lookup_slots(slot)
