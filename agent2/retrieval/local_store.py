"""
This agent's own half of the data, and the gap it cannot fill.

LocalStore answers a parsed query from the local database and classifies what
is missing. The three kinds of gap it distinguishes are the ones the document
defines, and they are found by different tests:

    spatial   the state has no rows here at all          (partition check)
    temporal  the state is held, but not for that year   (no row for the year)
    attribute the row exists and the value is NULL       (row present, value absent)

The distinction matters locally, for diagnosing this store. It does not travel:
all three become the same request to the peer, which is only ever asked for the
values, never for why they were absent here.

Statements come from SqlBuilder so each one is defined once — the form that
runs and the form that gets logged cannot drift apart.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from sqlalchemy import text

from ..database import SessionLocal
from ..pipeline.query_params import QueryParams
from .sql_builder import BuiltQuery, SqlBuilder

log = logging.getLogger("agent2.retrieval.local_store")

ALL_STATES = [
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen",
    "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen",
    "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Sachsen",
    "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen",
]


@dataclass
class DataRecord:
    state: str
    year: int
    values: Dict[str, Any] = field(default_factory=dict)
    source: str = "Agent-2"


@dataclass
class GapSlot:
    """A block of the query this agent could not fill: states x years x attributes."""
    spatial: List[str]
    temporal: List[int]
    attributes: List[str]


@dataclass
class LocalResult:
    found: List[DataRecord] = field(default_factory=list)
    gaps: List[GapSlot] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)


class LocalStore:
    """Reads this agent's partition and reports what it could not answer."""

    def __init__(self, builder: SqlBuilder | None = None):
        self._builder = builder or SqlBuilder()

    def _run(self, db, built: BuiltQuery, queries: List[str]):
        """Execute a built statement and record its readable form for the log."""
        queries.append(built.readable)
        log.debug("SQL > %s\n%s", built.label, built.readable)
        return db.execute(text(built.sql), built.params)

    def lookup(self, params: QueryParams) -> LocalResult:
        states = ALL_STATES if "all" in params.spatial else params.spatial
        years = params.temporal
        attributes = params.attributes

        log.info("       | States to query : %s", states)
        log.info("       | Years           : %s", years)
        log.info("       | Attributes      : %s", attributes)

        queries: List[str] = []
        db = SessionLocal()
        try:
            # ── Which states does this agent hold anything for? ──────────────
            held = set(self._run(db, self._builder.partition_check(states), queries)
                       .scalars().all())
            log.info("       | States found in this partition : %s", sorted(held))

            spatial_gap = [state for state in states if state not in held]
            local_states = [state for state in states if state in held]
            if spatial_gap:
                log.info("       | SPATIAL GAP (not in partition) : %s -> will ask the peer",
                         spatial_gap)

            if not local_states:
                # Nothing of this query lives here; the whole thing is one gap.
                log.info("       | No local states to query - skipping main lookup")
                return LocalResult(
                    found=[],
                    gaps=[GapSlot(spatial=spatial_gap, temporal=years, attributes=attributes)],
                    queries=queries,
                )

            # ── The values themselves ────────────────────────────────────────
            rows = self._run(
                db, self._builder.demographics(local_states, years, attributes), queries
            ).fetchall()
            log.info("       | Rows returned: %d", len(rows))

            by_slot = {
                (row[0], row[1]): {attributes[i]: row[2 + i] for i in range(len(attributes))}
                for row in rows
            }
            return LocalResult(
                found=self._collect_found(local_states, years, attributes, by_slot),
                gaps=self._classify_gaps(local_states, years, attributes, by_slot, spatial_gap),
                queries=queries,
            )
        finally:
            db.close()

    @staticmethod
    def _collect_found(states, years, attributes, by_slot) -> List[DataRecord]:
        """Keep whatever value is actually present, even in a row that is only
        partly filled — a NULL beside a real number does not discard both."""
        found: List[DataRecord] = []
        for state in states:
            for year in years:
                values = by_slot.get((state, year))
                if not values:
                    continue
                present = {k: v for k, v in values.items() if v is not None}
                if present:
                    log.info("       | FOUND        : %s year=%d -> %s", state, year, present)
                    found.append(DataRecord(state=state, year=year, values=present))
        return found

    @staticmethod
    def _classify_gaps(states, years, attributes, by_slot, spatial_gap) -> List[GapSlot]:
        """Group what is missing into as few request blocks as possible.

        Entries combine only when the block they describe is the same shape:
        states missing the same years and attributes travel together, states
        missing different years cannot."""
        attribute_gaps: Dict[tuple, List[int]] = {}
        temporal_gaps: Dict[str, List[int]] = {}

        for state in states:
            for year in years:
                values = by_slot.get((state, year))
                if values is None:
                    log.info("       | TEMPORAL GAP : %s year=%d (no row)", state, year)
                    temporal_gaps.setdefault(state, []).append(year)
                    continue
                missing = [a for a in attributes if values.get(a) is None]
                if missing:
                    log.info("       | ATTR GAP     : %s year=%d (NULL: %s)", state, year, missing)
                    attribute_gaps.setdefault((state, tuple(missing)), []).append(year)

        gaps = [GapSlot(spatial=[state], temporal=years_missing, attributes=list(attrs))
                for (state, attrs), years_missing in attribute_gaps.items()]
        gaps += [GapSlot(spatial=[state], temporal=years_missing, attributes=attributes)
                 for state, years_missing in temporal_gaps.items()]
        if spatial_gap:
            gaps.append(GapSlot(spatial=spatial_gap, temporal=years, attributes=attributes))

        log.info("       | Summary: attr_gaps=%d  temp_gaps=%d  spatial_gaps=%d",
                 len(attribute_gaps), len(temporal_gaps), len(spatial_gap))
        return gaps

    def lookup_slots(self, slot) -> Dict[str, Any]:
        """Answer a peer's KQML ask against this store.

        The reply reports the residue per state — the years actually not found
        — rather than echoing the whole requested range back, so the asking
        agent learns what is genuinely absent (gap = Q \\ D, not Q)."""
        states = slot.spatial if isinstance(slot.spatial, list) else [slot.spatial]
        years = slot.temporal
        attributes = slot.attributes

        queries: List[str] = []
        db = SessionLocal()
        try:
            rows = self._run(
                db, self._builder.demographics(states, years, attributes), queries
            ).fetchall()

            found, satisfied = [], set()
            for row in rows:
                state, year = row[0], row[1]
                values = {attributes[i]: row[2 + i] for i in range(len(attributes))}
                present = {k: v for k, v in values.items() if v is not None}
                if present:
                    found.append({"spatial": state, "year": year, **present})
                    satisfied.add((state, year))

            missing_by_state = {
                state: [year for year in years if (state, year) not in satisfied]
                for state in states
            }
            missing_by_state = {k: v for k, v in missing_by_state.items() if v}

            return {"found": found,
                    "missing": list(missing_by_state),
                    "missing_by_state": missing_by_state,
                    "queries": queries}
        finally:
            db.close()
