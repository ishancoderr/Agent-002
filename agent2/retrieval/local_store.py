"""
This agent's own half of the data, and how it fills the gap it cannot: asks
Agent-1 over KQML, then merges both agents' answers into one result.

LocalStore answers a parsed query from the local database and classifies what
is missing. The three kinds of gap it distinguishes are the ones the document
defines, and they are found by different tests:

    spatial   the state has no rows here at all          (partition check)
    temporal  the state is held, but not for that year   (no row for the year)
    attribute the row exists and the value is NULL       (row present, value absent)

The distinction matters locally, for diagnosing this store. It does not travel:
all three become the same request to the peer, which is only ever asked for the
values, never for why they were absent here.

Statements come from SqlGenerator so each one is defined once — the form that
runs and the form that gets logged cannot drift apart.

answer_query() is the "gap detect and call" step: LocalStore finds what's
missing, answer_query() asks Agent-1 for it (with retry) and merges both
agents' data into one result — query_controller.py calls this one function
instead of sequencing local lookup / KQML ask / merge itself.

Gap detection stays deterministic Python, never an LLM: it's exact
comparison (which (state, year, attribute) triples came back present vs.
missing), not a task with any ambiguity for a model to resolve — routing it
through an LLM would only make it slower and costlier for no more accuracy
than a plain loop already gets, 100% of the time, for free.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from sqlalchemy import text

from ..database import SessionLocal
from ..pipeline.gazetteer import GERMAN_STATES
from ..pipeline.query_params import DEFAULT_ENTITY_TYPE, QueryParams
from .sql_generator import BuiltQuery, SqlGenerator, ENTITY_ALIAS, YEAR_ALIAS

log = logging.getLogger("agent2.retrieval.local_store")

# Re-exported under this name for merger.py's existing import — the actual
# 16 names live once, in config/schema/name_aliases.yaml via gazetteer.py,
# not as a second hardcoded copy here.
ALL_STATES = GERMAN_STATES


@dataclass
class DataRecord:
    """One state's attribute values for one year, and which agent they came
    from — this agent (the default) or, once merged, the peer."""
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
    # Which entities.yaml entity type `spatial` names (e.g. "state", "river").
    # Carried on the wire now via kqml_messaging.MissingSlot's own entity_type
    # field, so the peer knows which table to look in instead of guessing —
    # see answer_query()/lookup_slots() below for where each side reads it.
    entity_type: str = ""


@dataclass
class LocalResult:
    """What LocalStore.lookup() answered: the records it found, the gaps it
    couldn't fill (to send to Agent-1), and every SQL statement it ran
    (for the log — see LocalStore._run())."""
    found: List[DataRecord] = field(default_factory=list)
    gaps: List[GapSlot] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)


@dataclass
class AnsweredQuery:
    """The final, fully-merged answer to a query: this agent's own data plus
    whatever Agent-1 filled in, ready for query_controller.py to format into
    a response. Also carries per-phase timing and the raw LocalResult, the
    same detail the controller used to measure itself across three separate
    calls (local lookup / KQML ask / merge) before they became one call."""
    merged: List[Dict[str, Any]]
    local_result: LocalResult
    kqml_turns: int = 0
    tokens_agent1: int = 0
    kqml_exchanges: List[Dict[str, Any]] = field(default_factory=list)
    phase1_ms: float = 0.0   # local DB lookup (partition check + demographics)
    phase2_ms: float = 0.0   # KQML round trip to Agent-1 (0 if no gaps)
    phase3_ms: float = 0.0   # merge


class LocalStore:
    """Reads this agent's partition and reports what it could not answer."""

    def __init__(self, generator: SqlGenerator | None = None):
        """Reuse a passed-in SqlGenerator, or make a default one."""
        self._generator = generator or SqlGenerator()

    def _run(self, db, built: BuiltQuery, queries: List[str]):
        """Execute a built statement and record its readable form for the log."""
        queries.append(built.readable)
        log.debug("SQL > %s\n%s", built.label, built.readable)
        return db.execute(text(built.sql), built.params)

    def lookup(self, params: QueryParams) -> LocalResult:
        """Answer a parsed query, end to end: check the partition, fetch
        values for what's held, classify what's still missing."""
        states = ALL_STATES if "all" in params.spatial else params.spatial
        years = params.temporal
        attributes = params.attributes

        log.info("       | States to query : %s", states)
        log.info("       | Years           : %s", years)
        log.info("       | Attributes      : %s", attributes)

        # QueryParams.entity_type is set by SPATIAL_OPERATION's own
        # extraction (config/prompts/spatial_operation.yaml); DIRECT_LOOKUP's
        # extraction prompt has no entity_type field yet (see
        # config/prompts/direct_lookup.yaml), so params.entity_type stays
        # None for it and this falls back to entities.yaml's own
        # ambiguous_name_default — one config value, not a second "state"
        # literal invented here. check_partition()/fetch_demographics()
        # themselves take entity_type as a real parameter either way, not a
        # hardcoded assumption — a caller with a city (or any other
        # entities.yaml type) already works once something upstream of here
        # actually determines one.
        entity_type = params.entity_type or DEFAULT_ENTITY_TYPE

        queries: List[str] = []
        db = SessionLocal()
        try:
            local_states, spatial_gap = self.check_partition(db, states, queries, entity_type)
            if not local_states:
                # Nothing of this query lives here; the whole thing is one gap.
                log.info("       | No local states to query - skipping main lookup")
                return LocalResult(
                    found=[],
                    gaps=[GapSlot(spatial=spatial_gap, temporal=years, attributes=attributes, entity_type=entity_type)],
                    queries=queries,
                )

            by_slot = self.fetch_demographics(db, local_states, years, attributes, queries, entity_type)
            return LocalResult(
                found=self.collect_found(local_states, years, attributes, by_slot),
                gaps=self.classify_gaps(local_states, years, attributes, by_slot, spatial_gap, entity_type),
                queries=queries,
            )
        finally:
            db.close()

    def check_partition(self, db, keys: List[str], queries: List[str], entity_type: str):
        """Which of `keys` this agent holds any row for at all, for the
        given entities.yaml entity_type — generates and runs
        SqlGenerator.partition_check(). Returns (local_keys, spatial_gap):
        the requested keys split into held vs. not-held-at-all. A key with
        no rows here is a spatial gap, a different thing from a key whose
        rows exist but are empty (see fetch_demographics)."""
        held = set(self._run(db, self._generator.partition_check(keys, entity_type), queries)
                   .scalars().all())
        log.info("       | Found in this partition : %s", sorted(held))

        spatial_gap = [key for key in keys if key not in held]
        local_keys = [key for key in keys if key in held]
        if spatial_gap:
            log.info("       | SPATIAL GAP (not in partition) : %s -> will ask the peer",
                     spatial_gap)
        return local_keys, spatial_gap

    def fetch_demographics(self, db, keys: List[str], years: List[int], attributes: List[str],
                           queries: List[str], entity_type: str) -> Dict[tuple, Dict[str, Any]]:
        """The values themselves, for `keys` already known to be held here,
        for the given entities.yaml entity_type — generates and runs
        SqlGenerator.demographics(). Returns {(key, year): {attribute:
        value}}, read by column name rather than position: the query was
        generated by an LLM (see SqlGenerator), and its column order isn't
        something to trust blindly. ENTITY_ALIAS/YEAR_ALIAS/each attribute
        are the exact aliases SqlGenerator.demographics() requires in its
        request — the same constants it builds that request from, not
        independently retyped here."""
        rows = self._run(
            db, self._generator.demographics(keys, years, attributes, entity_type), queries
        ).fetchall()
        log.info("       | Rows returned: %d", len(rows))
        return {
            (row._mapping[ENTITY_ALIAS], row._mapping[YEAR_ALIAS]):
                {attribute: row._mapping[attribute] for attribute in attributes}
            for row in rows
        }

    @staticmethod
    def collect_found(states, years, attributes, by_slot) -> List[DataRecord]:
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
    def classify_gaps(states, years, attributes, by_slot, spatial_gap, entity_type: str) -> List[GapSlot]:
        """Group what is missing into as few request blocks as possible.

        Entries combine only when the block they describe is the same shape:
        states missing the same years and attributes travel together, states
        missing different years cannot. This is exact set-comparison logic,
        deliberately never routed through an LLM (see this module's own
        docstring) — there's nothing here for a model to interpret, only
        arithmetic a loop already gets right every time."""
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

        gaps = [GapSlot(spatial=[state], temporal=years_missing, attributes=list(attrs), entity_type=entity_type)
                for (state, attrs), years_missing in attribute_gaps.items()]
        gaps += [GapSlot(spatial=[state], temporal=years_missing, attributes=attributes, entity_type=entity_type)
                 for state, years_missing in temporal_gaps.items()]
        if spatial_gap:
            gaps.append(GapSlot(spatial=spatial_gap, temporal=years, attributes=attributes, entity_type=entity_type))

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

        # `slot` arrives as kqml_messaging's MissingSlot, whose entity_type
        # field the asking peer set to its own params.entity_type (see
        # answer_query() below) — read it the same way lookup() reads its
        # own params.entity_type, falling back to the same config default
        # only for a slot old enough to predate this field (entity_type
        # defaults to None on the wire, same as here).
        entity_type = getattr(slot, "entity_type", None) or DEFAULT_ENTITY_TYPE

        queries: List[str] = []
        db = SessionLocal()
        try:
            rows = self._run(
                db, self._generator.demographics(states, years, attributes, entity_type), queries
            ).fetchall()

            found, satisfied = [], set()
            for row in rows:
                # By column name, not position — see the matching comment in
                # fetch_demographics(); ENTITY_ALIAS/YEAR_ALIAS again, not
                # retyped literals.
                state, year = row._mapping[ENTITY_ALIAS], row._mapping[YEAR_ALIAS]
                values = {attribute: row._mapping[attribute] for attribute in attributes}
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


def answer_query(params: QueryParams, store: LocalStore | None = None) -> AnsweredQuery:
    """Answer a parsed query end to end — the "gap detect and call" step:

      1. Check this agent's own data (LocalStore.lookup()) and work out
         what's missing.
      2. If anything's missing, ask Agent-1 over KQML for it, retrying up
         to 3 times with backoff if the peer doesn't answer.
      3. Merge this agent's own records with whatever Agent-1 returned into
         one result.

    query_controller.py calls this one function instead of sequencing local
    lookup / KQML ask / merge itself, the way it used to.

    Both imports below are deliberately local to this function, not at the
    top of the file: agent2.result.merger imports DataRecord from this same
    module, and agent2.messaging.kqml_client imports GapSlot from it too —
    importing either of them back at module level here would be a circular
    import. Deferred to call time, by which point both modules have already
    finished initializing, the cycle never actually happens."""
    from ..messaging import send_kqml_ask
    from ..result.merger import merge_results

    store = store or LocalStore()

    t0 = time.perf_counter()
    result = store.lookup(params)
    t1 = time.perf_counter()

    kqml_turns = 0
    tokens_agent1 = 0
    agent2_data: List[Dict[str, Any]] = []
    kqml_exchanges: List[Dict[str, Any]] = []

    if result.gaps:
        log.info("       | Gaps detected (%d) - sending KQML ask to Agent-1 ...", len(result.gaps))
        for attempt in range(1, 4):
            try:
                resp = send_kqml_ask(result.gaps)
                kqml_turns = 1
                agent2_data = resp.get("found", [])
                tokens_agent1 = resp.get("tokens_agent1", 0)
                if "ask_message" in resp and "tell_message" in resp:
                    kqml_exchanges.append({"ask": resp["ask_message"], "tell": resp["tell_message"]})
                log.info("       | KQML tell received from Agent-1 (attempt %d) - %d record(s)",
                         attempt, len(agent2_data))
                break
            except Exception as exc:
                log.warning("       | Agent-1 attempt %d failed - %s", attempt, exc)
                if attempt < 3:
                    time.sleep(1.0 * attempt)
                else:
                    log.warning("       | Agent-1 unreachable after 3 attempts - gaps unresolved")
    else:
        log.info("       | No gaps - Agent-1 not needed")
    t2 = time.perf_counter()

    merged = merge_results(
        result.found, agent2_data,
        requested_states=params.spatial, requested_years=params.temporal,
        requested_attrs=params.attributes,
    )
    t3 = time.perf_counter()

    return AnsweredQuery(
        merged=merged, local_result=result, kqml_turns=kqml_turns, tokens_agent1=tokens_agent1,
        kqml_exchanges=kqml_exchanges,
        phase1_ms=(t1 - t0) * 1000, phase2_ms=(t2 - t1) * 1000, phase3_ms=(t3 - t2) * 1000,
    )


# ── Compatibility functions for callers that just want one call ────────────
# (query_controller.py uses answer_query() directly instead, for the full
# local+peer+merge flow; kqml_controller.py — answering a PEER's ask, never
# calling out to Agent-1 itself — has no equivalent of answer_query() to use.)

_default_store = LocalStore()


def execute_local_lookup_from_slots(slot) -> Dict[str, Any]:
    """Answer a peer's KQML ask against this agent's own partition."""
    return _default_store.lookup_slots(slot)
