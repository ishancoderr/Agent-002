"""
Execute the local SQL lookup and classify every (state, year) outcome as:
  - found         : row exists AND all requested attributes are non-NULL
  - attribute_gap : row exists BUT some attributes are NULL
  - temporal_gap  : no row at all for a year in a state Agent-2 does have
  - spatial_gap   : state has zero rows in Agent-2's state_demographics at all
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from sqlalchemy import text

from ..database import SessionLocal
from ..pipeline.query_parser import QueryParams

log = logging.getLogger("agent2.retrieval.gap_detector")

ALL_STATES = [
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen",
    "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen",
    "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Sachsen",
    "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen",
]


def _print_sql(label: str, sql: str, params: Dict) -> None:
    param_str = "  ".join(f"{k}={v}" for k, v in params.items())
    log.debug("SQL ▶ %s | %s\n%s", label, param_str, sql.strip())


@dataclass
class DataRecord:
    state: str
    year: int
    values: Dict[str, Any] = field(default_factory=dict)
    source: str = "Agent-2"


@dataclass
class GapSlot:
    spatial: List[str]
    temporal: List[int]
    attributes: List[str]


@dataclass
class LocalResult:
    found: List[DataRecord] = field(default_factory=list)
    gaps: List[GapSlot] = field(default_factory=list)


def execute_local_lookup(params: QueryParams) -> LocalResult:
    states = ALL_STATES if "all" in params.spatial else params.spatial
    years  = params.temporal
    attrs  = params.attributes

    log.info("       │ States to query : %s", states)
    log.info("       │ Years           : %s", years)
    log.info("       │ Attributes      : %s", attrs)

    db = SessionLocal()
    try:
        attr_cols  = ",\n               ".join(f"sd.{a}" for a in attrs)
        states_arr = ", ".join(f"'{s}'" for s in states)
        years_arr  = ", ".join(str(y) for y in years)

        # ── Step A: Partition check first ─────────────────────────────────────
        sql_partition = f"""SELECT DISTINCT s.state_name
FROM state_demographics sd
JOIN states s ON s.state_id = sd.state_id
WHERE s.state_name = ANY(ARRAY[{states_arr}]);"""

        _print_sql("STEP A — Partition check (which states exist in Agent-2)", sql_partition, {
            "states": states,
        })

        present = db.execute(
            text("""
                SELECT DISTINCT s.state_name
                FROM state_demographics sd
                JOIN states s ON s.state_id = sd.state_id
                WHERE s.state_name = ANY(:states)
            """),
            {"states": states},
        ).scalars().all()
        states_in_db = set(present)

        log.info("       │ States found in Agent-2 partition : %s", sorted(states_in_db))

        spatial_gap_states = [s for s in states if s not in states_in_db]
        local_states       = [s for s in states if s in states_in_db]

        if spatial_gap_states:
            log.info("       │ SPATIAL GAP (not in partition)    : %s → will ask Agent-1", spatial_gap_states)
        if not local_states:
            log.info("       │ No local states to query — skipping main lookup")
            return LocalResult(
                found=[],
                gaps=[GapSlot(spatial=spatial_gap_states, temporal=years, attributes=attrs)],
            )

        # ── Step B: Main data lookup only for states in this partition ────────
        local_states_arr = ", ".join(f"'{s}'" for s in local_states)

        sql_main = f"""SELECT s.state_name,
               sd.stat_year,
               {attr_cols}
FROM state_demographics sd
JOIN states s ON s.state_id = sd.state_id
WHERE s.state_name = ANY(ARRAY[{local_states_arr}])
  AND sd.stat_year  = ANY(ARRAY[{years_arr}])
ORDER BY s.state_name, sd.stat_year;"""

        _print_sql("STEP B — Main data lookup (local states only)", sql_main, {
            "states": local_states,
            "years":  years,
        })

        rows = db.execute(
            text(f"""
                SELECT s.state_name, sd.stat_year,
                       {", ".join(f"sd.{a}" for a in attrs)}
                FROM state_demographics sd
                JOIN states s ON s.state_id = sd.state_id
                WHERE s.state_name = ANY(:states)
                  AND sd.stat_year  = ANY(:years)
                ORDER BY s.state_name, sd.stat_year
            """),
            {"states": local_states, "years": years},
        ).fetchall()

        log.info("       │ Rows returned: %d", len(rows))

        result_map: Dict[tuple, Dict] = {}
        for row in rows:
            state, yr = row[0], row[1]
            result_map[(state, yr)] = {attrs[i]: row[2 + i] for i in range(len(attrs))}

        found: List[DataRecord] = []
        attr_gaps: Dict[tuple, List[int]] = {}
        temp_gaps: Dict[str, List[int]] = {}

        for state in local_states:
            for yr in years:
                key = (state, yr)
                if key not in result_map:
                    log.info("       │ TEMPORAL GAP : %s year=%d (no row)", state, yr)
                    temp_gaps.setdefault(state, []).append(yr)
                else:
                    vals       = result_map[key]
                    null_attrs = [a for a in attrs if vals.get(a) is None]
                    present    = {k: v for k, v in vals.items() if v is not None}

                    if present:
                        log.info("       │ FOUND        : %s year=%d → %s", state, yr, present)
                        found.append(DataRecord(state=state, year=yr, values=present))

                    if null_attrs:
                        log.info("       │ ATTR GAP     : %s year=%d (NULL: %s)", state, yr, null_attrs)
                        gap_key = (state, tuple(null_attrs))
                        attr_gaps.setdefault(gap_key, []).append(yr)

        gaps: List[GapSlot] = []
        for (state, missing_attrs), yrs in attr_gaps.items():
            gaps.append(GapSlot(spatial=[state], temporal=yrs, attributes=list(missing_attrs)))
        for state, yrs in temp_gaps.items():
            gaps.append(GapSlot(spatial=[state], temporal=yrs, attributes=attrs))
        if spatial_gap_states:
            gaps.append(GapSlot(spatial=spatial_gap_states, temporal=years, attributes=attrs))

        log.info("       │ Summary: found=%d  attr_gaps=%d  temp_gaps=%d  spatial_gaps=%d",
                 len(found), len(attr_gaps), len(temp_gaps), len(spatial_gap_states))

        return LocalResult(found=found, gaps=gaps)

    finally:
        db.close()


def execute_local_lookup_from_slots(slot) -> Dict[str, Any]:
    """Used when Agent 1 sends a KQML ask TO Agent 2 (bidirectional)."""
    states    = slot.spatial if isinstance(slot.spatial, list) else [slot.spatial]
    years     = slot.temporal
    attrs     = slot.attributes

    attr_cols  = ",\n               ".join(f"sd.{a}" for a in attrs)
    states_arr = ", ".join(f"'{s}'" for s in states)
    years_arr  = ", ".join(str(y) for y in years)

    sql = f"""SELECT s.state_name,
               sd.stat_year,
               {attr_cols}
FROM state_demographics sd
JOIN states s ON s.state_id = sd.state_id
WHERE s.state_name = ANY(ARRAY[{states_arr}])
  AND sd.stat_year  = ANY(ARRAY[{years_arr}]);"""

    _print_sql("Bidirectional lookup (Agent-1 → Agent-2)", sql, {
        "states": states,
        "years":  years,
    })

    db = SessionLocal()
    try:
        rows = db.execute(
            text(f"""
                SELECT s.state_name, sd.stat_year,
                       {", ".join(f"sd.{a}" for a in attrs)}
                FROM state_demographics sd
                JOIN states s ON s.state_id = sd.state_id
                WHERE s.state_name = ANY(:states)
                  AND sd.stat_year  = ANY(:years)
            """),
            {"states": states, "years": years},
        ).fetchall()

        found = []
        complete_keys: set = set()
        partial_keys: set = set()
        for row in rows:
            state, yr = row[0], row[1]
            vals = {attrs[i]: row[2 + i] for i in range(len(attrs))}
            present = {k: v for k, v in vals.items() if v is not None}
            if present:
                found.append({"spatial": state, "year": yr, **present})
                partial_keys.add((state, yr))
            if len(present) == len(attrs):
                complete_keys.add((state, yr))

        # a slot is "missing" only if at least one (state, year) has no data at all
        missing_states = [
            s for s in states
            if any((s, yr) not in partial_keys for yr in years)
        ]
        return {"found": found, "missing": list(dict.fromkeys(missing_states))}

    finally:
        db.close()
