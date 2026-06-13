from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..retrieval.gap_detector import DataRecord

log = logging.getLogger("agent2.result.merger")


def merge_results(
    local: List[DataRecord],
    agent1_data: List[Dict[str, Any]],
    requested_states: Optional[List[str]] = None,
    requested_years: Optional[List[int]] = None,
    requested_attrs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    merged: Dict[tuple, Dict[str, Any]] = {}

    for r in local:
        key = (r.state, r.year)
        entry = {"state": r.state, "year": r.year, "source": "Agent-2", **r.values}
        merged[key] = entry
        log.info("       │ [Agent-2] %s %d → %s", r.state, r.year, r.values)

    for rec in agent1_data:
        state = rec.pop("spatial", rec.get("state", ""))
        year  = rec.get("year", "")
        key   = (state, year)
        attrs = {k: v for k, v in rec.items() if k not in ("state", "year")}

        if key in merged:
            merged[key].update(attrs)
            merged[key]["source"] = "Agent-2+Agent-1"
            log.info("       │ [Agent-1 fill] %s %s → added %s", state, year, attrs)
        else:
            entry = {"state": state, "year": year, "source": "Agent-1", **attrs}
            merged[key] = entry
            log.info("       │ [Agent-1 new ] %s %s → %s", state, year, attrs)

    if requested_states and requested_years and requested_attrs:
        from ..retrieval.gap_detector import ALL_STATES
        states = ALL_STATES if "all" in requested_states else requested_states
        for state in states:
            for year in requested_years:
                key = (state, year)
                if key not in merged:
                    entry: Dict[str, Any] = {
                        "state":  state,
                        "year":   year,
                        "source": "missing",
                    }
                    for attr in requested_attrs:
                        entry[attr] = None
                    merged[key] = entry
                    log.info("       │ [MISSING     ] %s %d → all null", state, year)

    result = sorted(merged.values(), key=lambda x: (x.get("state", ""), x.get("year", 0)))
    log.info("       │ Sorted merged list: %d records", len(result))
    return result
