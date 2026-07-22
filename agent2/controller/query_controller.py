"""
POST /query   — user submits a natural-language geospatial query
GET  /health  — liveness probe
"""
from __future__ import annotations

import logging
import time
import time as _time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kqml_messaging import MessageFactory

from ..pipeline import parse_query, validate_spatial
from ..retrieval import execute_local_lookup
from ..retrieval.geometry_resolver import resolve_geometries
from ..messaging import send_kqml_ask
from ..messaging.kqml_geometry_client import send_kqml_geometry_ask
from ..result import merge_results
from ..evaluation import log_evaluation_metrics

log = logging.getLogger("agent2.controller.query")
router = APIRouter()

SEPARATOR = "─" * 60


class UserQuery(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)


class QueryInfo(BaseModel):
    raw: str
    type: str
    spatial: List[str]
    temporal: List[int]
    attributes: List[str]


class DataGroups(BaseModel):
    complete: List[Dict[str, Any]]
    partial:  List[Dict[str, Any]]
    missing:  List[Dict[str, Any]]


class Summary(BaseModel):
    total_records:       int
    complete_records:    int
    partial_records:     int
    missing_records:     int
    total_data_points:   int
    present_data_points: int
    missing_data_points: int
    completeness_pct:    float


class Provenance(BaseModel):
    kqml_turns:           int
    records_from_agent_2: int
    records_from_agent_1: int
    records_from_both:    int
    records_unavailable:  int


class Tokens(BaseModel):
    agent_2: int
    agent_1: int
    total:   int


class Performance(BaseModel):
    phase1_ms: float
    phase2_ms: float
    phase3_ms: float
    total_ms:  float
    tokens:    Tokens


class QueryResponse(BaseModel):
    request_id:  str
    status:      str
    query:       QueryInfo
    data:        DataGroups
    summary:     Summary
    provenance:  Provenance
    performance: Performance


def _handle_geometry(params, raw_query: str, request_id: str,
                     timestamp: str, t0: float, tokens_agent2: int):
    """Handle GEOMETRY_LOOKUP queries — resolve locally then ask Agent-1 if missing."""
    entities = params.entities or []
    log.info("GEOM   │ Resolving geometry for %d entity/entities", len(entities))
    for e in entities:
        log.info("GEOM   │   %s (%s)", e["entity_name"], e["entity_type"])

    # Build one slot per requested entity
    slots = [
        MessageFactory.missing_geometry_slot(
            spatial_entity=e["entity_name"],
            entity_type=e["entity_type"],
        )
        for e in entities
    ]

    # Try local DB first (batch)
    found_local, still_missing = resolve_geometries(slots)
    t1 = time.perf_counter()

    # Ask Agent-1 for anything not found locally
    found_remote  = []
    kqml_turns    = 0
    tokens_agent1 = 0
    if still_missing:
        log.info("GEOM   │ %d entity/entities not found locally — asking Agent-1 ...", len(still_missing))
        try:
            resp         = send_kqml_geometry_ask(still_missing, request_id=request_id)
            found_remote = resp.get("found", [])
            kqml_turns   = 1
            log.info("GEOM   │ Agent-1 returned %d geometry result(s)", len(found_remote))
        except Exception as exc:
            log.warning("GEOM   │ Agent-1 unreachable — %s", exc)

    t2 = time.perf_counter()
    phase1_ms = (t1 - t0) * 1000
    phase2_ms = (t2 - t1) * 1000
    total_ms  = (t2 - t0) * 1000

    # Build lookup by canonical name for quick access
    local_names  = {fg.spatial_entity for fg in found_local}
    remote_names = {fg.spatial_entity for fg in found_remote}
    all_found_map = {fg.spatial_entity: fg for fg in found_local + found_remote}

    def _find_result(requested_name: str):
        if requested_name in all_found_map:
            return all_found_map[requested_name]
        # also try alias — geometry_resolver may have stored the DB canonical name
        for fg in found_local + found_remote:
            if fg.spatial_entity.lower() == requested_name.lower():
                return fg
        return None

    # Build ordered result list matching the request order
    geometries = []
    found_count = 0
    for e in entities:
        fg = _find_result(e["entity_name"])
        if fg:
            source = "Agent-2" if fg.spatial_entity in local_names else "Agent-1"
            log.info("GEOM   │ FOUND   %s (%s) from %s: %.60s…",
                     e["entity_name"], e["entity_type"], source, fg.geometry)
            geometries.append({
                "entity_name": fg.spatial_entity,
                "entity_type": fg.entity_type,
                "wkt":         fg.geometry,
                "srid":        fg.srid,
                "source":      source,
            })
            found_count += 1
        else:
            log.warning("GEOM   │ MISSING %s (%s) — not found in either agent",
                        e["entity_name"], e["entity_type"])
            geometries.append({
                "entity_name": e["entity_name"],
                "entity_type": e["entity_type"],
                "wkt":         None,
                "srid":        None,
                "source":      "unavailable",
            })

    total_req = len(entities)
    if found_count == total_req:
        status = "found"
    elif found_count == 0:
        status = "not_found"
    else:
        status = "partial"

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] geometry status=%s  found=%d/%d  %.0f ms",
             request_id, status, found_count, total_req, total_ms)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id":          request_id,
        "timestamp":           timestamp,
        "query":               raw_query,
        "phase1_ms":           round(phase1_ms, 1),
        "phase2_ms":           round(phase2_ms, 1),
        "phase3_ms":           0.0,
        "total_ms":            round(total_ms, 1),
        "tokens_agent2":       tokens_agent2,
        "tokens_agent1":       tokens_agent1,
        "tokens_total":        tokens_agent2 + tokens_agent1,
        "total_records":       total_req,
        "total_data_points":   total_req,
        "present_data_points": found_count,
        "missing_data_points": total_req - found_count,
        "complete_records":    found_count,
        "partial_records":     0,
        "empty_records":       total_req - found_count,
        "status":              status,
    })

    return {
        "request_id": request_id,
        "status":     status,
        "query": {
            "raw":      raw_query,
            "type":     "GEOMETRY_LOOKUP",
            "entities": entities,
        },
        "geometries": geometries,
        "performance": {
            "phase1_ms": round(phase1_ms, 1),
            "phase2_ms": round(phase2_ms, 1),
            "phase3_ms": 0.0,
            "total_ms":  round(total_ms, 1),
            "tokens": {
                "agent_2": tokens_agent2,
                "agent_1": tokens_agent1,
                "total":   tokens_agent2 + tokens_agent1,
            },
        },
    }


@router.post("/query")
def handle_query(body: UserQuery):  # no response_model — geometry branch returns a different shape
    t0        = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info(SEPARATOR)
    log.info("                       START")
    log.info("         New user query received by Agent 2")
    log.info(SEPARATOR)
    log.info("       │ Request ID : %s", request_id)
    log.info("       │ Query      : %r", body.query)

    # ── Step 1: Parse NL query ─────────────────────────────────────────────────
    log.info("STEP 1 │ Parsing natural-language query with GPT-4o mini ...")
    try:
        params, tokens_agent2 = parse_query(body.query)
    except Exception as exc:
        log.error("STEP 1 │ FAILED – %s", exc)
        raise HTTPException(status_code=400, detail=f"Parse error: {exc}") from exc

    log.info("STEP 1 │ Done")
    log.info("       │ Query type : %s", params.query_type)
    log.info("       │ Spatial    : %s", params.spatial)
    log.info("       │ Temporal   : %s", params.temporal)
    log.info("       │ Attributes : %s", params.attributes)
    if params.spatial_relationship:
        rel = params.spatial_relationship
        log.info("       │ Relationship: type=%s  refs=%s  dist_km=%s",
                 rel.type, rel.refs, rel.distance_km)

    # ── Geometry lookup shortcut (scenarios 11-13) ────────────────────────────
    if params.query_type == "GEOMETRY_LOOKUP":
        log.info("STEP 1 │ Routing to geometry handler ...")
        return _handle_geometry(params, body.query, request_id, timestamp, t0, tokens_agent2)

    # ── Step 2: Resolve spatial relationships ──────────────────────────────────
    if params.query_type != "DIRECT_LOOKUP":
        log.info("STEP 2 │ Resolving spatial relationship via PostGIS (%s) ...",
                 params.query_type)
        before = list(params.spatial)
        params = validate_spatial(params)
        log.info("STEP 2 │ Done")
        log.info("       │ Before  : %s", before)
        log.info("       │ Resolved: %s (%d states)", params.spatial, len(params.spatial))
    else:
        log.info("STEP 2 │ Skipped (DIRECT_LOOKUP – no spatial resolution needed)")

    # ── Step 3: Local database lookup ─────────────────────────────────────────
    log.info("STEP 3 │ Querying Agent-2 local database ...")
    log.info("       │ Looking for %d state(s) × %d year(s) × attrs=%s",
             len(params.spatial), len(params.temporal), params.attributes)

    local_result = execute_local_lookup(params)

    log.info("STEP 3 │ Done")
    log.info("       │ Found    : %d record(s)", len(local_result.found))
    log.info("       │ Gaps     : %d slot(s)", len(local_result.gaps))
    for i, gap in enumerate(local_result.gaps, 1):
        log.info("       │   Gap %d: spatial=%s  temporal=%s  attrs=%s",
                 i, gap.spatial, gap.temporal, gap.attributes)

    t1 = time.perf_counter()  # end of phase 1

    kqml_turns    = 0
    tokens_agent1 = 0
    agent1_data: List[Dict] = []

    # ── Step 4: KQML ask to Agent 1 (with retry) ──────────────────────────────
    if local_result.gaps:
        log.info("STEP 4 │ Gaps detected – sending KQML ask to Agent-1 ...  [req=%s]", request_id)
        log.info("       │ Missing slots to send: %d", len(local_result.gaps))
        for attempt in range(1, 4):
            try:
                resp          = send_kqml_ask(local_result.gaps, request_id=request_id)
                kqml_turns    = 1
                agent1_data   = resp.get("found", [])
                tokens_agent1 = resp.get("tokens_agent1", 0)
                log.info("STEP 4 │ KQML tell received from Agent-1 (attempt %d)", attempt)
                log.info("       │ Agent-1 found   : %d record(s)", len(agent1_data))
                log.info("       │ Agent-1 tokens  : %d", tokens_agent1)
                break
            except Exception as exc:
                log.warning("STEP 4 │ Agent-1 attempt %d failed – %s", attempt, exc)
                if attempt < 3:
                    _time.sleep(1.0 * attempt)
                else:
                    log.warning("STEP 4 │ Agent-1 unreachable after 3 attempts – gaps unresolved")
    else:
        log.info("STEP 4 │ Skipped – Agent 2 has complete data, Agent 1 not needed")

    t2 = time.perf_counter()  # end of phase 2

    # ── Step 5: Merge results ─────────────────────────────────────────────────
    log.info("STEP 5 │ Merging results ...")
    merged = merge_results(
        local_result.found,
        agent1_data,
        requested_states=params.spatial,
        requested_years=params.temporal,
        requested_attrs=params.attributes,
    )
    t3 = time.perf_counter()  # end of phase 3
    log.info("STEP 5 │ Done – %d total records", len(merged))

    # ── Split into complete / partial / missing ───────────────────────────────
    attrs = params.attributes
    complete_rows, partial_rows, missing_rows = [], [], []
    present_pts = 0
    for row in merged:
        n = sum(1 for a in attrs if row.get(a) is not None)
        present_pts += n
        if n == len(attrs):
            complete_rows.append(row)
        elif n == 0:
            missing_rows.append(row)
        else:
            partial_rows.append(row)

    total_pts    = len(merged) * len(attrs)
    missing_pts  = total_pts - present_pts
    completeness = round(present_pts / total_pts * 100, 1) if total_pts else 0.0

    # ── Provenance counts — single pass ──────────────────────────────────────
    from_a2 = from_a1 = from_both = unavail = 0
    for r in merged:
        src = r.get("source", "")
        if src == "Agent-2":   from_a2   += 1
        elif src == "Agent-1": from_a1   += 1
        elif "+" in src:       from_both += 1
        elif src == "missing": unavail   += 1

    log.info("       │ Groups  : complete=%d  partial=%d  missing=%d",
             len(complete_rows), len(partial_rows), len(missing_rows))
    log.info("       │ Sources : A2=%d  A1=%d  both=%d  unavail=%d",
             from_a2, from_a1, from_both, unavail)
    log.info("       │ Data completeness: %.1f%%  (%d / %d points)",
             completeness, present_pts, total_pts)

    # ── Final status ──────────────────────────────────────────────────────────
    if len(complete_rows) == len(merged):
        status = "complete"
    elif present_pts == 0:
        status = "not_found"
    else:
        status = "partial"

    phase1_ms = (t1 - t0) * 1000
    phase2_ms = (t2 - t1) * 1000
    phase3_ms = (t3 - t2) * 1000
    total_ms  = (t3 - t0) * 1000

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] status=%s  complete=%d  partial=%d  missing=%d  %.0f ms",
             request_id, status, len(complete_rows), len(partial_rows), len(missing_rows), total_ms)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id":          request_id,
        "timestamp":           timestamp,
        "query":               body.query,
        "phase1_ms":           phase1_ms,
        "phase2_ms":           phase2_ms,
        "phase3_ms":           phase3_ms,
        "total_ms":            total_ms,
        "tokens_agent2":       tokens_agent2,
        "tokens_agent1":       tokens_agent1,
        "tokens_total":        tokens_agent2 + tokens_agent1,
        "total_records":       len(merged),
        "total_data_points":   total_pts,
        "present_data_points": present_pts,
        "missing_data_points": missing_pts,
        "complete_records":    len(complete_rows),
        "partial_records":     len(partial_rows),
        "empty_records":       len(missing_rows),
        "status":              status,
    })

    return QueryResponse(
        request_id=request_id,
        status=status,
        query=QueryInfo(
            raw=body.query,
            type=params.query_type,
            spatial=params.spatial,
            temporal=params.temporal,
            attributes=params.attributes,
        ),
        data=DataGroups(
            complete=complete_rows,
            partial=partial_rows,
            missing=missing_rows,
        ),
        summary=Summary(
            total_records=len(merged),
            complete_records=len(complete_rows),
            partial_records=len(partial_rows),
            missing_records=len(missing_rows),
            total_data_points=total_pts,
            present_data_points=present_pts,
            missing_data_points=missing_pts,
            completeness_pct=completeness,
        ),
        provenance=Provenance(
            kqml_turns=kqml_turns,
            records_from_agent_2=from_a2,
            records_from_agent_1=from_a1,
            records_from_both=from_both,
            records_unavailable=unavail,
        ),
        performance=Performance(
            phase1_ms=round(phase1_ms, 1),
            phase2_ms=round(phase2_ms, 1),
            phase3_ms=round(phase3_ms, 1),
            total_ms=round(total_ms, 1),
            tokens=Tokens(
                agent_2=tokens_agent2,
                agent_1=tokens_agent1,
                total=tokens_agent2 + tokens_agent1,
            ),
        ),
    )


@router.get("/health")
def health():
    log.info("Health check OK")
    return {"status": "ok", "agent": "Agent-2"}
