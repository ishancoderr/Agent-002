"""
POST /query   — user submits a natural-language geospatial query
GET  /health  — liveness probe
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..pipeline import parse_query, validate_spatial
from ..retrieval import execute_local_lookup
from ..messaging import send_kqml_ask
from ..result import merge_results
from ..evaluation import log_evaluation_metrics

log = logging.getLogger("agent2.controller.query")
router = APIRouter()

SEPARATOR = "─" * 60


class UserQuery(BaseModel):
    query: str


class QueryResponse(BaseModel):
    status: str
    request_id: str
    query_params: Dict[str, Any]
    data: List[Dict[str, Any]]
    still_missing: List[str]
    kqml_turns: int
    total_records: int
    total_data_points: int
    present_data_points: int
    missing_data_points: int
    complete_records: int
    partial_records: int
    empty_records: int
    # evaluation metrics
    phase1_ms: float
    phase2_ms: float
    phase3_ms: float
    total_ms: float
    tokens_agent2: int
    tokens_agent1: int
    tokens_total: int


@router.post("/query", response_model=QueryResponse)
def handle_query(body: UserQuery):
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]

    log.info(SEPARATOR)
    log.info("STEP 0 │ New query received  [req=%s]", request_id)
    log.info("       │ Query : %r", body.query)
    log.info(SEPARATOR)

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

    kqml_turns   = 0
    tokens_agent1 = 0
    agent1_data: List[Dict] = []
    still_missing: List[str] = []

    # ── Step 4: KQML ask to Agent 1 (only if Agent 2 has gaps) ──────────────────
    phase2_ms = 0.0
    if not local_result.gaps:
        log.info("STEP 4 │ Skipped – Agent 2 has complete data, Agent 1 not needed")
    else:
        log.info("STEP 4 │ Gaps detected – sending KQML ask to Agent-1 ...  [req=%s]", request_id)
        log.info("       │ Missing slots to send: %d", len(local_result.gaps))
        _t_kqml_start = time.perf_counter()
        try:
            resp = send_kqml_ask(local_result.gaps, request_id=request_id)
            kqml_turns    = 1
            agent1_data   = resp.get("found", [])
            still_missing = resp.get("missing", [])
            tokens_agent1 = resp.get("tokens_agent1", 0)
            log.info("STEP 4 │ KQML tell received from Agent-1")
            log.info("       │ Agent-1 found   : %d record(s)", len(agent1_data))
            log.info("       │ Agent-1 tokens  : %d", tokens_agent1)
            log.info("       │ Still missing   : %s", still_missing if still_missing else "none")
        except Exception as exc:
            log.warning("STEP 4 │ Agent-1 unreachable – %s", exc)
            log.warning("       │ All gaps remain unresolved")
            still_missing = [s for gap in local_result.gaps for s in gap.spatial]
        phase2_ms = (time.perf_counter() - _t_kqml_start) * 1000

    # ── Step 5: Merge results ─────────────────────────────────────────────────
    log.info("STEP 5 │ Merging results ...")
    _t_merge_start = time.perf_counter()
    merged = merge_results(
        local_result.found,
        agent1_data,
        requested_states=params.spatial,
        requested_years=params.temporal,
        requested_attrs=params.attributes,
    )
    t3 = time.perf_counter()
    t2 = _t_merge_start  # phase3 = t3 - t2

    log.info("STEP 5 │ Done")
    log.info("       │ Agent-2 records : %d", len(local_result.found))
    log.info("       │ Agent-1 records : %d", len(agent1_data))
    log.info("       │ Total merged    : %d", len(merged))

    # ── Data quality stats ────────────────────────────────────────────────────
    attrs = params.attributes
    complete = partial = empty = present_pts = 0
    for row in merged:
        present = [a for a in attrs if row.get(a) is not None]
        if len(present) == len(attrs):
            complete += 1
        elif len(present) == 0:
            empty += 1
        else:
            partial += 1
        present_pts += len(present)

    total_pts   = len(merged) * len(attrs)
    missing_pts = total_pts - present_pts

    log.info("       │ Data quality :")
    log.info("       │   Total data points    : %d  (%d records × %d attrs)",
             total_pts, len(merged), len(attrs))
    log.info("       │   Present              : %d", present_pts)
    log.info("       │   Missing (null)       : %d", missing_pts)
    log.info("       │   Complete records     : %d  (all attrs present)", complete)
    log.info("       │   Partial records      : %d  (some attrs present)", partial)
    log.info("       │   Empty records        : %d  (all attrs null)", empty)

    # ── Final status ──────────────────────────────────────────────────────────
    if complete == len(merged):
        status = "complete"
    elif present_pts == 0:
        status = "not-found"
    else:
        status = "partial-complete"

    phase1_ms = (t1 - t0) * 1000
    phase3_ms = (t3 - t2) * 1000
    total_ms  = (t3 - t0) * 1000
    tokens_total = tokens_agent2 + tokens_agent1

    log.info(SEPARATOR)
    log.info("DONE   │ [req=%s] Status: %s  │  Records: %d  │  Present: %d/%d  │  %.0f ms",
             request_id, status, len(merged), present_pts, total_pts, total_ms)
    log.info("       │ Timing  phase1=%.0f ms  phase2=%.0f ms  phase3=%.0f ms  total=%.0f ms",
             phase1_ms, phase2_ms, phase3_ms, total_ms)
    log.info("       │ Tokens  agent2=%d  agent1=%d  total=%d",
             tokens_agent2, tokens_agent1, tokens_total)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id":   request_id,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query":        body.query,
        "phase1_ms":    phase1_ms,
        "phase2_ms":    phase2_ms,
        "phase3_ms":    phase3_ms,
        "total_ms":     total_ms,
        "tokens_agent2": tokens_agent2,
        "tokens_agent1": tokens_agent1,
        "tokens_total":  tokens_total,
    })

    return QueryResponse(
        status=status,
        request_id=request_id,
        query_params={
            "type":       params.query_type,
            "spatial":    params.spatial,
            "temporal":   params.temporal,
            "attributes": params.attributes,
        },
        data=merged,
        still_missing=still_missing,
        kqml_turns=kqml_turns,
        total_records=len(merged),
        total_data_points=total_pts,
        present_data_points=present_pts,
        missing_data_points=missing_pts,
        complete_records=complete,
        partial_records=partial,
        empty_records=empty,
        phase1_ms=round(phase1_ms, 1),
        phase2_ms=round(phase2_ms, 1),
        phase3_ms=round(phase3_ms, 1),
        total_ms=round(total_ms, 1),
        tokens_agent2=tokens_agent2,
        tokens_agent1=tokens_agent1,
        tokens_total=tokens_total,
    )


@router.get("/health")
def health():
    log.info("Health check OK")
    return {"status": "ok", "agent": "Agent-2"}
