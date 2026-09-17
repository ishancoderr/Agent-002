"""
POST /query   — user submits a natural-language geospatial query
GET  /health  — liveness probe
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from ..pipeline import run as run_pipeline
from ..pipeline.gazetteer import normalize_entity_name
from ..pipeline.query_params import DEFAULT_ENTITY_TYPE, system_capabilities_description
from ..retrieval import answer_query
from ..retrieval.geometry_resolver import (
    resolve_entities, build_city_buffer, buffer_from_point, cities_within_buffer,
    resolve_named_geometry, execute_operation, fold_union, targets_within_buffer,
)
from ..messaging.kqml_geometry_client import send_kqml_geometry_ask, send_kqml_city_buffer_ask
from ..evaluation import log_evaluation_metrics
from kqml_messaging import MessageFactory, MissingGeometrySlot, response_status

log = logging.getLogger("agent2.controller.query")
router = APIRouter()

SEPARATOR = "_" * 60


class UserQuery(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    # Per-request override of the OpenAI model used for classification and
    # extraction. Omit to use CLASSIFY_MODEL/EXTRACT_MODEL from the server's
    # .env (see agent2/.env.example) — this is for a caller who wants a
    # specific model for one query, not for setting the deployment default.
    model: Optional[str] = Field(None, min_length=1, max_length=100)


class QueryInfo(BaseModel):
    raw: str
    type: str
    spatial: List[str]
    temporal: List[int]
    attributes: List[str]
    # Which entities.yaml entity type the rows in `data` are about — a
    # DIRECT_LOOKUP row's own field is still literally named "state" for
    # every entity type (merger.py/DataRecord predate multi-entity support,
    # and renaming that field is a separate, larger wire-contract change),
    # so a client needs this to know to label the column "river" instead of
    # "state" for a river query rather than showing a state-only heading
    # for every entity type.
    entity_type: str


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
    kqml_turns:             int
    records_from_agent_2:   int
    records_from_agent_1:   int
    records_from_both:      int
    records_unavailable:    int


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


@router.post("/query")
def handle_query(body: UserQuery):
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info(SEPARATOR)
    log.info("                       START")
    log.info("         New user query received by Agent 1")
    log.info(SEPARATOR)
    log.info("       │ Request ID : %s", request_id)
    log.info("       │ Query      : %r", body.query)

    # ── Steps 1-2: run the pipeline (parse, classify, extract, and — for
    # DIRECT_LOOKUP/relationship queries — resolve spatially via PostGIS) ───────
    log.info("STEP 1 │ Parsing natural-language query with GPT-4o mini ...")
    try:
        params = run_pipeline(body.query, model=body.model)
    except Exception as exc:
        log.error("STEP 1 │ FAILED – %s", exc)
        raise HTTPException(status_code=400, detail=f"Parse error: {exc}") from exc
    tokens_agent2 = params.classify_tokens + params.extract_tokens

    log.info("STEP 1 │ Done")
    log.info("       │ Query type : %s", params.query_type)

    # ── Unrelated — reject before any DB lookup or KQML exchange ───────────────
    if params.query_type == "UNRELATED":
        return _handle_unrelated(params, body.query, request_id, timestamp, t0, tokens_agent2)

    # ── Data was asked for but no year was named or implied ────────────────────
    if params.query_type == "NEEDS_YEAR":
        return _handle_needs_year(params, body.query, request_id, timestamp, t0, tokens_agent2)

    # ── Geometry lookup — bypass demographics pipeline entirely ───────────────
    if params.query_type == "GEOMETRY_LOOKUP":
        return _handle_geometry(params, body.query, request_id, timestamp, t0, tokens_agent2)

    # ── Spatial operation (Scenarios 13-16, 20) — named entities, no data lookup
    if params.query_type == "SPATIAL_OPERATION":
        return _handle_spatial_operation(params, body.query, request_id, timestamp, t0, tokens_agent2)

    # ── Relationship buffer over an open target type (Scenario 21) ────────────
    if params.query_type == "SPATIAL_RELATIONSHIP_BUFFER":
        return _handle_relationship_buffer(params, body.query, request_id, timestamp, t0, tokens_agent2)

    log.info("       │ Spatial    : %s", params.spatial)
    log.info("       │ Temporal   : %s", params.temporal)
    log.info("       │ Attributes : %s", params.attributes)
    if params.spatial_relationship:
        rel = params.spatial_relationship
        log.info("       │ Relationship: type=%s  refs=%s  dist_km=%s",
                 rel.type, rel.refs, rel.distance_km)

    # ── Step 2: Spatial relationship resolution already ran inside
    # run_pipeline() (PostGIS, for DIRECT_LOOKUP/relationship queries only) ────
    if params.query_type != "DIRECT_LOOKUP":
        log.info("STEP 2 │ Resolved via PostGIS (%s): %s (%d states)",
                 params.query_type, params.spatial, len(params.spatial))
    else:
        log.info("STEP 2 │ Skipped (DIRECT_LOOKUP – no spatial resolution needed)")

    # ── Pure spatial question — the query asked only WHICH states, not for data
    if params.query_type != "DIRECT_LOOKUP" and not params.attributes:
        t1 = time.perf_counter()
        states = [] if params.spatial == ["all"] else list(params.spatial)
        status = response_status(has_found=bool(states), has_missing=not states)
        total_ms = (t1 - t0) * 1000

        log.info(SEPARATOR)
        log.info("DONE   │ [%s] %s (spatial-only) status=%s  states=%d  %.0f ms",
                 request_id, params.query_type, status, len(states), total_ms)
        log.info(SEPARATOR)

        log_evaluation_metrics({
            "request_id":        request_id,
            "timestamp":         timestamp,
            "query":             body.query,
            "query_type":        params.query_type,
            "classify_tokens":   params.classify_tokens,
            "extract_tokens":    params.extract_tokens,
            "classify_model":     params.classify_model,
            "extract_model":      params.extract_model,
            "extracted_data":    params.extracted_data,
            "local_resolution": {
                "resolution_method": f"PostGIS spatial relationship ({params.query_type})",
                "resolved_states":   states,
                "note":              "verdict relationships (adjacency/direction/distance) are "
                                      "computed locally via PostGIS once geometries are present — "
                                      "no demographic data was requested, so no KQML ask was needed",
            },
            "kqml_exchanges":    [],
            "phase1_ms":         total_ms,
            "phase2_ms":         0.0,
            "phase3_ms":         0.0,
            "total_ms":          total_ms,
            "tokens_agent2":     tokens_agent2,
            "tokens_agent1":     0,
            "tokens_total":      tokens_agent2,
            "total_records":     len(states),
            "total_data_points": len(states),
            "present_data_points": len(states),
            "missing_data_points": 0,
            "complete_records":  len(states),
            "partial_records":   0,
            "empty_records":     0,
            "status":            status,
        })

        rel = params.spatial_relationship
        return {
            "request_id": request_id,
            "status":     status,
            "query": {
                "raw":  body.query,
                "type": params.query_type,
                "relationship": {
                    "type":        rel.type if rel else None,
                    "subject":     rel.subject if rel else None,
                    "refs":        rel.refs if rel else [],
                    "distance_km": rel.distance_km if rel else None,
                },
            },
            # A question that named a subject asked a yes/no. `verdict` carries
            # it; `states` still carries the set the verdict was read from, so
            # the answer can be checked. verdict is null both when the question
            # asked for the list rather than a verdict, and when the subject or
            # reference has no geometry anywhere - see `unknown_states` for the
            # latter: a null verdict with the subject listed there means the
            # relationship genuinely could not be tested, not "no".
            "verdict": params.verdict,
            "states": states,
            "unknown_states": params.unknown_states,
            "summary": {"total": len(states), "verdict": params.verdict,
                       "unknown": len(params.unknown_states)},
            "performance": {
                "phase1_ms": round(total_ms, 1),
                "phase2_ms": 0.0,
                "phase3_ms": 0.0,
                "total_ms":  round(total_ms, 1),
                "tokens": {"agent_2": tokens_agent2, "agent_1": 0, "total": tokens_agent2},
            },
        }

    # ── Steps 3-5: local lookup, KQML ask for gaps, merge — one call now,
    # in retrieval/local_store.py's answer_query() ─────────────────────────────
    log.info("STEP 3-5 │ Resolving locally, asking Agent-1 for gaps, merging ...")
    log.info("       │ Looking for %d state(s) × %d year(s) × attrs=%s",
             len(params.spatial), len(params.temporal), params.attributes)

    answered = answer_query(params)
    local_result   = answered.local_result
    merged         = answered.merged
    kqml_turns     = answered.kqml_turns
    tokens_agent1  = answered.tokens_agent1
    kqml_exchanges = answered.kqml_exchanges
    phase1_ms      = answered.phase1_ms
    phase2_ms      = answered.phase2_ms
    phase3_ms      = answered.phase3_ms

    log.info("STEP 3-5 │ Done")
    log.info("       │ Found locally : %d record(s)", len(local_result.found))
    log.info("       │ Gaps          : %d slot(s)", len(local_result.gaps))
    for i, gap in enumerate(local_result.gaps, 1):
        log.info("       │   Gap %d: spatial=%s  temporal=%s  attrs=%s",
                 i, gap.spatial, gap.temporal, gap.attributes)
    log.info("       │ KQML turns    : %d", kqml_turns)
    log.info("       │ Merged total  : %d record(s)", len(merged))

    # ── Split into three groups ───────────────────────────────────────────────
    attrs = params.attributes
    complete_rows: List[Dict] = []
    partial_rows:  List[Dict] = []
    missing_rows:  List[Dict] = []
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

    total_pts   = len(merged) * len(attrs)
    missing_pts = total_pts - present_pts
    completeness = round(present_pts / total_pts * 100, 1) if total_pts else 0.0

    # ── Provenance counts — single pass ──────────────────────────────────────
    from_a1 = from_a2 = from_both = unavail = 0
    for r in merged:
        src = r.get("source", "")
        if src == "Agent-2":    from_a1   += 1
        elif src == "Agent-1":  from_a2   += 1
        elif "+" in src:        from_both += 1
        elif src == "missing":  unavail   += 1

    log.info("       │ Groups  : complete=%d  partial=%d  missing=%d",
             len(complete_rows), len(partial_rows), len(missing_rows))
    log.info("       │ Sources : A1=%d  A2=%d  both=%d  unavail=%d",
             from_a1, from_a2, from_both, unavail)
    log.info("       │ Data completeness: %.1f%%  (%d / %d points)",
             completeness, present_pts, total_pts)

    # ── Final status ──────────────────────────────────────────────────────────
    status = response_status(has_found=present_pts > 0,
                             has_missing=len(complete_rows) != len(merged))

    total_ms = (time.perf_counter() - t0) * 1000

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] status=%s  complete=%d  partial=%d  missing=%d  %.0f ms",
             request_id, status, len(complete_rows), len(partial_rows), len(missing_rows), total_ms)
    log.info(SEPARATOR)

    # ── Evaluation metrics ────────────────────────────────────────────────────
    log_evaluation_metrics({
        "request_id":          request_id,
        "timestamp":           timestamp,
        "query":               body.query,
        "query_type":          params.query_type,
        "classify_tokens":     params.classify_tokens,
        "extract_tokens":      params.extract_tokens,
        "classify_model":     params.classify_model,
        "extract_model":      params.extract_model,
        "extracted_data":      params.extracted_data,
        "local_resolution": {
            "states_queried":     params.spatial,
            "found_locally":      len(local_result.found),
            "gaps_found":         len(local_result.gaps),
            "gap_detail":         [
                {"spatial": g.spatial, "temporal": g.temporal, "attributes": g.attributes}
                for g in local_result.gaps
            ],
            "sql_queries":        local_result.queries,
        },
        "parsed_spatial":      params.spatial,
        "parsed_temporal":     params.temporal,
        "parsed_attributes":   params.attributes,
        "kqml_exchanges":      kqml_exchanges,
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
            entity_type=params.entity_type or DEFAULT_ENTITY_TYPE,
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
            records_from_agent_2=from_a1,
            records_from_agent_1=from_a2,
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


def _handle_unrelated(params, raw_query: str, request_id: str, timestamp: str, t0: float, tokens_agent2: int):
    """The query names no German state/city and asks for none of this system's
    data — nothing to look up locally, nothing to send Agent-1. Rejected outright
    rather than coerced into a meaningless DIRECT_LOOKUP."""
    total_ms = (time.perf_counter() - t0) * 1000

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] UNRELATED — rejected  %.0f ms", request_id, total_ms)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id":        request_id,
        "timestamp":         timestamp,
        "query":             raw_query,
        "query_type":        "UNRELATED",
        "classify_tokens":   params.classify_tokens,
        "extract_tokens":    0,
        "classify_model":     params.classify_model,
        "extract_model":      params.extract_model,
        "extracted_data":    {},
        "local_resolution": {
            "note": "rejected before any local database lookup was attempted",
        },
        "kqml_exchanges":    [],
        "phase1_ms":         total_ms,
        "phase2_ms":         0.0,
        "phase3_ms":         0.0,
        "total_ms":          total_ms,
        "tokens_agent2":     tokens_agent2,
        "tokens_agent1":     0,
        "tokens_total":      tokens_agent2,
        "total_records":     0,
        "total_data_points": 0,
        "present_data_points": 0,
        "missing_data_points": 0,
        "complete_records":  0,
        "partial_records":   0,
        "empty_records":     0,
        "status":            "rejected",
    })

    return {
        "request_id": request_id,
        "status":     "rejected",
        "message": system_capabilities_description(),
        "query": {"raw": raw_query, "type": "UNRELATED"},
        "performance": {
            "phase1_ms": round(total_ms, 1),
            "phase2_ms": 0.0,
            "phase3_ms": 0.0,
            "total_ms":  round(total_ms, 1),
            "tokens": {"agent_2": tokens_agent2, "agent_1": 0, "total": tokens_agent2},
        },
    }


def _handle_needs_year(params, raw_query: str, request_id: str, timestamp: str, t0: float, tokens_agent2: int):
    """The query asked for data (population/marriages/live_births) but named
    or implied no year. Guessing one used to mean a question about 1985
    silently got answered with 2021's figures instead - a wrong answer with
    nothing disclosing the swap. Rejecting and saying exactly what is missing
    is the honest response, the same way an UNRELATED question is rejected
    rather than forced into a category it doesn't fit."""
    total_ms = (time.perf_counter() - t0) * 1000

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] NEEDS_YEAR — rejected  %.0f ms", request_id, total_ms)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id":        request_id,
        "timestamp":         timestamp,
        "query":             raw_query,
        "query_type":        "NEEDS_YEAR",
        "classify_tokens":   params.classify_tokens,
        "extract_tokens":    params.extract_tokens,
        "classify_model":     params.classify_model,
        "extract_model":      params.extract_model,
        "extracted_data":    params.extracted_data,
        "local_resolution": {
            "note": "rejected before any local database lookup was attempted "
                    "- no year was named or implied",
        },
        "kqml_exchanges":    [],
        "phase1_ms":         total_ms,
        "phase2_ms":         0.0,
        "phase3_ms":         0.0,
        "total_ms":          total_ms,
        "tokens_agent2":     tokens_agent2,
        "tokens_agent1":     0,
        "tokens_total":      tokens_agent2,
        "total_records":     0,
        "total_data_points": 0,
        "present_data_points": 0,
        "missing_data_points": 0,
        "complete_records":  0,
        "partial_records":   0,
        "empty_records":     0,
        "status":            "rejected",
    })

    return {
        "request_id": request_id,
        "status":     "rejected",
        "message": (
            f"This question asks for {', '.join(params.attributes)} but doesn't say "
            "which year. Add one to answer it, for example:\n"
            f"  - a specific year: \"...in 2021\"\n"
            f"  - a range: \"...from 2019 to 2023\"\n"
            f"  - a relative reference: \"...this year\" or \"...now\""
        ),
        "query": {"raw": raw_query, "type": "NEEDS_YEAR", "attributes": params.attributes},
        "performance": {
            "phase1_ms": round(total_ms, 1),
            "phase2_ms": 0.0,
            "phase3_ms": 0.0,
            "total_ms":  round(total_ms, 1),
            "tokens": {"agent_2": tokens_agent2, "agent_1": 0, "total": tokens_agent2},
        },
    }


def _handle_geometry(params, raw_query: str, request_id: str,
                     timestamp: str, t0: float, tokens_agent2: int):
    """Handle GEOMETRY_LOOKUP queries — resolve locally then ask Agent-1 if missing."""
    entities = params.entities or []
    if not entities:
        return {"request_id": request_id, "status": "error",
                "message": "No entities found in geometry query.",
                "performance": {"phase1_ms": round((time.perf_counter()-t0)*1000,1),
                                "phase2_ms":0,"phase3_ms":0,"total_ms":0,
                                "tokens":{"agent_2":tokens_agent2,"agent_1":0,"total":tokens_agent2}}}
    log.info("GEOM   │ Resolving %d entity/entities: %s", len(entities), entities)

    # Check the local DB first — using the plain {"entity_name",
    # "entity_type"} dicts extraction already produced, not a
    # MissingGeometrySlot built for every entity up front. Building one
    # Pydantic-validates entity_type against kqml_messaging.EntityType
    # (city/state only), which would fail immediately for any other of
    # entities.yaml's own types (e.g. river) even when this agent already
    # holds it locally and Agent-1 was never going to be asked at all.
    sql_queries: List[str] = []
    found_local, still_missing = resolve_entities(entities, queries=sql_queries)
    local_names = {fg["entity_name"] for fg in found_local}

    t1 = time.perf_counter()

    # Only entities genuinely absent locally are ever turned into a real
    # MissingGeometrySlot — the one place that validation actually needs to
    # happen, since it's the one place an entity_type has to travel over
    # the wire. An entity_type the KQML protocol doesn't carry yet (again,
    # river) can't be asked for; it's logged and left "not_found" rather
    # than taking the rest of the request down with it.
    remote_askable: List[MissingGeometrySlot] = []
    for m in still_missing:
        try:
            remote_askable.append(MessageFactory.missing_geometry_slot(
                spatial_entity=m["entity_name"], entity_type=m["entity_type"],
            ))
        except (ValueError, ValidationError) as exc:
            log.warning(
                "GEOM   │ %r has entity_type=%r, which the KQML protocol doesn't "
                "carry yet — cannot ask Agent-1 for it: %s",
                m["entity_name"], m["entity_type"], exc,
            )

    # Ask Agent-1 for anything not found locally
    found_remote = []
    kqml_turns   = 0
    tokens_agent1 = 0
    kqml_exchanges: List[Dict] = []
    if remote_askable:
        log.info("GEOM   │ %d missing locally — asking Agent-1 ...", len(remote_askable))
        try:
            resp          = send_kqml_geometry_ask(remote_askable)
            found_remote  = resp.get("found", [])
            kqml_turns    = 1
            if "ask_message" in resp and "tell_message" in resp:
                kqml_exchanges.append({"ask": resp["ask_message"], "tell": resp["tell_message"]})
            log.info("GEOM   │ Agent-1 returned %d geometry result(s)", len(found_remote))
        except Exception as exc:
            log.warning("GEOM   │ Agent-1 unreachable — %s", exc)

    t2 = time.perf_counter()

    # Index found geometries by name (case-insensitive) — both as the same
    # plain-dict shape, found_local already is one and found_remote (real
    # FoundGeometrySlot objects the shared library validated on the way
    # back from Agent-1) is normalized to match, so _find_result()/the
    # result-list build below never need to care which side an answer
    # came from.
    local_index  = {fg["entity_name"].lower(): fg for fg in found_local}
    remote_index = {
        fg.spatial_entity.lower(): {
            "entity_name": fg.spatial_entity, "entity_type": fg.entity_type,
            "wkt": fg.geometry, "srid": fg.srid,
        }
        for fg in found_remote
    }

    def _find_result(requested_name: str):
        """Return (result dict, source) or (None, 'not_found')."""
        # The gazetteer supplies the database's own spelling; the name as asked
        # for is kept as a fallback in case the reply used that form.
        variants = list(dict.fromkeys(filter(None, [
            normalize_entity_name(requested_name, "city"),
            requested_name,
        ])))
        for v in variants:
            key = v.lower()
            if key in local_index:
                return local_index[key], "Agent-2"
            if key in remote_index:
                return remote_index[key], "Agent-1"
        return None, "not_found"

    # Build result list preserving request order
    geometries = []
    for e in entities:
        fg, source = _find_result(e["entity_name"])
        if fg is not None:
            geometries.append({
                "entity_name": fg["entity_name"],
                "entity_type": fg["entity_type"],
                "wkt":         fg["wkt"],
                "srid":        fg["srid"],
                "source":      source,
            })
        else:
            geometries.append({
                "entity_name": e["entity_name"],
                "entity_type": e["entity_type"],
                "wkt":         None,
                "srid":        None,
                "source":      "not_found",
            })

    t3 = time.perf_counter()  # end of phase 3 (merge / index / result-list build)

    phase1_ms = (t1 - t0) * 1000
    phase2_ms = (t2 - t1) * 1000
    phase3_ms = (t3 - t2) * 1000
    total_ms  = (t3 - t0) * 1000

    found_count   = sum(1 for g in geometries if g["wkt"] is not None)
    missing_count = len(geometries) - found_count

    status = response_status(has_found=found_count > 0, has_missing=missing_count > 0)

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] geometry status=%s  found=%d  missing=%d  %.0f ms",
             request_id, status, found_count, missing_count, total_ms)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id":        request_id,
        "timestamp":         timestamp,
        "query":             raw_query,
        "query_type":        "GEOMETRY_LOOKUP",
        "classify_tokens":   params.classify_tokens,
        "extract_tokens":    params.extract_tokens,
        "classify_model":     params.classify_model,
        "extract_model":      params.extract_model,
        "extracted_data":    params.extracted_data,
        "local_resolution": {
            "entities_requested": [e["entity_name"] for e in entities],
            "found_locally":      sorted(local_names),
            "still_missing_after_local_db": [m["entity_name"] for m in still_missing],
            "sql_queries":        sql_queries,
        },
        "kqml_exchanges":    kqml_exchanges,
        "phase1_ms":         phase1_ms,
        "phase2_ms":         phase2_ms,
        "phase3_ms":         phase3_ms,
        "total_ms":          total_ms,
        "tokens_agent2":     tokens_agent2,
        "tokens_agent1":     tokens_agent1,
        "tokens_total":      tokens_agent2 + tokens_agent1,
        "total_records":     len(geometries),
        "total_data_points": len(geometries),
        "present_data_points": found_count,
        "missing_data_points": missing_count,
        "complete_records":  found_count,
        "partial_records":   0,
        "empty_records":     missing_count,
        "status":            status,
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
        "summary": {
            "total":    len(geometries),
            "found":    found_count,
            "missing":  missing_count,
        },
        "performance": {
            "phase1_ms": round(phase1_ms, 1),
            "phase2_ms": round(phase2_ms, 1),
            "phase3_ms": round(phase3_ms, 1),
            "total_ms":  round(total_ms, 1),
            "tokens": {
                "agent_2": tokens_agent2,
                "agent_1": tokens_agent1,
                "total":   tokens_agent2 + tokens_agent1,
            },
        },
    }


def _wkt_geometry_type(wkt: str) -> str:
    """The leading word of a WKT string names its geometry type. A WKT string
    always carries this - it is not extra work to read it, only to bother
    checking - and it is the only way to tell a real overlap (POLYGON /
    MULTIPOLYGON) from two shapes that merely touch (LINESTRING) or a pair
    with no intersection at all (an empty collection). Two administrative
    states sharing only a border, per Scenario 14, produce exactly this: a
    geometrically correct result that is not an area, and looks identical to
    a real shared region unless this is read out."""
    if not wkt:
        return "EMPTY"
    return wkt.strip().split("(", 1)[0].strip().split()[0].upper()


_AREA_GEOMETRY_TYPES = {"POLYGON", "MULTIPOLYGON"}


def _handle_spatial_operation(params, raw_query: str, request_id: str,
                              timestamp: str, t0: float, tokens_agent2: int):
    """Scenarios 13-16 (Union/Intersection/Difference/SymDifference) and 20
    (Buffer+Within over NAMED targets). Every entity involved is named, so this
    resolves each one's geometry exactly like GEOMETRY_LOOKUP (locally, or via a
    normal :missing-geometries ask to the peer), checks SRID agreement, and only
    then runs the operation locally — the operation itself is never missing
    (Section 3.5), only an input can be."""
    operation   = params.operation
    entity_type = params.entity_type or DEFAULT_ENTITY_TYPE
    names       = params.spatial or []

    if not operation or len(names) < 2:
        return {"request_id": request_id, "status": "error",
                "message": "SPATIAL_OPERATION requires an operation and at least two named entities.",
                "performance": {"phase1_ms": round((time.perf_counter()-t0)*1000, 1),
                                "phase2_ms": 0, "phase3_ms": 0, "total_ms": 0,
                                "tokens": {"agent_2": tokens_agent2, "agent_1": 0, "total": tokens_agent2}}}

    log.info("SPOP   │ Operation: %s  entities=%s  entity_type=%s", operation, names, entity_type)

    # ── Resolve every named entity's geometry, locally first ──────────────────
    sql_queries: List[str] = []
    resolved: Dict[str, Dict] = {}
    still_missing: List[str] = []
    for name in names:
        g = resolve_named_geometry(name, entity_type, queries=sql_queries)
        if g is not None:
            resolved[name] = g
        else:
            still_missing.append(name)

    t1 = time.perf_counter()

    # ── Ask the peer for whatever wasn't held locally, exactly like GEOMETRY_LOOKUP ──
    kqml_turns    = 0
    tokens_agent1 = 0
    kqml_exchanges: List[Dict] = []
    sources: Dict[str, str] = {n: "Agent-2" for n in resolved}
    if still_missing:
        log.info("SPOP   │ %d entity/entities missing locally — asking Agent-1 ...", len(still_missing))
        # A BufferWithin reference is often a city while its targets are states
        # (Scenario 20), so guess per-name from the state gazetteer rather than
        # trusting one entity_type for the whole operation.
        from ..pipeline.query_parser import GERMAN_STATES
        slots = [
            MessageFactory.missing_geometry_slot(
                spatial_entity=n, entity_type=("state" if n in GERMAN_STATES else "city"),
            )
            for n in still_missing
        ]
        try:
            resp = send_kqml_geometry_ask(slots)
            kqml_turns = 1
            if "ask_message" in resp and "tell_message" in resp:
                kqml_exchanges.append({"ask": resp["ask_message"], "tell": resp["tell_message"]})
            for fg in resp.get("found", []):
                resolved[fg.spatial_entity] = {"name": fg.spatial_entity, "wkt": fg.geometry, "srid": fg.srid}
                sources[fg.spatial_entity] = "Agent-1"
            log.info("SPOP   │ Agent-1 returned %d geometry/geometries", len(resp.get("found", [])))
        except Exception as exc:
            log.warning("SPOP   │ Agent-1 unreachable — %s", exc)

    t2 = time.perf_counter()

    local_resolution = {
        "operation":           operation,
        "entities_requested":  names,
        "resolved_locally":    [n for n, s in sources.items() if s == "Agent-2"],
        "still_missing_after_local_db": still_missing,
        "sql_queries":         sql_queries,
    }

    unresolved = [n for n in names if n not in resolved]
    if unresolved:
        log.warning("SPOP   │ Still unresolved: %s — cannot run operation", unresolved)
        t3 = time.perf_counter()
        total_ms = (t3 - t0) * 1000
        log_evaluation_metrics({
            "request_id": request_id, "timestamp": timestamp, "query": raw_query,
            "query_type": "SPATIAL_OPERATION",
            "classify_tokens": params.classify_tokens, "extract_tokens": params.extract_tokens, "classify_model": params.classify_model, "extract_model": params.extract_model,
            "extracted_data": params.extracted_data, "local_resolution": local_resolution,
            "kqml_exchanges": kqml_exchanges,
            "phase1_ms": (t1 - t0) * 1000, "phase2_ms": (t2 - t1) * 1000, "phase3_ms": 0.0,
            "total_ms": total_ms,
            "tokens_agent2": tokens_agent2, "tokens_agent1": tokens_agent1,
            "tokens_total": tokens_agent2 + tokens_agent1,
            "total_records": len(names), "total_data_points": len(names),
            "present_data_points": len(names) - len(unresolved), "missing_data_points": len(unresolved),
            "complete_records": 0, "partial_records": 0, "empty_records": len(unresolved),
            "status": response_status(has_found=False, has_missing=True),
        })
        return {
            "request_id": request_id, "status": response_status(has_found=False, has_missing=True),
            "query": {"raw": raw_query, "type": "SPATIAL_OPERATION", "operation": operation, "spatial": names},
            "still_missing": unresolved,
            "performance": {
                "phase1_ms": round((t1 - t0) * 1000, 1), "phase2_ms": round((t2 - t1) * 1000, 1),
                "phase3_ms": 0.0, "total_ms": round(total_ms, 1),
                "tokens": {"agent_2": tokens_agent2, "agent_1": tokens_agent1,
                           "total": tokens_agent2 + tokens_agent1},
            },
        }

    # ── Run the operation locally, now that every input is present ────────────
    try:
        if operation == "Union":
            result = fold_union([resolved[n] for n in names], queries=sql_queries)
            payload = {"result": {"wkt": result["wkt"], "srid": result["srid"],
                                  "geometry_type": _wkt_geometry_type(result["wkt"])}}
        elif operation in ("Intersection", "Difference", "SymDifference"):
            # Order matters for Difference (first minus second); the other two
            # are symmetric, but the named order is preserved regardless.
            result = execute_operation(operation, resolved[names[0]], resolved[names[1]], queries=sql_queries)
            geometry_type = _wkt_geometry_type(result["wkt"])
            payload = {"result": {"wkt": result["wkt"], "srid": result["srid"],
                                  "geometry_type": geometry_type}}
            if operation == "Intersection":
                # A correct intersection can come back as a line (two states
                # only touch) or empty (they don't touch at all) rather than a
                # polygon (a real overlap) - Scenario 14's own example. All
                # three are valid, complete answers; only the last one means
                # any land is actually shared, so that reading is spelled out
                # rather than left for the WKT prefix to imply.
                has_area = geometry_type in _AREA_GEOMETRY_TYPES
                payload["result"]["has_shared_area"] = has_area
                if not has_area:
                    payload["note"] = (
                        f"{names[0]} and {names[1]} share no area"
                        + (f" — they meet only along a boundary ({geometry_type})."
                           if geometry_type in ("LINESTRING", "MULTILINESTRING")
                           else " — their boundaries do not touch at all."
                           if geometry_type in ("GEOMETRYCOLLECTION", "EMPTY", "POINT", "MULTIPOINT")
                           else ".")
                    )
        elif operation == "BufferWithin":
            ref = resolved[names[0]]
            targets = [resolved[n] for n in names[1:]]
            matches = targets_within_buffer(ref, params.distance_km or 100.0, targets, queries=sql_queries)
            payload = {"reference": names[0], "distance_km": params.distance_km or 100.0,
                       "targets": matches}
        else:
            raise ValueError(f"Unknown operation {operation!r}")
    except ValueError as exc:
        # SRID mismatch or similar — a real error, not a gap.
        log.error("SPOP   │ Operation failed: %s", exc)
        t3 = time.perf_counter()
        total_ms = (t3 - t0) * 1000
        log_evaluation_metrics({
            "request_id": request_id, "timestamp": timestamp, "query": raw_query,
            "query_type": "SPATIAL_OPERATION",
            "classify_tokens": params.classify_tokens, "extract_tokens": params.extract_tokens, "classify_model": params.classify_model, "extract_model": params.extract_model,
            "extracted_data": params.extracted_data, "local_resolution": local_resolution,
            "kqml_exchanges": kqml_exchanges,
            "phase1_ms": (t1-t0)*1000, "phase2_ms": (t2-t1)*1000, "phase3_ms": 0.0,
            "total_ms": total_ms,
            "tokens_agent2": tokens_agent2, "tokens_agent1": tokens_agent1,
            "tokens_total": tokens_agent2 + tokens_agent1,
            "total_records": len(names), "total_data_points": len(names),
            "present_data_points": 0, "missing_data_points": len(names),
            "complete_records": 0, "partial_records": 0, "empty_records": len(names),
            "status": "error",
        })
        return {"request_id": request_id, "status": "error", "message": str(exc),
                "performance": {"phase1_ms": round((t1-t0)*1000,1), "phase2_ms": round((t2-t1)*1000,1),
                                "phase3_ms": 0.0, "total_ms": round(total_ms,1),
                                "tokens": {"agent_2": tokens_agent2, "agent_1": tokens_agent1,
                                           "total": tokens_agent2 + tokens_agent1}}}

    t3 = time.perf_counter()
    phase1_ms = (t1 - t0) * 1000
    phase2_ms = (t2 - t1) * 1000
    phase3_ms = (t3 - t2) * 1000
    total_ms  = (t3 - t0) * 1000

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] SPATIAL_OPERATION op=%s status=complete  %.0f ms",
             request_id, operation, total_ms)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id": request_id, "timestamp": timestamp, "query": raw_query,
        "query_type": "SPATIAL_OPERATION",
        "classify_tokens": params.classify_tokens, "extract_tokens": params.extract_tokens, "classify_model": params.classify_model, "extract_model": params.extract_model,
        "extracted_data": params.extracted_data,
        "local_resolution": {**local_resolution, "sources": sources},
        "kqml_exchanges": kqml_exchanges,
        "phase1_ms": phase1_ms, "phase2_ms": phase2_ms, "phase3_ms": phase3_ms, "total_ms": total_ms,
        "tokens_agent2": tokens_agent2, "tokens_agent1": tokens_agent1,
        "tokens_total": tokens_agent2 + tokens_agent1,
        "total_records": len(names), "total_data_points": len(names),
        "present_data_points": len(names), "missing_data_points": 0,
        "complete_records": len(names), "partial_records": 0, "empty_records": 0,
        "status": response_status(has_found=True, has_missing=False),
    })

    return {
        "request_id": request_id,
        "status":     response_status(has_found=True, has_missing=False),
        "query": {
            "raw":       raw_query,
            "type":      "SPATIAL_OPERATION",
            "operation": operation,
            "spatial":   names,
            "sources":   sources,
        },
        **payload,
        "performance": {
            "phase1_ms": round(phase1_ms, 1),
            "phase2_ms": round(phase2_ms, 1),
            "phase3_ms": round(phase3_ms, 1),
            "total_ms":  round(total_ms, 1),
            "tokens": {
                "agent_2": tokens_agent2,
                "agent_1": tokens_agent1,
                "total":   tokens_agent2 + tokens_agent1,
            },
        },
    }


def _handle_relationship_buffer(params, raw_query: str, request_id: str,
                                timestamp: str, t0: float, tokens_agent2: int):
    """Scenario 21: which cities lie within N km of a reference city. The targets
    cannot be named in advance, so a buffer zone is built once and sent to the
    peer as a shape to test its own catalogue against, rather than a named ask."""
    ref_city    = params.spatial[0] if params.spatial else None
    distance_km = params.distance_km or 100.0

    if not ref_city:
        return {"request_id": request_id, "status": "error",
                "message": "No reference city found in spatial-operation query.",
                "performance": {"phase1_ms": round((time.perf_counter()-t0)*1000, 1),
                                "phase2_ms": 0, "phase3_ms": 0, "total_ms": 0,
                                "tokens": {"agent_2": tokens_agent2, "agent_1": 0, "total": tokens_agent2}}}

    log.info("SPOP   │ Buffer query: %s within %s km", ref_city, distance_km)

    kqml_turns    = 0
    tokens_agent1 = 0
    kqml_exchanges: List[Dict] = []
    sql_queries: List[str] = []

    # ── Resolve the reference point, and build the buffer around it ───────────
    buffer = build_city_buffer(ref_city, distance_km, queries=sql_queries)
    ref_source = "Agent-2"
    if buffer is None:
        # Not held locally — ask the peer for just the reference point (a normal
        # named geometry ask), then build the buffer from that point ourselves.
        log.info("SPOP   │ %r not held locally — resolving reference point from Agent-1 ...", ref_city)
        from kqml_messaging import MessageFactory as _MF
        slot = _MF.missing_geometry_slot(spatial_entity=ref_city, entity_type="city")
        resp = send_kqml_geometry_ask([slot])
        if "ask_message" in resp and "tell_message" in resp:
            kqml_exchanges.append({"ask": resp["ask_message"], "tell": resp["tell_message"]})
        found = resp.get("found", [])
        if not found:
            log.warning("SPOP   │ Reference city %r not found anywhere — cannot build buffer", ref_city)
            t1 = t2 = t3 = time.perf_counter()
            total_ms = (t3 - t0) * 1000
            log_evaluation_metrics({
                "request_id": request_id, "timestamp": timestamp, "query": raw_query,
                "query_type": "SPATIAL_RELATIONSHIP_BUFFER",
                "classify_tokens": params.classify_tokens, "extract_tokens": params.extract_tokens, "classify_model": params.classify_model, "extract_model": params.extract_model,
                "extracted_data": params.extracted_data,
                "local_resolution": {
                    "reference_city": ref_city, "distance_km": distance_km,
                    "note": "reference city not held locally, and not found on the peer either",
                    "sql_queries": sql_queries,
                },
                "kqml_exchanges": kqml_exchanges,
                "phase1_ms": (t1-t0)*1000, "phase2_ms": 0.0, "phase3_ms": 0.0, "total_ms": total_ms,
                "tokens_agent2": tokens_agent2, "tokens_agent1": 0, "tokens_total": tokens_agent2,
                "total_records": 0, "total_data_points": 0, "present_data_points": 0,
                "missing_data_points": 0, "complete_records": 0, "partial_records": 0,
                "empty_records": 0, "status": response_status(has_found=False, has_missing=True),
            })
            return {
                "request_id": request_id, "status": response_status(has_found=False, has_missing=True),
                "query": {"raw": raw_query, "type": "SPATIAL_RELATIONSHIP_BUFFER",
                          "reference_city": ref_city, "distance_km": distance_km},
                "cities": [],
                "summary": {"total": 0, "found": 0},
                "performance": {"phase1_ms": round((t1-t0)*1000, 1), "phase2_ms": 0, "phase3_ms": 0,
                                "total_ms": round(total_ms, 1),
                                "tokens": {"agent_2": tokens_agent2, "agent_1": 0, "total": tokens_agent2}},
            }
        fg = found[0]
        buffer = buffer_from_point(fg.geometry, fg.srid, distance_km, queries=sql_queries)
        buffer["ref_name"] = fg.spatial_entity
        ref_source = "Agent-1"
        kqml_turns = 1

    ref_name = buffer["ref_name"]
    wkt, srid = buffer["wkt"], buffer["srid"]

    t1 = time.perf_counter()

    # ── Test our own catalogue first ───────────────────────────────────────────
    local_matches = cities_within_buffer(wkt, srid, exclude=[ref_name], queries=sql_queries)
    log.info("SPOP   │ Local catalogue matches: %d", len(local_matches))

    # ── Send the buffer to Agent-1 to test its own catalogue ──────────────────
    remote_matches = []
    exclude = [ref_name] + [fg.spatial_entity for fg in local_matches]
    try:
        resp = send_kqml_city_buffer_ask(wkt, srid, exclude)
        remote_matches = resp.get("found", [])
        kqml_turns = 1
        if "ask_message" in resp and "tell_message" in resp:
            kqml_exchanges.append({"ask": resp["ask_message"], "tell": resp["tell_message"]})
        log.info("SPOP   │ Agent-1 catalogue matches: %d", len(remote_matches))
    except Exception as exc:
        log.warning("SPOP   │ Agent-1 unreachable — %s", exc)

    t2 = time.perf_counter()

    cities = [
        {"city_name": fg.spatial_entity, "wkt": fg.geometry, "srid": fg.srid, "source": "Agent-2"}
        for fg in local_matches
    ] + [
        {"city_name": fg.spatial_entity, "wkt": fg.geometry, "srid": fg.srid, "source": "Agent-1"}
        for fg in remote_matches
    ]
    cities.sort(key=lambda c: c["city_name"])

    t3 = time.perf_counter()
    phase1_ms = (t1 - t0) * 1000
    phase2_ms = (t2 - t1) * 1000
    phase3_ms = (t3 - t2) * 1000
    total_ms  = (t3 - t0) * 1000

    status = response_status(has_found=bool(cities), has_missing=not cities)

    log.info(SEPARATOR)
    log.info("DONE   │ [%s] spatial-operation status=%s  found=%d  %.0f ms",
             request_id, status, len(cities), total_ms)
    log.info(SEPARATOR)

    log_evaluation_metrics({
        "request_id": request_id, "timestamp": timestamp, "query": raw_query,
        "query_type": "SPATIAL_RELATIONSHIP_BUFFER",
        "classify_tokens": params.classify_tokens, "extract_tokens": params.extract_tokens, "classify_model": params.classify_model, "extract_model": params.extract_model,
        "extracted_data": params.extracted_data,
        "local_resolution": {
            "reference_city":     ref_name,
            "reference_source":   ref_source,
            "distance_km":        distance_km,
            "local_catalogue_matches": len(local_matches),
            "note": "candidate cities cannot be named in advance — a buffer zone is built "
                    "once, tested against this agent's own catalogue, then sent to the peer "
                    "as a :spatial-query to test its catalogue too",
            "sql_queries": sql_queries,
        },
        "kqml_exchanges": kqml_exchanges,
        "phase1_ms": phase1_ms, "phase2_ms": phase2_ms, "phase3_ms": phase3_ms, "total_ms": total_ms,
        "tokens_agent2": tokens_agent2, "tokens_agent1": tokens_agent1,
        "tokens_total": tokens_agent2 + tokens_agent1,
        "total_records": len(cities), "total_data_points": len(cities),
        "present_data_points": len(cities), "missing_data_points": 0,
        "complete_records": len(cities), "partial_records": 0, "empty_records": 0,
        "status": status,
    })

    return {
        "request_id": request_id,
        "status":     status,
        "query": {
            "raw":            raw_query,
            "type":           "SPATIAL_RELATIONSHIP_BUFFER",
            "reference_city": ref_name,
            "reference_source": ref_source,
            "distance_km":    distance_km,
        },
        "cities": cities,
        "summary": {
            "total": len(cities),
            "from_agent_2": len(local_matches),
            "from_agent_1": len(remote_matches),
        },
        "performance": {
            "phase1_ms": round(phase1_ms, 1),
            "phase2_ms": round(phase2_ms, 1),
            "phase3_ms": round(phase3_ms, 1),
            "total_ms":  round(total_ms, 1),
            "tokens": {
                "agent_2": tokens_agent2,
                "agent_1": tokens_agent1,
                "total":   tokens_agent2 + tokens_agent1,
            },
        },
    }


@router.get("/health")
def health():
    log.info("Health check OK")
    return {"status": "ok", "agent": "Agent-2"}
