"""
Resolve MissingGeometrySlot requests against Agent-2's local cities/states catalog.
Returns WKT geometries (SRID 4326) for cities (centroid) and states (geo_shape).

Every SQL statement below comes from SqlGenerator (agent2/retrieval/
sql_generator.py) rather than being hand-written here: table/column
identifiers, which entity_type maps to which table, and the SQL text itself
are all resolved from config/schema/entities.yaml through it, not duplicated
in this file. Each shape (an exact-name lookup, a buffer, a binary
operation, ...) is generated once per process and reused — see
SqlGenerator._cached_query() — since none of these statements' SQL TEXT
varies by anything other than which entity_type/operation they're for; only
the bound values differ call to call, and those always travel as
SQLAlchemy params, never interpolated into the SQL text.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from kqml_messaging import MissingGeometrySlot, FoundGeometrySlot, MessageFactory, check_srid_agreement

from ..database import SessionLocal
from ..pipeline.gazetteer import normalize_entity_name
from ..pipeline.query_params import VALID_ENTITY_TYPES
from .sql_generator import BuiltQuery, SqlGenerator, ENTITY_ALIAS, WKT_ALIAS, SRID_ALIAS, MEETS_ZONE_ALIAS

log = logging.getLogger("agent2.retrieval.geometry_resolver")

# City names are normalised through the gazetteer, which is built from the
# name/alias JSON and verified against the database — see pipeline/gazetteer.py.
# Private alias tables used to live here and in spatial_validator, pointing in
# opposite directions; each produced names the database does not hold.

# Buffer/within functions below are specifically about cities — which entity
# type they operate on is a fact about what those operations mean (see their
# own docstrings), not something to parameterize.

# One generator per process so its per-shape SQL cache (_cached_query()) is
# actually shared across every lookup this module ever does, not rebuilt —
# and re-validated by an LLM call — on every single call.
_generator = SqlGenerator()


def _run(db, built: BuiltQuery, queries: Optional[List[str]] = None):
    """Execute a built statement and record its readable form for the log,
    the same convention agent2/retrieval/local_store.py's LocalStore._run()
    uses — a statement is never run without its human-readable form (values
    substituted, never executed) being available for the query trace."""
    if queries is not None:
        queries.append(built.readable)
    log.debug("SQL > %s\n%s", built.label, built.readable)
    return db.execute(text(built.sql), built.params)


def resolve_geometries(
    slots: List[MissingGeometrySlot],
    queries: Optional[List[str]] = None,
) -> Tuple[List[FoundGeometrySlot], List[MissingGeometrySlot]]:
    """
    Try to resolve each slot from the local DB.
    Returns (found_list, still_missing_list). When `queries` is given, every
    SQL statement actually run against Agent-2's DB is appended to it, for
    evaluation-log tracing.

    Used for a PEER's own ask (kqml_controller.py) — every slot here already
    arrived as a real MissingGeometrySlot, already Pydantic-validated by the
    shared library on the way in, so requiring one here costs nothing more.
    A locally-initiated lookup (see resolve_entities() below) is a different
    situation: it must not require building one of these — and paying for
    Pydantic construction — just to check the local DB, when a plain dict is
    all a local lookup ever needed in the first place.
    """
    found:   List[FoundGeometrySlot]   = []
    missing: List[MissingGeometrySlot] = []

    db = SessionLocal()
    try:
        for slot in slots:
            entity_type = slot.entity_type  # already a plain str
            wkt, matched_name, _exists = _lookup(slot.spatial_entity, entity_type, db, queries=queries)
            if wkt is not None:
                log.info("       │ Geometry FOUND : %s → %s (%s)",
                         slot.spatial_entity, matched_name, entity_type)
                found.append(
                    MessageFactory.found_geometry_slot(
                        spatial_entity=matched_name,   # actual DB name
                        entity_type=slot.entity_type,
                        geometry=wkt,
                        srid=4326,
                    )
                )
            else:
                log.info("       │ Geometry MISSING: %s (%s)", slot.spatial_entity, entity_type)
                missing.append(slot)
    finally:
        db.close()

    return found, missing


def resolve_entities(
    entities: List[Dict[str, str]],
    queries: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    """Like resolve_geometries(), for a user's own GEOMETRY_LOOKUP request
    (query_controller.py's _handle_geometry()) rather than a peer's ask.

    Takes plain {"entity_name", "entity_type"} dicts — the shape
    QueryParams.entities already carries — instead of requiring a real
    MissingGeometrySlot per entity up front. entity_type is a plain string
    now (kqml_messaging.MissingGeometrySlot.entity_type is no longer
    restricted to a fixed enum), so this isn't about avoiding a validation
    failure any more — it's about not paying for building and Pydantic-
    validating a wire-protocol object just to check the local DB, when
    nothing here is ever going to cross the wire unless the entity turns
    out to be genuinely missing. query_controller.py builds a real slot
    itself, per entity, only for what comes back in `missing` here.

    Returns (found, missing): found is
    [{"entity_name", "entity_type", "wkt", "srid"}, ...] (plain dicts, not
    FoundGeometrySlot, for the same reason); missing is the subset of
    `entities` not found here, unchanged."""
    found:   List[Dict[str, object]] = []
    missing: List[Dict[str, str]]    = []

    db = SessionLocal()
    try:
        for entity in entities:
            entity_name, entity_type = entity["entity_name"], entity["entity_type"]
            wkt, matched_name, _exists = _lookup(entity_name, entity_type, db, queries=queries)
            if wkt is not None:
                log.info("       │ Geometry FOUND : %s → %s (%s)", entity_name, matched_name, entity_type)
                found.append({
                    "entity_name": matched_name,   # actual DB name
                    "entity_type": entity_type,
                    "wkt":         wkt,
                    "srid":        4326,
                })
            else:
                log.info("       │ Geometry MISSING: %s (%s)", entity_name, entity_type)
                missing.append(entity)
    finally:
        db.close()

    return found, missing


def _lookup(entity_name: str, entity_type: str, db, queries: Optional[List[str]] = None):
    """
    Try to find the entity in the DB using multiple name forms:
    1. Exact match
    2. Alias (German↔English)
    3. Case-insensitive ILIKE
    4. Partial ILIKE (shortest match wins)
    Returns (wkt, matched_name, exists). `exists` is True whenever a row was
    found under this entity_type - even one with a NULL geometry - which lets
    a caller distinguish "this name has no row at all under this type" from
    "this name exists here but its shape is missing". resolve_named_geometry
    depends on that distinction: without it, a name that exists under both
    tables (Berlin is both a city and a state) would fall through to the wrong
    type the moment the correct one's shape is absent, silently returning a
    city point in place of a missing state polygon. When `queries` is given,
    every attempted SQL statement (values substituted, for readability) is
    appended to it.
    """
    if entity_type not in VALID_ENTITY_TYPES:
        log.warning("       │ Unknown entity_type %r for %r — skipping", entity_type, entity_name)
        return None, None, False

    # Build candidate name list — English alias first, then original German form
    candidates = list(dict.fromkeys(filter(None, [
        normalize_entity_name(entity_name, "city"),   # the spelling the database uses
        entity_name,                       # then the name as given
    ])))

    # 1 — exact match with valid geometry (alias first, then original)
    for name in candidates:
        row = _run(db, _generator.entity_exact_geometry(name, entity_type), queries).fetchone()
        if row:
            matched_name = row._mapping[ENTITY_ALIAS]
            log.info("       │ Resolved (exact)  : %r → %r", entity_name, matched_name)
            return row._mapping[WKT_ALIAS], matched_name, True

    # 2 — entity exists but geometry is NULL → stop here, do NOT fall through to partial
    for name in candidates:
        exists = _run(db, _generator.entity_exists(name, entity_type), queries).fetchone()
        if exists:
            log.info("       │ %s %r found in DB but geometry is NULL — will ask peer", entity_type, name)
            return None, None, True

    # 3 — entity not in DB at all: try case-insensitive full match with valid geometry
    for name in candidates:
        row = _run(db, _generator.entity_ilike_geometry(name, entity_type), queries).fetchone()
        if row:
            matched_name = row._mapping[ENTITY_ALIAS]
            log.info("       │ Resolved (ilike)  : %r → %r", entity_name, matched_name)
            return row._mapping[WKT_ALIAS], matched_name, True

    log.info("       │ %s %r not found in DB (tried: %s)", entity_type, entity_name, candidates)
    return None, None, False


# ── Buffer / within (Scenario 21: target that cannot be named) ──────────────
# A city-distance query ("which cities lie within 100 km of X") cannot be
# expressed as a named missing-slot, since the cities that satisfy it are the
# answer, not the input. Instead a buffer zone is built once (locally, if the
# reference city is held; via the peer's geometry catalog otherwise) and sent
# to the peer as a shape to test its own catalog against, per Section 4's
# achieve/spatial-query pattern.

def build_city_buffer(
    ref_city: str, distance_km: float, queries: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    """Build a buffer polygon (metres) around a locally-held city's centroid.
    Returns {"ref_name", "wkt", "srid"} or None if the reference city isn't held here."""
    candidates = list(dict.fromkeys(filter(None, [
        normalize_entity_name(ref_city, "city"),   # the spelling the database uses
        ref_city,                          # then the name as given
    ])))
    db = SessionLocal()
    try:
        for name in candidates:
            row = _run(db, _generator.city_buffer(name, distance_km), queries).fetchone()
            if row:
                return {
                    "ref_name": row._mapping[ENTITY_ALIAS],
                    "wkt": row._mapping[WKT_ALIAS],
                    "srid": row._mapping[SRID_ALIAS] or 4326,
                }
        return None
    finally:
        db.close()


def buffer_from_point(
    wkt_point: str, srid: int, distance_km: float, queries: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Build a buffer polygon around an already-known point (e.g. one resolved
    from the peer), without requiring a local row for the reference city."""
    db = SessionLocal()
    try:
        row = _run(db, _generator.point_buffer(wkt_point, srid, distance_km), queries).fetchone()
        return {"wkt": row._mapping[WKT_ALIAS], "srid": srid}
    finally:
        db.close()


def cities_within_buffer(
    wkt: str, srid: int, exclude: Optional[List[str]] = None, queries: Optional[List[str]] = None,
) -> List[FoundGeometrySlot]:
    """Test the LOCAL cities catalog against a buffer polygon, per Scenario 21's
    achieve/spatial-query pattern: the peer runs the test over its own catalog
    rather than being asked for a fact it could have named."""
    exclude = [e for e in (exclude or []) if e]
    db = SessionLocal()
    try:
        rows = _run(db, _generator.within_buffer(wkt, srid, exclude), queries).fetchall()
        return [
            MessageFactory.found_geometry_slot(
                spatial_entity=r._mapping[ENTITY_ALIAS],
                entity_type="city",
                geometry=r._mapping[WKT_ALIAS],
                srid=r._mapping[SRID_ALIAS] or srid,
            )
            for r in rows
        ]
    finally:
        db.close()


# ── Constructive operations (Section 3: Scenarios 13-16, 20) ────────────────
# Both agents run the same code, so an operation is never itself missing —
# only an input shape can be (Section 3.5). Every operation here first
# resolves each named entity's geometry (locally or via the existing
# missing-geometries exchange), then runs a local computation once all
# shapes have arrived. SRID agreement is checked before the operation is
# attempted, exactly as the document specifies, reusing the same check the
# kqml_messaging library exposes for this purpose.

def resolve_named_geometry(
    name: str, entity_type: str, queries: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    """Look up a single named entity's geometry in the local catalogue.

    The declared entity_type is tried first and is authoritative if the name
    exists under it at all — even with a NULL geometry. Only when the name has
    no row whatsoever under that type is a different one considered, which is
    what lets a BufferWithin reference resolve as a city when the operation's
    blanket entity_type was "state" (Scenario 20's own setup: "which of NRW
    and Niedersachsen lie within 100 km of Dortmund" — Dortmund is a city, the
    targets are states, one entity_type does not fit every name).

    That same fallback must not fire for a genuine gap: Berlin is both a city
    and a state, so a Berlin state polygon that is missing here would silently
    resolve as Berlin's city point instead — the wrong feature entirely,
    returned as if it were the state's territory — if type-guessing did not
    stop the moment it learns the name exists under the type actually asked
    for. Returns {"name": matched_name, "wkt": ..., "srid": 4326}, or None if
    genuinely not held here under any applicable type."""
    db = SessionLocal()
    try:
        wkt, matched_name, exists = _lookup(name, entity_type, db, queries=queries)
        if wkt is not None:
            return {"name": matched_name, "wkt": wkt, "srid": 4326}
        if exists:
            # Found under the type that was actually asked for; its geometry
            # is genuinely absent here. Ask the peer for THIS type, not a
            # different feature that happens to share the name.
            return None

        for et in sorted(VALID_ENTITY_TYPES - {entity_type}):
            wkt, matched_name, _exists = _lookup(name, et, db, queries=queries)
            if wkt is not None:
                return {"name": matched_name, "wkt": wkt, "srid": 4326}
        return None
    finally:
        db.close()


def execute_operation(
    operation: str, geom_a: Dict[str, object], geom_b: Dict[str, object],
    queries: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Run Union/Intersection/Difference/SymDifference on two already-resolved
    geometries. An operation on shapes held in different reference systems is an
    error, checked here before the operation runs rather than after it fails."""
    check_srid_agreement(geom_a["srid"], geom_b["srid"])
    db = SessionLocal()
    try:
        built = _generator.binary_operation(
            operation, geom_a["wkt"], geom_a["srid"], geom_b["wkt"], geom_b["srid"]
        )
        row = _run(db, built, queries).fetchone()
        return {"wkt": row._mapping[WKT_ALIAS], "srid": geom_a["srid"]}
    finally:
        db.close()


def fold_union(geometries: List[Dict[str, object]], queries: Optional[List[str]] = None) -> Dict[str, object]:
    """Union takes two shapes at a time; combining more than two is a fold across
    the pairwise operation, carrying the intermediate shape forward without a name
    (Section 3.3's explanation of how Union generalizes to a group)."""
    result = geometries[0]
    for g in geometries[1:]:
        result = execute_operation("Union", result, g, queries=queries)
    return result


def targets_within_buffer(
    ref_geom: Dict[str, object], distance_km: float, targets: List[Dict[str, object]],
    queries: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Named-target buffer test (Scenario 20): build a zone around ref_geom and test
    each already-resolved named target against it. Unlike Scenario 21's open-target
    buffer, this zone is purely local scratch and never crosses the wire, since every
    target is named and its shape was fetched the ordinary way."""
    db = SessionLocal()
    try:
        results = []
        for t in targets:
            built = _generator.intersects_buffer(
                t["wkt"], t["srid"], ref_geom["wkt"], ref_geom["srid"], distance_km
            )
            row = _run(db, built, queries).fetchone()
            results.append({"name": t["name"], "meets_zone": bool(row._mapping[MEETS_ZONE_ALIAS])})
        return results
    finally:
        db.close()
