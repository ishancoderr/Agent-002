"""
Resolve geometry requests against the local cities/states catalog.

Used when a peer agent sends missing_geometries in a KQML ask.
Returns WKT + srid for found features; passes through unresolved ones as still-missing.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from kqml_messaging import EntityType, MissingGeometrySlot, FoundGeometrySlot, MessageFactory, check_srid_agreement

from ..database import SessionLocal
from ..pipeline.gazetteer import normalize_entity_name

log = logging.getLogger("agent2.retrieval.geometry_resolver")


def resolve_geometries(
    requests: List[MissingGeometrySlot],
    queries: Optional[List[str]] = None,
) -> Tuple[List[FoundGeometrySlot], List[MissingGeometrySlot]]:
    """
    Look up each requested geometry from the local catalog.

    Returns:
        found   — FoundGeometrySlot list (WKT + srid=4326)
        missing — MissingGeometrySlot list not found locally
    When `queries` is given, every SQL statement actually run against Agent-2's
    DB is appended to it, for evaluation-log tracing.
    """
    found:   List[FoundGeometrySlot]   = []
    missing: List[MissingGeometrySlot] = []

    db = SessionLocal()
    try:
        for req in requests:
            entity_type = req.entity_type.value  # plain str, not the EntityType repr, for logging
            wkt, matched_name, _exists = _lookup(req.spatial_entity, entity_type, db, queries=queries)
            if wkt:
                log.info("       │ GEOM FOUND  : %s (%s) → %s…", req.spatial_entity, entity_type, wkt[:50])
                found.append(FoundGeometrySlot(
                    spatial_entity=matched_name,
                    entity_type=req.entity_type,
                    geometry=wkt,
                    srid=4326,
                ))
            else:
                log.info("       │ GEOM MISSING: %s (%s) — not in local catalog", req.spatial_entity, entity_type)
                missing.append(req)
    finally:
        db.close()

    return found, missing


def _lookup(entity_name: str, entity_type: str, db, queries: Optional[List[str]] = None):
    """
    Try to find the entity in the DB using multiple name forms:
    1. Exact match
    2. Alias (via the gazetteer, which is built from the name/alias JSON and
       verified against the database — see pipeline/gazetteer.py; private
       alias tables used to live here and in spatial_validator, pointing in
       opposite directions, and each produced names the database does not
       hold)
    3. Case-insensitive ILIKE
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
    if entity_type not in ("city", "state"):
        log.warning("       │ Unknown entity_type %r for %r — skipping", entity_type, entity_name)
        return None, None, False

    candidates = list(dict.fromkeys(filter(None, [
        normalize_entity_name(entity_name, "city"),   # the spelling the database uses
        entity_name,                                  # then the name as given
    ])))

    table    = "cities" if entity_type == "city"  else "states"
    col      = "centroid" if entity_type == "city" else "geo_shape"
    name_col = "city_name" if entity_type == "city" else "state_name"

    # 1 — exact match with valid geometry (alias first, then original)
    for name in candidates:
        sql = (f"SELECT ST_AsText({col}), {name_col} FROM {table} "
               f"WHERE {name_col} = '{name}' AND ST_AsText({col}) IS NOT NULL LIMIT 1")
        if queries is not None:
            queries.append(sql)
        row = db.execute(
            text(f"""
                SELECT ST_AsText({col}), {name_col} FROM {table}
                WHERE {name_col} = :n AND ST_AsText({col}) IS NOT NULL
                LIMIT 1
            """),
            {"n": name},
        ).fetchone()
        if row:
            log.info("       │ Resolved (exact)  : %r → %r", entity_name, row[1])
            return row[0], row[1], True

    # 2 — entity exists but geometry is NULL → stop here, do NOT fall through to partial
    for name in candidates:
        sql = f"SELECT 1 FROM {table} WHERE {name_col} = '{name}' LIMIT 1"
        if queries is not None:
            queries.append(sql)
        exists = db.execute(
            text(f"SELECT 1 FROM {table} WHERE {name_col} = :n LIMIT 1"),
            {"n": name},
        ).fetchone()
        if exists:
            log.info("       │ %s %r found in DB but geometry is NULL — will ask peer", entity_type, name)
            return None, None, True

    # 3 — entity not in DB at all: try case-insensitive full match with valid geometry
    for name in candidates:
        sql = (f"SELECT ST_AsText({col}), {name_col} FROM {table} "
               f"WHERE {name_col} ILIKE '{name}' AND ST_AsText({col}) IS NOT NULL LIMIT 1")
        if queries is not None:
            queries.append(sql)
        row = db.execute(
            text(f"""
                SELECT ST_AsText({col}), {name_col} FROM {table}
                WHERE {name_col} ILIKE :n AND ST_AsText({col}) IS NOT NULL
                LIMIT 1
            """),
            {"n": name},
        ).fetchone()
        if row:
            log.info("       │ Resolved (ilike)  : %r → %r", entity_name, row[1])
            return row[0], row[1], True

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
            sql = (f"SELECT city_name, ST_AsText(ST_Buffer(centroid::geography, {distance_km * 1000})::geometry), "
                   f"ST_SRID(centroid) FROM cities WHERE city_name = '{name}' AND centroid IS NOT NULL LIMIT 1")
            if queries is not None:
                queries.append(sql)
            row = db.execute(
                text("""
                    SELECT city_name,
                           ST_AsText(ST_Buffer(centroid::geography, :dist)::geometry),
                           ST_SRID(centroid)
                    FROM cities
                    WHERE city_name = :n AND centroid IS NOT NULL
                    LIMIT 1
                """),
                {"n": name, "dist": distance_km * 1000},
            ).fetchone()
            if row:
                return {"ref_name": row[0], "wkt": row[1], "srid": row[2] or 4326}
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
        sql = (f"SELECT ST_AsText(ST_Buffer(ST_GeomFromText('{wkt_point[:60]}...', {srid})"
               f"::geography, {distance_km * 1000})::geometry)")
        if queries is not None:
            queries.append(sql)
        row = db.execute(
            text("""
                SELECT ST_AsText(
                    ST_Buffer(ST_GeomFromText(:wkt, :srid)::geography, :dist)::geometry
                )
            """),
            {"wkt": wkt_point, "srid": srid, "dist": distance_km * 1000},
        ).fetchone()
        return {"wkt": row[0], "srid": srid}
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
        sql = (f"SELECT city_name, ST_AsText(centroid), ST_SRID(centroid) FROM cities "
               f"WHERE centroid IS NOT NULL AND ST_Within(centroid, ST_GeomFromText('{wkt[:60]}...', {srid})) "
               f"AND NOT (city_name = ANY({exclude}))")
        if queries is not None:
            queries.append(sql)
        rows = db.execute(
            text("""
                SELECT city_name, ST_AsText(centroid), ST_SRID(centroid)
                FROM cities
                WHERE centroid IS NOT NULL
                  AND ST_Within(centroid, ST_GeomFromText(:wkt, :srid))
                  AND NOT (city_name = ANY(:exclude))
            """),
            {"wkt": wkt, "srid": srid, "exclude": exclude},
        ).fetchall()
        return [
            MessageFactory.found_geometry_slot(
                spatial_entity=r[0], entity_type=EntityType.CITY, geometry=r[1], srid=r[2] or srid,
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

_OP_SQL_FN = {
    "Union": "ST_Union",
    "Intersection": "ST_Intersection",
    "Difference": "ST_Difference",          # order matters: fn(A, B) = A minus B
    "SymDifference": "ST_SymDifference",
}


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

        for et in ("city", "state"):
            if et == entity_type:
                continue
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
    fn = _OP_SQL_FN[operation]
    db = SessionLocal()
    try:
        sql = f"SELECT ST_AsText({fn}(<geom_a wkt>, <geom_b wkt>))"
        if queries is not None:
            queries.append(sql)
        row = db.execute(
            text(f"""
                SELECT ST_AsText({fn}(
                    ST_GeomFromText(:wkt_a, :srid_a),
                    ST_GeomFromText(:wkt_b, :srid_b)
                ))
            """),
            {"wkt_a": geom_a["wkt"], "srid_a": geom_a["srid"],
             "wkt_b": geom_b["wkt"], "srid_b": geom_b["srid"]},
        ).fetchone()
        return {"wkt": row[0], "srid": geom_a["srid"]}
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
            sql = (f"SELECT ST_Intersects(<{t['name']} wkt>, "
                   f"ST_Buffer(<ref wkt>::geography, {distance_km * 1000})::geometry)")
            if queries is not None:
                queries.append(sql)
            row = db.execute(
                text("""
                    SELECT ST_Intersects(
                        ST_GeomFromText(:wkt_t, :srid_t),
                        ST_Buffer(ST_GeomFromText(:wkt_ref, :srid_ref)::geography, :dist)::geometry
                    )
                """),
                {"wkt_t": t["wkt"], "srid_t": t["srid"], "wkt_ref": ref_geom["wkt"],
                 "srid_ref": ref_geom["srid"], "dist": distance_km * 1000},
            ).fetchone()
            results.append({"name": t["name"], "meets_zone": bool(row[0])})
        return results
    finally:
        db.close()
