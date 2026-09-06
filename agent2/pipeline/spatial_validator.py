"""
Step 2 — Resolve spatial-relationship queries into concrete state name lists
using PostGIS functions (ST_Touches, ST_Azimuth, ST_DWithin).
Skipped entirely for DIRECT_LOOKUP queries.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import SessionLocal
from .gazetteer import normalize_entity_name
from .query_parser import QueryParams, SpatialRelationship

log = logging.getLogger("agent2.pipeline.spatial")


def _all_state_geometries(db: Session) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Every state's geometry as (name, wkt) pairs — local rows where held,
    plus a peer fetch for any gap, used for this computation only.

    Adjacency, direction and distance each need every candidate state's shape
    to know the full result set: a candidate with no shape would otherwise
    just vanish from a "which states touch X" query, since ST_Intersects and
    ST_DWithin return NULL (falsy) against a NULL geometry, not an "unknown".
    An earlier version filled that gap by writing the peer's answer straight
    into the local `states` table and committing it. That made a relationship
    query silently and permanently erase a genuine geometry gap belonging to a
    state the query never asked about, which corrupted the geometry-lookup
    scenarios (9-12) that same table backs — running an adjacency check on
    Bayern/Sachsen made an unrelated state's "not held here" answer disappear
    for good. The fetch is now transient, exactly as the peer-city-coordinate
    lookup below already treats it: used to compute this one answer, and
    never written back.

    Returns (pairs, still_missing) — still_missing lists states neither store
    could supply a shape for.
    """
    rows = db.execute(text("SELECT state_name, ST_AsText(geo_shape) FROM states")).fetchall()
    held: dict = {name: wkt for name, wkt in rows if wkt is not None}
    missing = [name for name, wkt in rows if wkt is None]
    if not missing:
        return list(held.items()), []

    log.info("       │ States with no local shape: %s — fetching from the peer "
             "for this computation only (not persisted)", missing)
    try:
        from kqml_messaging import MessageFactory
        from ..messaging.kqml_geometry_client import send_kqml_geometry_ask

        slots = [MessageFactory.missing_geometry_slot(spatial_entity=s, entity_type="state")
                 for s in missing]
        found = send_kqml_geometry_ask(slots).get("found", [])
        for fg in found:
            held[fg.spatial_entity] = fg.geometry
    except Exception as exc:                    # noqa: BLE001
        log.warning("       │ Could not fetch missing geometries from the peer: %s", exc)

    still_missing = [name for name in missing if name not in held]
    if still_missing:
        log.warning("       │ Genuinely absent from both stores: %s", still_missing)
    return list(held.items()), still_missing


def _geoms_cte(pairs: List[Tuple[str, str]]) -> Tuple[str, dict]:
    """A 'geoms(state_name, shape)' CTE fragment and its bound parameters,
    built from (name, wkt) pairs, so a query can join against every state's
    shape without any of them living in the `states` table."""
    params: dict = {}
    rows_sql = []
    for i, (name, wkt) in enumerate(pairs):
        params[f"gn{i}"] = name
        params[f"gw{i}"] = wkt
        rows_sql.append(f"(:gn{i}, ST_GeomFromText(:gw{i}, 4326))")
    values_sql = ",\n                ".join(rows_sql)
    cte = f"geoms(state_name, shape) AS (\n                VALUES {values_sql}\n            )"
    return cte, params


_DIRECTION_SQL = {
    "north_of": "(az <= 45 OR az >= 315)",
    "south_of": "(az BETWEEN 135 AND 225)",
    "east_of":  "(az BETWEEN 45  AND 135)",
    "west_of":  "(az BETWEEN 225 AND 315)",
}


def validate_spatial(params: QueryParams) -> QueryParams:
    if params.query_type == "DIRECT_LOOKUP":
        log.info("       │ DIRECT_LOOKUP — spatial resolution skipped")
        return params

    db = SessionLocal()
    try:
        log.info("       │ Resolving %s via PostGIS ...", params.query_type)
        unknown: List[str] = []
        if params.query_type == "SPATIAL_ADJACENCY":
            params.spatial, unknown = _adjacency(params.spatial_relationship, db)
        elif params.query_type == "SPATIAL_DIRECTION":
            params.spatial, unknown = _direction(params.spatial_relationship, db)
        elif params.query_type == "SPATIAL_DISTANCE":
            params.spatial, unknown = _distance(params.spatial_relationship, db)
        params.unknown_states = unknown
        if unknown:
            log.warning("       │ UNKNOWN (no geometry anywhere, excluded from the "
                        "candidate test): %s", unknown)

        # A question that named a subject asked a yes/no, not for a list. The
        # set that was just computed is the set for which the relationship
        # holds, so the verdict is simply whether the subject is in it - unless
        # the subject itself is one of the unknown states, in which case there
        # is no candidate set to have tested it against at all.
        params.verdict = _verdict_for(params.spatial_relationship, params.spatial, unknown)
        if params.spatial_relationship and params.spatial_relationship.subject:
            log.info("       │ Verdict: %s %s %s -> %s",
                     params.spatial_relationship.subject,
                     params.spatial_relationship.type,
                     params.spatial_relationship.refs,
                     "YES" if params.verdict else "NO" if params.verdict is False else "UNKNOWN")
        log.info("       │ Resolved to %d state(s): %s", len(params.spatial), params.spatial)
    finally:
        db.close()

    return params


def _verdict_for(rel: Optional[SpatialRelationship], qualifying: List[str],
                 unknown: List[str]) -> Optional[bool]:
    """Yes/no for a question that named a subject, else None.

    None also covers a case distinct from "no subject was named": the subject
    was named, but its own geometry is held nowhere, so the relationship could
    not be tested for it at all. A null input must not collapse into False -
    Scenario 17's own reasoning ("a shape that is null must not be reported as
    a false, since not knowing whether two states touch is a different answer
    from knowing that they do not") - so that case is also reported as
    unknown (None) rather than as a confident "no" the data cannot support.

    The same applies if one of the *references* is unresolvable: the
    candidate test can only run against states this computation actually knew
    the shape of, so a reference with no geometry anywhere makes the whole
    test undetermined, not a clean "no" for every subject.

    Matching is done on the normalised name so that a subject written as
    "München" is still recognised in a list holding the database's "Munich"."""
    if rel is None or not rel.subject:
        return None
    normalized_unknown = {normalize_entity_name(name, "state") for name in unknown}
    if any(normalize_entity_name(ref, "state") in normalized_unknown for ref in rel.refs):
        return None
    subject = normalize_entity_name(rel.subject, "state")
    if subject in normalized_unknown:
        return None
    return any(subject == normalize_entity_name(name, "state") for name in qualifying)


def _adjacency(rel: SpatialRelationship, db: Session) -> Tuple[List[str], List[str]]:
    pairs, unknown = _all_state_geometries(db)
    if not pairs:
        return [], unknown
    cte, geom_params = _geoms_cte(pairs)

    sets: List[set] = []
    for ref in rel.refs:
        rows = db.execute(
            text(f"""
                WITH {cte}
                SELECT g2.state_name
                FROM geoms g1
                JOIN geoms g2
                  ON ST_Intersects(g1.shape, g2.shape)
                 AND NOT ST_Equals(g1.shape, g2.shape)
                WHERE g1.state_name = :ref
                  AND g2.state_name != :ref
            """),
            {**geom_params, "ref": ref},
        ).fetchall()
        sets.append({r[0] for r in rows})
        log.info("       │ States touching %s: %s", ref, sorted({r[0] for r in rows}))

    if not sets:
        return [], unknown
    result = sets[0]
    for s in sets[1:]:
        result &= s
    return sorted(result), unknown


def _direction(rel: SpatialRelationship, db: Session) -> Tuple[List[str], List[str]]:
    ref  = rel.refs[0] if rel.refs else "Bayern"
    cond = _DIRECTION_SQL.get(rel.type, _DIRECTION_SQL["north_of"])

    pairs, unknown = _all_state_geometries(db)
    if not pairs:
        return [], unknown
    cte, geom_params = _geoms_cte(pairs)

    rows = db.execute(
        text(f"""
            WITH {cte},
            azimuths AS (
                SELECT g2.state_name,
                       degrees(ST_Azimuth(
                           ST_Centroid(g1.shape),
                           ST_Centroid(g2.shape))) AS az
                FROM geoms g1
                JOIN geoms g2 ON g1.state_name != g2.state_name
                WHERE g1.state_name = :ref
            )
            SELECT state_name FROM azimuths WHERE {cond} ORDER BY az
        """),
        {**geom_params, "ref": ref},
    ).fetchall()

    return [r[0] for r in rows], unknown


# City names are normalised through the gazetteer, which is built from the
# name/alias JSON and verified against the database. A private alias table used
# to live here mapping the other way (Munich -> München); the database stores
# "Munich", so that table produced names no row has. It only ever worked
# because the unmodified name happened to be tried first.


def _resolve_city_coords(city: str, db: Session):
    """
    Find a city in the DB and return its (lat, lng) from the cities table.
    Resolution order:
      1. Exact match on city_name
      2. Alias map → exact match
      3. Case-insensitive ILIKE match
      4. Partial ILIKE match (city name contains the search term)
    Returns (lat, lng) tuple or None if not found.
    """
    candidates = list(dict.fromkeys(filter(None, [
        normalize_entity_name(city, "city"),   # the spelling the database uses
        city,                                  # then the name as given
    ])))

    # Exact and alias matches first
    for name in candidates:
        row = db.execute(
            text("SELECT lat, lng FROM cities WHERE city_name = :n LIMIT 1"),
            {"n": name},
        ).fetchone()
        if row:
            log.info("       │ City resolved (exact) : %r → lat=%s lng=%s", city, row[0], row[1])
            return row[0], row[1]

    # Case-insensitive full match
    for name in candidates:
        row = db.execute(
            text("SELECT lat, lng, city_name FROM cities WHERE city_name ILIKE :n LIMIT 1"),
            {"n": name},
        ).fetchone()
        if row:
            log.info("       │ City resolved (ilike) : %r → %r lat=%s lng=%s", city, row[2], row[0], row[1])
            return row[0], row[1]

    # Partial match — order by name length so shortest (most exact) match wins
    for name in candidates:
        row = db.execute(
            text("""
                SELECT lat, lng, city_name FROM cities
                WHERE city_name ILIKE :n
                ORDER BY LENGTH(city_name)
                LIMIT 1
            """),
            {"n": f"%{name}%"},
        ).fetchone()
        if row:
            log.info("       │ City resolved (partial): %r → %r lat=%s lng=%s", city, row[2], row[0], row[1])
            return row[0], row[1]

    # Not held here — ask the peer before giving up. Agent-2 holds no centroid
    # for Munich, so without this step a distance query measured from Munich
    # could not be answered at all even though the peer has the point.
    log.info("       │ City %r not in local DB (tried: %s) — asking the peer", city, candidates)
    coords = _fetch_city_coords_from_peer(city)
    if coords:
        return coords

    log.warning("       │ City %r not found locally or at the peer", city)
    return None


def _fetch_city_coords_from_peer(city: str) -> Optional[Tuple[float, float]]:
    """Ask the peer for a city's centroid over KQML and read lat/lng out of it.

    The coordinates are used for this query only and are not written back to
    the local store: this agent still does not hold that city, and recording it
    here would misrepresent which partition the value came from."""
    try:
        from kqml_messaging import MessageFactory

        from ..messaging.kqml_geometry_client import send_kqml_geometry_ask

        slot = MessageFactory.missing_geometry_slot(spatial_entity=city, entity_type="city")
        found = send_kqml_geometry_ask([slot]).get("found", [])
        if not found:
            log.warning("       │ Peer has no geometry for city %r", city)
            return None

        coords = _parse_point_wkt(found[0].geometry)
        if coords is None:
            log.warning("       │ Cannot parse WKT for city %r: %r", city, found[0].geometry)
            return None

        log.info("       │ Peer centroid for %r: lat=%s lng=%s", city, coords[0], coords[1])
        return coords
    except Exception as exc:                    # noqa: BLE001
        # An unreachable peer must not take the query down; the caller treats a
        # None as "not found", which is the honest answer here.
        log.warning("       │ Peer city fetch failed for %r: %s", city, exc)
        return None


def _parse_point_wkt(wkt: str) -> Optional[Tuple[float, float]]:
    """Parse WKT POINT(lng lat) -> (lat, lng). Returns None if it is not a POINT."""
    match = re.match(r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", wkt.strip(), re.IGNORECASE)
    if not match:
        return None
    lng, lat = float(match.group(1)), float(match.group(2))
    return lat, lng


# No distance tolerance is applied. "Within 100 km" is compared against exactly
# 100000 m, as the specification defines it: a tolerance would silently widen
# the threshold and, because only one agent had it, made the two agents give
# different answers for a state sitting near the boundary.

def _distance(rel: SpatialRelationship, db: Session) -> Tuple[List[str], List[str]]:
    # A missing reference or threshold makes the question unanswerable. Filling
    # either one in with a default would return a confident answer to a
    # question nobody asked, so the query is refused instead.
    if not rel.refs:
        raise ValueError("SPATIAL_DISTANCE query requires a reference in 'refs' "
                         "but none was provided.")
    if rel.distance_km is None:
        raise ValueError("SPATIAL_DISTANCE query requires 'distance_km' "
                         "but it was not provided.")

    raw_city = rel.refs[0]
    dist_m = rel.distance_km * 1000

    coords = _resolve_city_coords(raw_city, db)
    if coords is None:
        # Without the reference point nothing can be measured for any state -
        # every state is unknown here, not "not within range" (Scenario 17's
        # ternary reasoning: a missing input must not collapse into a false).
        pairs, state_unknown = _all_state_geometries(db)
        all_states = [name for name, _ in pairs] + state_unknown
        log.warning("       │ Cannot resolve city %r — distance undetermined for all states", raw_city)
        return [], all_states

    lat, lng = coords
    pairs, unknown = _all_state_geometries(db)
    if not pairs:
        return [], unknown
    cte, geom_params = _geoms_cte(pairs)

    rows = db.execute(
        text(f"""
            WITH {cte}
            SELECT state_name
            FROM geoms
            WHERE ST_DWithin(
                shape::geography,
                ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography,
                :dist
            )
            ORDER BY ST_Distance(
                shape::geography,
                ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography
            )
        """),
        {**geom_params, "lat": lat, "lng": lng, "dist": dist_m},
    ).fetchall()

    log.info("       │ States within %d km of %r : %s",
             int(dist_m / 1000), raw_city, [r[0] for r in rows])
    return [r[0] for r in rows], unknown
