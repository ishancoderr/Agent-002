"""
Step 2 — Resolve spatial-relationship queries into concrete state name lists
using PostGIS functions (ST_Intersects, ST_Azimuth, ST_DWithin).
Skipped entirely for DIRECT_LOOKUP queries.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import SessionLocal
from .query_parser import QueryParams, SpatialRelationship

log = logging.getLogger("agent2.pipeline.spatial")


def _fill_missing_state_geometries(db: Session) -> List[str]:
    """
    Find states with NULL geo_shape and sync ALL state geometries from Agent-1
    to ensure both agents use identical reference polygons for consistent
    PostGIS distance results.
    Returns list of state names that were updated.
    """
    rows = db.execute(
        text("SELECT state_name FROM states WHERE geo_shape IS NULL")
    ).fetchall()
    missing = [r[0] for r in rows]

    log.info("       │ Geo-shape check: %d NULL state(s) found", len(missing))

    if not missing:
        return []

    log.info("       │ States with NULL geo_shape: %s — syncing from Agent-1", missing)

    try:
        from kqml_messaging import MessageFactory
        from ..messaging.kqml_geometry_client import send_kqml_geometry_ask

        slots = [
            MessageFactory.missing_geometry_slot(spatial_entity=s, entity_type="state")
            for s in missing
        ]
        resp  = send_kqml_geometry_ask(slots)
        found = resp.get("found", [])

        filled = []
        for fg in found:
            db.execute(
                text("UPDATE states SET geo_shape = ST_GeomFromText(:wkt, 4326) WHERE state_name = :name"),
                {"wkt": fg.geometry, "name": fg.spatial_entity},
            )
            log.info("       │ Filled geo_shape for %s from Agent-1", fg.spatial_entity)
            filled.append(fg.spatial_entity)

        if filled:
            db.commit()
            log.info("       │ Committed %d geometry/geometries to local DB", len(filled))
        return filled

    except Exception as exc:
        log.warning("       │ Could not fetch missing geometries from Agent-1: %s", exc)
        return []

_DIRECTION_SQL = {
    "north_of": "(az <= 45 OR az >= 315)",
    "south_of": "(az BETWEEN 135 AND 225)",
    "east_of":  "(az BETWEEN 45  AND 135)",
    "west_of":  "(az BETWEEN 225 AND 315)",
}

# Maps any German/umlaut or ASCII variant → English DB name (Agent-2 stores English).
# Used as a defensive fallback if the user types German directly.
_CITY_ALIASES: dict = {
    "München":    "Munich",
    "Muenchen":   "Munich",
    "Köln":       "Cologne",
    "Koeln":      "Cologne",
    "Nürnberg":   "Nuremberg",
    "Nuernberg":  "Nuremberg",
    "Düsseldorf": "Dusseldorf",
    "Duesseldorf":"Dusseldorf",
}


def validate_spatial(params: QueryParams) -> QueryParams:
    if params.query_type == "DIRECT_LOOKUP":
        log.info("       │ DIRECT_LOOKUP — spatial resolution skipped")
        return params

    db = SessionLocal()
    try:
        # Fill any NULL geo_shapes from Agent-1 before running PostGIS queries
        filled = _fill_missing_state_geometries(db)
        if filled:
            log.info("       │ Pre-filled geometries from Agent-1: %s", filled)

        log.info("       │ Resolving %s via PostGIS ...", params.query_type)
        if params.query_type == "SPATIAL_ADJACENCY":
            params.spatial = _adjacency(params.spatial_relationship, db)
        elif params.query_type == "SPATIAL_DIRECTION":
            params.spatial = _direction(params.spatial_relationship, db)
        elif params.query_type == "SPATIAL_DISTANCE":
            params.spatial = _distance(params.spatial_relationship, db)
        log.info("       │ Resolved to %d state(s): %s", len(params.spatial), params.spatial)
    finally:
        db.close()

    return params


def _adjacency(rel: SpatialRelationship, db: Session) -> List[str]:
    sets: List[set] = []
    for ref in rel.refs:
        rows = db.execute(
            text("""
                SELECT s2.state_name
                FROM states s1
                JOIN states s2
                  ON ST_Intersects(s1.geo_shape, s2.geo_shape)
                 AND NOT ST_Equals(s1.geo_shape, s2.geo_shape)
                WHERE s1.state_name = :ref
                  AND s2.state_name != :ref
            """),
            {"ref": ref},
        ).fetchall()
        sets.append({r[0] for r in rows})
        log.info("       │ States touching %s: %s", ref, sorted({r[0] for r in rows}))

    if not sets:
        return []
    result = sets[0]
    for s in sets[1:]:
        result &= s
    return sorted(result)


def _direction(rel: SpatialRelationship, db: Session) -> List[str]:
    if not rel.refs:
        raise ValueError("SPATIAL_DIRECTION query requires a reference state in 'refs' but none was provided.")
    ref = rel.refs[0]
    cond = _DIRECTION_SQL.get(rel.type)
    if cond is None:
        valid = ", ".join(_DIRECTION_SQL.keys())
        raise ValueError(f"Unknown direction type {rel.type!r}. Valid types: {valid}.")

    rows = db.execute(
        text(f"""
            WITH azimuths AS (
                SELECT s2.state_name,
                       degrees(ST_Azimuth(
                           ST_Centroid(s1.geo_shape),
                           ST_Centroid(s2.geo_shape))) AS az
                FROM states s1
                JOIN states s2 ON s1.state_name != s2.state_name
                WHERE s1.state_name = :ref
            )
            SELECT state_name FROM azimuths WHERE {cond} ORDER BY az
        """),
        {"ref": ref},
    ).fetchall()

    return [r[0] for r in rows]


def _resolve_city_coords(city: str, db: Session) -> Optional[Tuple[float, float]]:
    """
    Find a city's (lat, lng).
    Resolution order:
      1. Exact match on user-supplied name
      2. Exact match on German alias (e.g. Munich → München)
      3. Ask Agent-1 via KQML
    No fuzzy/partial matching — wrong city is worse than no city.
    """
    candidates = list(dict.fromkeys(filter(None, [
        city,
        _CITY_ALIASES.get(city),
        _CITY_ALIASES.get(city.title()),
    ])))

    for name in candidates:
        row = db.execute(
            text("SELECT lat, lng FROM cities WHERE city_name = :n LIMIT 1"),
            {"n": name},
        ).fetchone()
        if row:
            log.info("       │ City resolved locally: %r → %r  lat=%s lng=%s", city, name, row[0], row[1])
            return row[0], row[1]

    # Not found locally — ask Agent-1
    log.info("       │ City %r not in local DB (tried: %s) — asking Agent-1", city, candidates)
    coords = _fetch_city_coords_from_peer(city, db)
    if coords:
        return coords

    raise ValueError(f"City {city!r} not found locally or in Agent-1 — cannot resolve coordinates.")


def _fetch_city_coords_from_peer(city: str, db: Session) -> Optional[Tuple[float, float]]:
    """
    Ask Agent-1 for a city's centroid WKT via KQML and parse lat/lng from it.
    Does not write to the local DB — coordinates are used directly for this query.
    """
    try:
        from kqml_messaging import MessageFactory
        from ..messaging.kqml_geometry_client import send_kqml_geometry_ask

        slot  = MessageFactory.missing_geometry_slot(spatial_entity=city, entity_type="city")
        resp  = send_kqml_geometry_ask([slot])
        found = resp.get("found", [])

        if not found:
            log.warning("       │ Agent-1 has no geometry for city %r", city)
            return None

        wkt = found[0].geometry  # e.g. "POINT(11.575 48.1375)"
        coords = _parse_point_wkt(wkt)
        if coords is None:
            log.warning("       │ Cannot parse WKT for city %r: %r", city, wkt)
            return None

        lat, lng = coords
        log.info("       │ Agent-1 centroid for %r: lat=%s lng=%s", city, lat, lng)
        return lat, lng

    except Exception as exc:
        log.warning("       │ Peer city fetch failed for %r: %s", city, exc)
        return None


def _parse_point_wkt(wkt: str) -> Optional[Tuple[float, float]]:
    """Parse WKT POINT(lng lat) → (lat, lng). Returns None if not a POINT."""
    import re
    m = re.match(r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", wkt.strip(), re.IGNORECASE)
    if not m:
        return None
    lng, lat = float(m.group(1)), float(m.group(2))
    return lat, lng


def _distance(rel: SpatialRelationship, db: Session) -> List[str]:
    if not rel.refs:
        raise ValueError("SPATIAL_DISTANCE query requires a reference city in 'refs' but none was provided.")
    if rel.distance_km is None:
        raise ValueError("SPATIAL_DISTANCE query requires 'distance_km' but it was not provided.")

    raw_city = rel.refs[0]
    dist_m   = rel.distance_km * 1000

    lat, lng = _resolve_city_coords(raw_city, db)
    rows = db.execute(
        text("""
            SELECT s.state_name
            FROM states s
            WHERE ST_DWithin(
                s.geo_shape::geography,
                ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography,
                :dist
            )
            ORDER BY ST_Distance(
                s.geo_shape::geography,
                ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography
            )
        """),
        {"lat": lat, "lng": lng, "dist": dist_m},
    ).fetchall()

    log.info("       │ States within %d km of %r : %s",
             int(dist_m / 1000), raw_city, [r[0] for r in rows])
    return [r[0] for r in rows]
