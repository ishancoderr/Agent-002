"""
Step 2 — Resolve spatial-relationship queries into concrete state name lists
using PostGIS functions (ST_Touches, ST_Azimuth, ST_DWithin).
Skipped entirely for DIRECT_LOOKUP queries.
"""
from __future__ import annotations

import logging
from typing import List

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import SessionLocal
from .query_parser import QueryParams, SpatialRelationship

log = logging.getLogger("agent2.pipeline.spatial")

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
                JOIN states s2 ON ST_Touches(s1.geo_shape, s2.geo_shape)
                WHERE s1.state_name = :ref AND s2.state_name != :ref
            """),
            {"ref": ref},
        ).fetchall()
        sets.append({r[0] for r in rows})

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


def _distance(rel: SpatialRelationship, db: Session) -> List[str]:
    if not rel.refs:
        raise ValueError("SPATIAL_DISTANCE query requires a reference city in 'refs' but none was provided.")
    if rel.distance_km is None:
        raise ValueError("SPATIAL_DISTANCE query requires 'distance_km' but it was not provided.")
    city   = rel.refs[0]
    dist_m = rel.distance_km * 1000

    rows = db.execute(
        text("""
            SELECT s.state_name
            FROM states s
            WHERE ST_DWithin(
                s.geo_shape::geography,
                (SELECT centroid::geography FROM cities WHERE city_name = :city),
                :dist
            )
            ORDER BY ST_Distance(
                s.geo_shape::geography,
                (SELECT centroid::geography FROM cities WHERE city_name = :city)
            )
        """),
        {"city": city, "dist": dist_m},
    ).fetchall()

    return [r[0] for r in rows]
