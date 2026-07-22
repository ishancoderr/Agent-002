"""
Resolve geometry requests against the local cities/states catalog.

Used when a peer agent sends missing_geometries in a KQML ask.
Returns WKT + srid for found features; passes through unresolved ones as still-missing.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from sqlalchemy import text

from kqml_messaging import MissingGeometrySlot, FoundGeometrySlot

from ..database import SessionLocal

log = logging.getLogger("agent2.retrieval.geometry_resolver")


def resolve_geometries(
    requests: List[MissingGeometrySlot],
) -> Tuple[List[FoundGeometrySlot], List[MissingGeometrySlot]]:
    """
    Look up each requested geometry from the local catalog.

    Returns:
        found   — FoundGeometrySlot list (WKT + srid=4326)
        missing — MissingGeometrySlot list not found locally
    """
    found:   List[FoundGeometrySlot]   = []
    missing: List[MissingGeometrySlot] = []

    db = SessionLocal()
    try:
        for req in requests:
            wkt = _lookup(req.spatial_entity, req.entity_type, db)
            if wkt:
                log.info("       │ GEOM FOUND  : %s (%s) → %s…", req.spatial_entity, req.entity_type, wkt[:50])
                found.append(FoundGeometrySlot(
                    spatial_entity=req.spatial_entity,
                    entity_type=req.entity_type,
                    geometry=wkt,
                    srid=4326,
                ))
            else:
                log.info("       │ GEOM MISSING: %s (%s) — not in local catalog", req.spatial_entity, req.entity_type)
                missing.append(req)
    finally:
        db.close()

    return found, missing


_CITY_ALIASES: dict = {
    "München": "Munich",
    "Muenchen": "Munich",
    "Cologne": "Köln",
    "Köln": "Cologne",
    "Koeln": "Cologne",
    "Nürnberg": "Nuremberg",
    "Nuernberg": "Nuremberg",
    "Düsseldorf": "Dusseldorf",
    "Duesseldorf": "Dusseldorf",
}


def _lookup(entity: str, entity_type: str, db) -> str | None:
    if entity_type == "city":
        candidates = list(dict.fromkeys(filter(None, [
            entity,
            _CITY_ALIASES.get(entity),
            _CITY_ALIASES.get(entity.title()),
        ])))
        for name in candidates:
            row = db.execute(
                text("SELECT ST_AsText(centroid) FROM cities WHERE city_name = :n LIMIT 1"),
                {"n": name},
            ).fetchone()
            if row is not None:
                log.info("       │ City resolved as %r", name)
                return row[0]
        # partial ILIKE fallback
        row = db.execute(
            text("SELECT ST_AsText(centroid) FROM cities WHERE city_name ILIKE :n ORDER BY LENGTH(city_name) LIMIT 1"),
            {"n": f"%{entity}%"},
        ).fetchone()
        return row[0] if row is not None else None

    elif entity_type == "state":
        row = db.execute(
            text("SELECT ST_AsText(geo_shape) FROM states WHERE state_name = :n LIMIT 1"),
            {"n": entity},
        ).fetchone()
        return row[0] if row is not None else None

    else:
        log.warning("       │ Unknown entity_type %r for %r — skipping", entity_type, entity)
        return None
