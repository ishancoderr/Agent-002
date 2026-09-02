"""
Send a KQML geometry ask to Agent-1 and parse the found_geometries reply.
Mirrors kqml_client.py but for the geometry track (scenarios 11-13).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from kqml_messaging import EntityType, MessageFactory, MissingGeometrySlot, FoundGeometrySlot
from kqml_messaging.serializers import JSONSerializer

from .agent_registry import AGENT_REGISTRY

log = logging.getLogger("agent2.messaging.kqml_geometry_client")

AGENT1_URL = AGENT_REGISTRY.get("Agent-1", "http://localhost:8000")


def send_kqml_geometry_ask(
    missing_geometries: List[MissingGeometrySlot],
    request_id: str = "",
) -> Dict[str, Any]:
    """
    Ask Agent-1 for the WKT geometry of one or more named features.

    Returns:
        {
          "found":   [FoundGeometrySlot, ...],
          "missing": [MissingGeometrySlot, ...]   # still not resolved by peer
        }
    """
    msg = MessageFactory.ask(
        sender="Agent-2",
        receiver="Agent-1",
        missing_geometries=missing_geometries,
        reply_with=request_id or None,
    )

    payload = JSONSerializer.to_dict(msg)

    log.info("       │ Sending geometry ask to Agent-1 (%d feature(s))", len(missing_geometries))
    for g in missing_geometries:
        log.info("       │   %s  type=%s", g.spatial_entity, g.entity_type.value)

    response = httpx.post(
        f"{AGENT1_URL}/kqml/receive",
        json=payload,
        timeout=httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=3.0),
    )
    response.raise_for_status()
    tell_payload = response.json()

    tell = JSONSerializer.from_dict(tell_payload)

    found:   List[FoundGeometrySlot]   = tell.content.found_geometries
    missing: List[MissingGeometrySlot] = tell.content.missing_geometries

    log.info("       │ Agent-1 geometry reply: found=%d  missing=%d", len(found), len(missing))
    for f in found:
        log.info("       │   FOUND   %s (%s) srid=%d  %s…", f.spatial_entity, f.entity_type.value, f.srid, f.geometry[:40])
    for m in missing:
        log.info("       │   MISSING %s (%s)", m.spatial_entity, m.entity_type.value)

    return {"found": found, "missing": missing, "ask_message": payload, "tell_message": tell_payload}


def send_kqml_city_buffer_ask(wkt: str, srid: int, exclude: List[str]) -> Dict[str, Any]:
    """
    Scenario 21: ask Agent-1 to test its own city catalogue against a
    constructed buffer zone, since the cities that satisfy the query cannot
    be named in advance. Returns {"found": List[FoundGeometrySlot]}.
    """
    sq = MessageFactory.spatial_query(
        topic="Within", geometry=wkt, target_entity=EntityType.CITY, srid=srid, exclude=exclude,
    )
    msg = MessageFactory.ask_spatial_query(sender="Agent-2", receiver="Agent-1", spatial_query=sq)
    payload = JSONSerializer.to_dict(msg)

    log.info("       │ Sending KQML city-buffer ask to Agent-1 (exclude=%d)", len(exclude))
    log.info("       │ Posting to %s ...", AGENT1_URL)

    response = httpx.post(
        f"{AGENT1_URL}/kqml/receive",
        json=payload,
        timeout=httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=3.0),
    )
    response.raise_for_status()
    tell_payload = response.json()

    tell = JSONSerializer.from_dict(tell_payload)
    found: List[FoundGeometrySlot] = list(tell.content.found_geometries or [])

    for fg in found:
        log.info("       │ City received : %s srid=%s  %.40s…", fg.spatial_entity, fg.srid, fg.geometry)

    return {"found": found, "ask_message": payload, "tell_message": tell_payload}
