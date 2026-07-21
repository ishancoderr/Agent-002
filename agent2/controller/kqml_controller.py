"""
POST /kqml/receive  — handles incoming KQML 'ask' messages from peer agents
                      (supports bidirectional Agent-1 → Agent-2 queries)
                      Handles both data slots and geometry slots (scenarios 11-13).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from kqml_messaging import MissingSlot, MissingGeometrySlot

from ..retrieval import execute_local_lookup_from_slots
from ..retrieval.geometry_resolver import resolve_geometries

log = logging.getLogger("agent2.controller.kqml")
router = APIRouter()

SEPARATOR = "─" * 60


class KQMLMessage(BaseModel):
    sender: str
    receiver: str
    reply_with: Optional[str] = None
    in_reply_to: Optional[str] = None
    language: str = "GeoSQL"
    ontology: str = "German-Geostats-v1"
    content: Dict[str, Any]


@router.post("/kqml/receive")
def receive_kqml(msg: KQMLMessage):
    log.info(SEPARATOR)
    log.info("                       START")
    log.info("       Incoming KQML request received by Agent 2")
    log.info(SEPARATOR)
    log.info("KQML   │ From: %s  req=%s", msg.sender, msg.reply_with)
    log.info("       │ Missing slots    : %d", len(msg.content.get("missing_slots", [])))
    log.info("       │ Missing geometries: %d", len(msg.content.get("missing_geometries", [])))

    # ── Data slots ────────────────────────────────────────────────────────────
    found_slots   = []
    missing_slots = []

    for i, slot_raw in enumerate(msg.content.get("missing_slots", []), 1):
        slot = MissingSlot(**slot_raw)
        log.info("       │ Data slot %d: spatial=%s  temporal=%s  attrs=%s",
                 i, slot.spatial, slot.temporal, slot.attributes)

        result = execute_local_lookup_from_slots(slot)
        log.info("       │   → found=%d  missing=%s",
                 len(result["found"]), result["missing"] or "none")

        if result["found"]:
            found_slots.append({
                "spatial":    slot.spatial,
                "temporal":   slot.temporal,
                "attributes": slot.attributes,
                "data":       result["found"],
            })
        if result["missing"]:
            missing_slots.append({
                "spatial":    result["missing"],
                "temporal":   slot.temporal,
                "attributes": slot.attributes,
            })

    # ── Geometry slots (scenarios 11-13) ─────────────────────────────────────
    found_geometries   = []
    missing_geometries = []

    raw_geom_requests = msg.content.get("missing_geometries", [])
    if raw_geom_requests:
        geom_requests = [MissingGeometrySlot(**g) for g in raw_geom_requests]
        found_geom, missing_geom = resolve_geometries(geom_requests)

        found_geometries = [
            {
                "spatial_entity": fg.spatial_entity,
                "entity_type":    fg.entity_type,
                "geometry":       fg.geometry,
                "srid":           fg.srid,
            }
            for fg in found_geom
        ]
        missing_geometries = [
            {"spatial_entity": mg.spatial_entity, "entity_type": mg.entity_type}
            for mg in missing_geom
        ]

    log.info("KQML   │ Reply: found_slots=%d  missing_slots=%d  found_geom=%d  missing_geom=%d",
             len(found_slots), len(missing_slots), len(found_geometries), len(missing_geometries))
    log.info(SEPARATOR)

    return {
        "performative": "tell",
        "sender":       "Agent-2",
        "receiver":     msg.sender,
        "in_reply_to":  msg.reply_with,
        "language":     "GeoSQL",
        "ontology":     "German-Geostats-v1",
        "metadata":     {"token_usage": 0},
        "content": {
            "found_slots":        found_slots,
            "missing_slots":      missing_slots,
            "found_geometries":   found_geometries,
            "missing_geometries": missing_geometries,
        },
    }
