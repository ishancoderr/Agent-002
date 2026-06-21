"""
POST /kqml/receive  — handles incoming KQML 'ask' messages from peer agents
                      (supports bidirectional Agent-1 → Agent-2 queries)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from kqml_messaging import MissingSlot

from ..retrieval import execute_local_lookup_from_slots

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
    log.info("KQML   │ Incoming ask from %s  req=%s", msg.sender, msg.reply_with)
    log.info("       │ Missing slots: %d", len(msg.content.get("missing_slots", [])))

    found_slots   = []
    missing_slots = []

    for i, slot_raw in enumerate(msg.content.get("missing_slots", []), 1):
        slot = MissingSlot(**slot_raw)
        log.info("       │ Slot %d: spatial=%s  temporal=%s  attrs=%s",
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

    log.info("KQML   │ Reply: found_slots=%d  missing_slots=%d",
             len(found_slots), len(missing_slots))
    log.info(SEPARATOR)

    return {
        "sender":      "Agent-2",
        "receiver":    msg.sender,
        "in_reply_to": msg.reply_with,
        "language":    "GeoSQL",
        "ontology":    "German-Geostats-v1",
        "content":     {"found_slots": found_slots, "missing_slots": missing_slots, "tokens_consumed": 0},
    }
