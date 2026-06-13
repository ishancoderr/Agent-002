"""
Build a KQML 'ask' message from gap slots, POST it to Agent 1,
and parse the 'tell' response back into found/missing lists.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from kqml_messaging import MessageFactory, MissingSlot
from kqml_messaging.serializers import JSONSerializer

from .agent_registry import AGENT_REGISTRY
from ..retrieval.gap_detector import GapSlot

log = logging.getLogger("agent2.messaging.kqml_client")

AGENT1_URL = AGENT_REGISTRY.get("Agent-1", "http://localhost:8000")


def send_kqml_ask(gaps: List[GapSlot]) -> Dict[str, Any]:
    missing_slots: List[MissingSlot] = [
        MessageFactory.missing_slot(
            spatial=gap.spatial[0] if len(gap.spatial) == 1 else gap.spatial,
            temporal=gap.temporal,
            attributes=gap.attributes,
        )
        for gap in gaps
    ]

    msg = MessageFactory.ask(
        sender="Agent-2",
        receiver="Agent-1",
        missing_slots=missing_slots,
    )

    payload = JSONSerializer.to_dict(msg)

    total_missing_pts = sum(
        len(gap.temporal) * len(gap.attributes) for gap in gaps
    )
    log.info("       │ Sending KQML ask to Agent-1")
    log.info("       │ Required slots : %d", total_missing_pts)
    log.info("       │ Posting to %s ...", AGENT1_URL)

    response = httpx.post(
        f"{AGENT1_URL}/kqml/receive",
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()

    tell = JSONSerializer.from_dict(response.json())

    found: List[Dict] = []
    still_missing: List[str] = []

    for slot in tell.content.found_slots:
        for record in slot.data:
            flat = record.to_flat_dict()
            found.append(flat)

    for slot in tell.content.missing_slots:
        s = slot.spatial
        states = [s] if isinstance(s, str) else list(s)
        still_missing.extend(states)

    found_pts = sum(
        len(v) for v in [
            {k: v for k, v in r.items() if k not in ("spatial", "year")}
            for r in found
        ]
    )
    log.info("       │ Agent-1 filled : %d data points", found_pts)
    if still_missing:
        log.info("       │ Still missing  : %s", sorted(set(still_missing)))

    return {"found": found, "missing": still_missing}
