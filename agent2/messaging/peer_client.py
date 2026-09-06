"""
The last step of the pipeline: ask the other agent for what this one lacks.

PeerClient is the single place that knows how to reach the peer. Everything it
sends is a KQML message built by the shared library, and the three kinds of ask
correspond to the three things that can be missing:

    demographics    :missing-slots      values this agent does not hold
    geometry        :missing-geometries shapes this agent cannot supply
    spatial query   :spatial-query      a constructed zone the peer must test
                                        against its own catalogue, used only
                                        when the targets cannot be named

The first two name what they want. The third cannot — the cities that satisfy
it are the answer, not the input — which is why a shape, rather than a name,
crosses the wire in that one case.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from kqml_messaging import (EntityType, FoundGeometrySlot, MessageFactory,
                            MissingGeometrySlot, MissingSlot)
from kqml_messaging.serializers import JSONSerializer

from .agent_registry import AGENT_REGISTRY
from ..retrieval.local_store import GapSlot

log = logging.getLogger("agent2.messaging.peer_client")

# A call is two LLM-free database lookups at the peer, but the round trip still
# has to tolerate a slow peer without hanging this agent indefinitely.
TIMEOUT = httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=3.0)


class PeerClient:
    """Sends this agent's gaps to the peer and reads the reply."""

    def __init__(self, name: str = "Agent-1", url: str | None = None, sender: str = "Agent-2"):
        self.name = name
        self.url = url or AGENT_REGISTRY.get(name, "http://localhost:8000")
        self.sender = sender

    def _post(self, payload: dict) -> dict:
        log.info("       | Posting to %s ...", self.url)
        response = httpx.post(f"{self.url}/kqml/receive", json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()

    def ask_data(self, gaps: List[GapSlot], request_id: str = "") -> Dict[str, Any]:
        """Ask for values. One entry per gap block, because states missing
        different years describe different blocks and must not be merged.

        `request_id`, when given, becomes the KQML message's `:reply-with`, so
        the exchange can be tied back to the user query that triggered it in
        the evaluation log. Left blank, one is generated for the message alone."""
        message = MessageFactory.ask(
            sender=self.sender, receiver=self.name,
            missing_slots=[
                MessageFactory.missing_slot(
                    spatial=gap.spatial[0] if len(gap.spatial) == 1 else gap.spatial,
                    temporal=gap.temporal,
                    attributes=gap.attributes,
                )
                for gap in gaps
            ],
            reply_with=request_id or None,
        )
        ask_payload = JSONSerializer.to_dict(message)
        log.info("       | Sending KQML ask to %s (%d slot(s))", self.name, len(gaps))

        tell_payload = self._post(ask_payload)
        tell = JSONSerializer.from_dict(tell_payload)

        found: List[Dict] = []
        for slot in tell.content.found_slots:
            found.extend(record.to_flat_dict() for record in slot.data)

        still_missing: List[str] = []
        for slot in tell.content.missing_slots:
            spatial = slot.spatial
            still_missing.extend([spatial] if isinstance(spatial, str) else list(spatial))
        still_missing = list(dict.fromkeys(still_missing))

        tokens = 0
        if getattr(tell, "metadata", None) is not None:
            tokens = getattr(tell.metadata, "token_usage", 0) or 0

        log.info("       | %s filled %d record(s), tokens=%d", self.name, len(found), tokens)
        if still_missing:
            log.info("       | Still missing : %s", sorted(set(still_missing)))

        return {"found": found, "missing": still_missing, "tokens_agent2": tokens,
                "ask_message": ask_payload, "tell_message": tell_payload}

    def ask_geometry(self, slots: List[MissingGeometrySlot]) -> Dict[str, Any]:
        """Ask for named shapes. The request says nothing about what the shape
        is wanted for — the peer supplies geometry, and what this agent does
        with it is not the peer's concern."""
        message = MessageFactory.ask(sender=self.sender, receiver=self.name,
                                     missing_geometries=slots)
        ask_payload = JSONSerializer.to_dict(message)
        log.info("       | Sending KQML geometry ask to %s (%d slot(s))", self.name, len(slots))

        tell_payload = self._post(ask_payload)
        tell = JSONSerializer.from_dict(tell_payload)

        found: List[FoundGeometrySlot] = list(tell.content.found_geometries or [])
        missing: List[MissingGeometrySlot] = list(tell.content.missing_geometries or [])
        for geometry in found:
            log.info("       | Geometry received : %s (%s) srid=%s  %.60s...",
                     geometry.spatial_entity, geometry.entity_type.value,
                     geometry.srid, geometry.geometry)
        if missing:
            log.info("       | Still missing geometries: %s",
                     [(m.spatial_entity, m.entity_type.value) for m in missing])

        return {"found": found, "missing": missing,
                "ask_message": ask_payload, "tell_message": tell_payload}

    def ask_within_zone(self, wkt: str, srid: int, exclude: List[str]) -> Dict[str, Any]:
        """Send a constructed zone for the peer to test its own catalogue against.

        The zone carries no entity name, because a shape built here has no row
        in either store. The exclusion list stops a city both agents hold from
        coming back twice."""
        spatial_query = MessageFactory.spatial_query(
            topic="Within", geometry=wkt, target_entity=EntityType.CITY,
            srid=srid, exclude=exclude,
        )
        message = MessageFactory.ask_spatial_query(
            sender=self.sender, receiver=self.name, spatial_query=spatial_query
        )
        ask_payload = JSONSerializer.to_dict(message)
        log.info("       | Sending KQML zone test to %s (exclude=%d)", self.name, len(exclude))

        tell_payload = self._post(ask_payload)
        tell = JSONSerializer.from_dict(tell_payload)
        found: List[FoundGeometrySlot] = list(tell.content.found_geometries or [])
        log.info("       | %s returned %d matching cit(y/ies)", self.name, len(found))

        return {"found": found, "ask_message": ask_payload, "tell_message": tell_payload}
