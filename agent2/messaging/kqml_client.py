"""
Compatibility surface for the demographic KQML ask.

Building and sending the message now lives in PeerClient (peer_client.py),
which is the one place that knows how to reach the peer. This module keeps the
original function name so the controller is unaffected by that move.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..retrieval.local_store import GapSlot
from .peer_client import PeerClient

_peer = PeerClient()


def send_kqml_ask(gaps: List[GapSlot], request_id: str = "") -> Dict[str, Any]:
    """Ask the peer for the values this agent could not fill."""
    return _peer.ask_data(gaps, request_id=request_id)
