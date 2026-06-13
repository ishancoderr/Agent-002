from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

AGENT_REGISTRY: Dict[str, str] = {
    name: url
    for name, url in {
        "Agent-1": os.getenv("AGENT1_URL", "http://localhost:8000"),
        "Agent-3": os.getenv("AGENT3_URL", ""),
        "Agent-4": os.getenv("AGENT4_URL", ""),
    }.items()
    if url
}
