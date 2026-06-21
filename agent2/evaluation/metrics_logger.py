"""
Writes structured evaluation metrics to evaluation_metrics.log (project root).
Uses its own FileHandler so nothing leaks into the uvicorn console.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

_LOG_PATH = Path(__file__).parent.parent.parent / "evaluation_metrics.log"

_metrics_log = logging.getLogger("agent2.evaluation.metrics")
_metrics_log.propagate = False  # no console output

if not _metrics_log.handlers:
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _metrics_log.addHandler(_fh)
    _metrics_log.setLevel(logging.INFO)

_LINE = "=" * 60
_SEP  = "-" * 60


def log_evaluation_metrics(metrics: Dict[str, Any]) -> None:
    """Write one evaluation block to evaluation_metrics.log."""
    m = metrics
    block = (
        f"\n{_LINE}\n"
        f"REQUEST ID : {m.get('request_id', 'n/a')}\n"
        f"TIMESTAMP  : {m.get('timestamp', 'n/a')}\n"
        f"{_SEP}\n"
        f"Query      : {m.get('query', '')}\n"
        f"{_SEP}\n"
        f"TIMING\n"
        f"  Phase 1 (local)       : {m.get('phase1_ms', 0):>6.0f} ms   (parse + spatial + DB)\n"
        f"  Phase 2 (KQML)        : {m.get('phase2_ms', 0):>6.0f} ms   (ask → tell round-trip)\n"
        f"  Phase 3 (merge)       : {m.get('phase3_ms', 0):>6.0f} ms\n"
        f"  Total                 : {m.get('total_ms', 0):>6.0f} ms\n"
        f"{_SEP}\n"
        f"TOKENS\n"
        f"  Agent 2               : {m.get('tokens_agent2', 0):>6}\n"
        f"  Agent 1               : {m.get('tokens_agent1', 0):>6}\n"
        f"  Total                 : {m.get('tokens_total', 0):>6}\n"
        f"{_LINE}"
    )
    _metrics_log.info(block)
