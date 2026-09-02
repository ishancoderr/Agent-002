"""
Writes structured evaluation metrics to evaluation_metrics.log (project root).
Uses its own FileHandler so nothing leaks into the uvicorn console.

Every request — whatever category it lands in (DIRECT_LOOKUP, a spatial
relationship, GEOMETRY_LOOKUP, SPATIAL_OPERATION, SPATIAL_RELATIONSHIP_BUFFER,
or UNRELATED) — logs the same five sections, so any request can be replayed
step by step from this file alone:
  1. CLASSIFICATION : which category stage 1 (GPT-4) chose, and its token cost
  2. EXTRACTION      : the raw fields stage 2 (gpt-4o-mini) pulled out
  3. LOCAL RESOLUTION: what this agent found in its own database, unassisted
  4. KQML EXCHANGE    : every ask/tell round-trip with the peer, verbatim
  5. RESULT / TIMING / TOKENS : the final outcome and cost
"""
from __future__ import annotations

import json
import logging
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


def _fmt_classification(m: Dict[str, Any]) -> str:
    return (
        f"{_SEP}\n"
        f"STEP 1 — CLASSIFICATION (GPT-4, stage 1)\n"
        f"  Category chosen      : {m.get('query_type', 'n/a')}\n"
        f"  Tokens used          : {m.get('classify_tokens', 0)}\n"
    )


def _fmt_extraction(m: Dict[str, Any]) -> str:
    extracted = m.get("extracted_data")
    parts = [
        f"{_SEP}\n"
        f"STEP 2 — EXTRACTION (gpt-4o-mini, template = {m.get('query_type', 'n/a')})\n"
        f"  Tokens used          : {m.get('extract_tokens', 0)}\n"
    ]
    if m.get("query_type") == "UNRELATED":
        parts.append("  (skipped — UNRELATED is rejected right after classification,\n"
                      "   no extraction call is made)\n")
    elif extracted:
        parts.append("  Raw fields extracted :\n")
        parts.append(
            "\n".join(f"    {ln}" for ln in json.dumps(extracted, indent=2, ensure_ascii=False).splitlines())
            + "\n"
        )
    return "".join(parts)


def _fmt_local_resolution(m: Dict[str, Any]) -> str:
    local = m.get("local_resolution")
    if not local:
        return ""
    parts = [f"{_SEP}\nSTEP 3 — LOCAL RESOLUTION (this agent's own database, before asking the peer)\n"]
    for k, v in local.items():
        if k == "sql_queries":
            continue  # rendered separately below, as its own subsection
        parts.append(f"  {k:<21}: {v}\n")
    queries = local.get("sql_queries")
    if queries:
        parts.append(f"  SQL queries run against this agent's own database ({len(queries)}):\n")
        for i, q in enumerate(queries, 1):
            parts.append(f"    [{i}] {q}\n")
    return "".join(parts)


def _fmt_kqml(m: Dict[str, Any]) -> str:
    exchanges = m.get("kqml_exchanges") or []
    if not exchanges:
        return f"{_SEP}\nSTEP 4 — KQML EXCHANGE : none (no gaps -- answered entirely from the local partition)\n"
    parts = [f"{_SEP}\nSTEP 4 — KQML EXCHANGE ({len(exchanges)} round-trip(s) with the peer agent)\n"]
    for i, ex in enumerate(exchanges, 1):
        parts.append(f"--- ask #{i} " + "-" * 40 + "\n")
        parts.append(json.dumps(ex.get("ask"), indent=2, ensure_ascii=False) + "\n")
        parts.append(f"--- tell #{i} " + "-" * 39 + "\n")
        parts.append(json.dumps(ex.get("tell"), indent=2, ensure_ascii=False) + "\n")
    return "".join(parts)


def log_evaluation_metrics(metrics: Dict[str, Any]) -> None:
    """Write one evaluation block to evaluation_metrics.log."""
    m = metrics
    block = (
        f"\n{_LINE}\n"
        f"REQUEST ID : {m.get('request_id', 'n/a')}\n"
        f"TIMESTAMP  : {m.get('timestamp', 'n/a')}\n"
        f"{_SEP}\n"
        f"Query      : {m.get('query', '')}\n"
        f"{_fmt_classification(m)}"
        f"{_fmt_extraction(m)}"
        f"{_fmt_local_resolution(m)}"
        f"{_fmt_kqml(m)}"
        f"{_SEP}\n"
        f"STEP 5 — RESULT\n"
        f"  Status                : {m.get('status', 'n/a')}\n"
        f"{_SEP}\n"
        f"TIMING\n"
        f"  Phase 1 (local)       : {m.get('phase1_ms', 0):>6.0f} ms   (parse + spatial + DB)\n"
        f"  Phase 2 (KQML)        : {m.get('phase2_ms', 0):>6.0f} ms   (ask -> tell round-trip)\n"
        f"  Phase 3 (merge)       : {m.get('phase3_ms', 0):>6.0f} ms\n"
        f"  Total                 : {m.get('total_ms', 0):>6.0f} ms\n"
        f"{_SEP}\n"
        f"TOKENS\n"
        f"  Agent 2               : {m.get('tokens_agent2', 0):>6}\n"
        f"  Agent 1               : {m.get('tokens_agent1', 0):>6}\n"
        f"  Total                 : {m.get('tokens_total', 0):>6}\n"
        f"{_SEP}\n"
        f"DATA QUALITY\n"
        f"  Records total         : {m.get('total_records', 0):>6}\n"
        f"  Data points total     : {m.get('total_data_points', 0):>6}\n"
        f"  Data points present   : {m.get('present_data_points', 0):>6}\n"
        f"  Data points missing   : {m.get('missing_data_points', 0):>6}\n"
        f"  Complete records      : {m.get('complete_records', 0):>6}\n"
        f"  Partial records       : {m.get('partial_records', 0):>6}\n"
        f"  Empty records         : {m.get('empty_records', 0):>6}\n"
        f"{_LINE}"
    )
    _metrics_log.info(block)
