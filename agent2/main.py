"""
Agent 2 – FastAPI entry point
  POST /query          ← user submits a natural-language query
  POST /kqml/receive   ← peer agents send KQML 'ask' messages here
  GET  /health         ← liveness probe
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .controller import query_router, kqml_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("agent2")

app = FastAPI(title="Agent 2 – Geospatial Missing Data", version="1.0.0")

app.include_router(query_router)
app.include_router(kqml_router)


_LINE = "_" * 60


@app.middleware("http")
async def log_request_start(request, call_next):
    if request.method == "POST":
        if request.url.path == "/query":
            log.info(_LINE)
            log.info("                        START")
            log.info("          New user query received by Agent 2")
            log.info(_LINE)
        elif request.url.path == "/kqml/receive":
            log.info(_LINE)
            log.info("                        START")
            log.info("        Incoming KQML request received by Agent 2")
            log.info(_LINE)
    return await call_next(request)
