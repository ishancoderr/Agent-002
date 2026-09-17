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

# The minimum version of the shared KQML library this agent's code needs.
# Raise it whenever this agent starts using something the library only gained
# in a newer release.
REQUIRED_KQML_VERSION = "2.3.0"


def _as_numbers(version: str) -> tuple:
    """Compare only the numeric head of a version, so a "2.3.0.dev1" or a
    "2.3.0rc1" still counts as 2.3.0 rather than failing to parse."""
    numbers = []
    for part in version.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def _check_library_version() -> None:
    """Refuse to start against a shared library older than this agent needs.

    Both agents install kqml-messaging from a git URL, which cannot carry a
    version specifier - so requirements.txt cannot express this, and a stale
    copy already sitting in site-packages would satisfy the install either way.
    Without this check the mismatch surfaces at the first query as a bare
    ImportError for whatever name is missing, which says nothing about the
    cause. This says it plainly, at startup, once.
    """
    import kqml_messaging

    installed = getattr(kqml_messaging, "__version__", "0")
    if _as_numbers(installed) < _as_numbers(REQUIRED_KQML_VERSION):
        raise RuntimeError(
            "kqml-messaging {} is installed, but this agent needs {} or newer.\n"
            "  Loaded from: {}\n\n"
            "  Reinstall the shared library:\n"
            "    pip install -e <path to the kqml-geo checkout>\n"
            "  or, if you install it from git:\n"
            "    pip install --force-reinstall "
            "'kqml-messaging @ git+https://github.com/ishancoderr/kqml-geo.git'"
            .format(installed, REQUIRED_KQML_VERSION, kqml_messaging.__file__)
        )
    log.info("kqml-messaging %s (needs >= %s)", installed, REQUIRED_KQML_VERSION)


_check_library_version()
