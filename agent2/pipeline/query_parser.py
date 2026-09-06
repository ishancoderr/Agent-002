"""
Step 1 — natural language in, QueryParams out.

This module only sequences the pipeline; each step lives in its own class:

    raw query
      -> CleanQuery       clean_query.py   names rewritten to their DB spelling
      -> QueryClassifier  classifier.py    which of the eight categories is it
      -> QueryExtractor   extractor.py     that category's fields, its own prompt
      -> QueryParams      query_params.py  validated, ready for the controller

The category then decides which KQML shape the rest of the pipeline builds:

  DIRECT_LOOKUP / SPATIAL_ADJACENCY / SPATIAL_DIRECTION / SPATIAL_DISTANCE
      (with attributes)          -> :missing-slots / :found-slots
  GEOMETRY_LOOKUP, SPATIAL_OPERATION
      (named-entity geometry)    -> :missing-geometries / :found-geometries
  SPATIAL_RELATIONSHIP_BUFFER
      (target cannot be named)   -> :spatial-query (buffer-and-test)
  UNRELATED
      rejected before extraction — there is nothing to extract

See query_controller.py's dispatch on params.query_type.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai
from dotenv import load_dotenv

from .classifier import QueryClassifier
from .clean_query import CleanQuery
from .extractor import QueryExtractor
from .gazetteer import GERMAN_STATES  # re-exported: query_controller imports it from here
from .query_params import (DEFAULT_YEAR, MAX_YEAR, MIN_YEAR, QueryParams,
                           RELATIONSHIP_TYPES, SpatialRelationship,
                           VALID_ATTRS, VALID_ENTITY_TYPES, VALID_OPERATIONS,
                           VALID_QUERY_TYPES)

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = logging.getLogger("agent2.pipeline.parser")


def _sanitize_entities(raw: Any) -> List[Dict[str, str]]:
    """Validate the entities list from GEOMETRY_LOOKUP extraction.

    Names are already canonical — CleanQuery rewrote them in the query text
    before the model ever saw them — so this only checks the entity type."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("entity_name", "")).strip()
        etype = str(item.get("entity_type", "city")).strip().lower()
        if not name:
            continue
        if etype not in VALID_ENTITY_TYPES:
            etype = "state" if name in GERMAN_STATES else "city"
            log.warning("       | Unknown entity_type for %r - inferred as %s", name, etype)
        result.append({"entity_name": name, "entity_type": etype})
    return result


def _years_from(data: Dict[str, Any]) -> List[int]:
    """The extraction prompts emit `temporal` as the full list of years, so a
    range is already expanded. Values outside the stored range are dropped
    rather than clamped: a query for 1800 should not silently become 1990."""
    raw = data.get("temporal", [DEFAULT_YEAR])
    if not isinstance(raw, list):
        raw = [raw]
    try:
        years = sorted({int(year) for year in raw})
    except (TypeError, ValueError):
        log.warning("       | Invalid temporal values %s - defaulting to [%d]", raw, DEFAULT_YEAR)
        return [DEFAULT_YEAR]

    in_range = [year for year in years if MIN_YEAR <= year <= MAX_YEAR]
    if not in_range:
        log.warning("       | No temporal values in range - defaulting to [%d]", DEFAULT_YEAR)
        return [DEFAULT_YEAR]
    return in_range


def _attributes_from(data: Dict[str, Any], query_type: str) -> List[str]:
    """Only the three stored attributes are accepted. A pure spatial question
    asks for the state list itself, so an empty result there is correct and
    must not be filled in with a default the user never asked for."""
    raw = data.get("attributes", [])
    if not isinstance(raw, list):
        raw = [raw]
    attributes = [attribute for attribute in raw if attribute in VALID_ATTRS]

    dropped = [attribute for attribute in raw if attribute not in VALID_ATTRS]
    if dropped:
        log.warning("       | Dropped unknown attributes: %s", dropped)

    if not attributes and query_type == "DIRECT_LOOKUP":
        log.warning("       | No valid attributes in %s - defaulting to population", raw)
        return ["population"]
    return attributes


def parse_query(query: str) -> Tuple[QueryParams, int]:
    """Returns (QueryParams, tokens_consumed)."""
    original_query = query
    log.info("       | Input (raw)     : %r", query)
    query = CleanQuery(query).cleaned
    log.info("       | Input (cleaned) : %r", query)

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    classifier = QueryClassifier(client)
    extractor = QueryExtractor(client)

    query_type, tokens_classify = classifier.classify(query)
    log.info("       | STEP 1a | Classified as: %s", query_type)

    # ── Unrelated — refuse before spending an extraction call ────────────────
    if query_type == "UNRELATED":
        log.info("       | UNRELATED query - rejecting without an extraction call")
        return QueryParams(
            query_type="UNRELATED", spatial=[], temporal=[], attributes=[],
            raw_query=original_query, classify_tokens=tokens_classify,
            extract_tokens=0, extracted_data={},
        ), tokens_classify

    data, tokens_extract = extractor.extract(query, query_type)
    tokens_consumed = tokens_classify + tokens_extract

    def _trace() -> Dict[str, Any]:
        return {"classify_tokens": tokens_classify,
                "extract_tokens": tokens_consumed - tokens_classify,
                "extracted_data": data}

    # ── Geometry lookup ──────────────────────────────────────────────────────
    if query_type == "GEOMETRY_LOOKUP":
        entities = _sanitize_entities(data.get("entities", []))
        if entities:
            log.info("       | GEOMETRY_LOOKUP : %d entities - %s", len(entities), entities)
            return QueryParams(
                query_type="GEOMETRY_LOOKUP", spatial=[], temporal=[], attributes=[],
                raw_query=original_query, entities=entities, **_trace(),
            ), tokens_consumed
        # Nothing to look a shape up for: fall back to a data lookup instead.
        log.warning("       | GEOMETRY_LOOKUP with empty entities - falling back to DIRECT_LOOKUP")
        query_type = "DIRECT_LOOKUP"
        data, extra = extractor.extract(query, query_type)
        tokens_consumed += extra

    # ── Constructive operation (Scenarios 13-16, 20) ─────────────────────────
    if query_type == "SPATIAL_OPERATION":
        operation = data.get("operation", "")
        raw_spatial = data.get("spatial", [])
        names = [raw_spatial] if isinstance(raw_spatial, str) else list(raw_spatial or [])
        names = [name for name in names if name]

        if operation in VALID_OPERATIONS and len(names) >= 2:
            try:
                distance_km = (float(data["distance_km"])
                               if data.get("distance_km") is not None else None)
            except (TypeError, ValueError):
                distance_km = None
            if operation == "BufferWithin" and distance_km is None:
                distance_km = 100.0
            entity_type = data.get("entity_type", "state") or "state"
            log.info("       | SPATIAL_OPERATION : op=%s spatial=%s entity_type=%s distance_km=%s",
                     operation, names, entity_type, distance_km)
            return QueryParams(
                query_type="SPATIAL_OPERATION", spatial=names, temporal=[], attributes=[],
                raw_query=original_query, operation=operation, entity_type=entity_type,
                distance_km=distance_km, **_trace(),
            ), tokens_consumed
        # An operation needs a verb and two shapes; without them there is
        # nothing to construct, so answer it as a data lookup.
        log.warning("       | SPATIAL_OPERATION malformed (operation=%r, spatial=%s) - "
                    "falling back to DIRECT_LOOKUP", operation, names)
        query_type = "DIRECT_LOOKUP"
        data, extra = extractor.extract(query, query_type)
        tokens_consumed += extra

    # ── Buffer over an open target type (Scenario 21) ────────────────────────
    if query_type == "SPATIAL_RELATIONSHIP_BUFFER":
        raw_spatial = data.get("spatial", [])
        ref_cities = [raw_spatial] if isinstance(raw_spatial, str) else list(raw_spatial or [])
        ref_cities = [city for city in ref_cities if city]

        if ref_cities:
            try:
                distance_km = float(data.get("distance_km", 100))
            except (TypeError, ValueError):
                distance_km = 100.0
            log.info("       | SPATIAL_RELATIONSHIP_BUFFER : ref=%s distance_km=%s target=%s",
                     ref_cities, distance_km, data.get("target_entity", "city"))
            return QueryParams(
                query_type="SPATIAL_RELATIONSHIP_BUFFER", spatial=ref_cities,
                temporal=[], attributes=[], raw_query=original_query,
                distance_km=distance_km,
                target_entity=data.get("target_entity", "city"), **_trace(),
            ), tokens_consumed
        # No reference to build a zone around.
        log.warning("       | SPATIAL_RELATIONSHIP_BUFFER with no reference city - "
                    "falling back to DIRECT_LOOKUP")
        query_type = "DIRECT_LOOKUP"
        data, extra = extractor.extract(query, query_type)
        tokens_consumed += extra

    # ── Demographic lookup, and the relationships that also ask for data ─────
    temporal = _years_from(data)
    attributes = _attributes_from(data, query_type)
    if not attributes:
        log.info("       | Pure spatial question - no data attributes requested")

    spatial = data.get("spatial", "all")
    if spatial == "all" or spatial is None or not isinstance(spatial, (str, list)):
        spatial = ["all"]
    elif isinstance(spatial, str):
        spatial = [spatial]

    relationship = None
    raw_relationship = data.get("spatial_relationship")
    if isinstance(raw_relationship, dict):
        refs = raw_relationship.get("refs", [])
        if not isinstance(refs, list):
            refs = [refs] if refs else []
        subject = raw_relationship.get("subject")
        relationship = SpatialRelationship(
            type=raw_relationship.get("type", ""),
            refs=refs,
            distance_km=raw_relationship.get("distance_km"),
            subject=subject.strip() if isinstance(subject, str) and subject.strip() else None,
        )

    if relationship is not None:
        # The state list is resolved from the relationship, not from `spatial`.
        spatial = ["all"]
    elif query_type in RELATIONSHIP_TYPES:
        log.warning("       | %s produced no spatial_relationship - the resolver "
                    "will have nothing to work from", query_type)

    log.info("       | Query type  : %s", query_type)
    log.info("       | Temporal    : %s (%d years)", temporal, len(temporal))
    log.info("       | Attributes  : %s", attributes)
    log.info("       | Spatial     : %s", spatial)
    log.info("       | Tokens      : classify=%d  extract=%d  total=%d",
             tokens_classify, tokens_consumed - tokens_classify, tokens_consumed)

    return QueryParams(
        query_type=query_type, spatial=spatial, temporal=temporal,
        attributes=attributes, spatial_relationship=relationship,
        raw_query=original_query, **_trace(),
    ), tokens_consumed
