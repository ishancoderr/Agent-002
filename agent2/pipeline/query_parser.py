"""
Step 1 — natural language in, QueryParams out.

This module only sequences the pipeline; each step lives in its own class:

    raw query
      -> CleanQuery       clean_query.py         names rewritten to their DB spelling
      -> QueryClassifier  query_classifier.py    which of the eight categories is it
      -> QueryExtractor   query_extractor.py     that category's fields, its own prompt
      -> QueryParams      query_params.py        validated, ready for the controller

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

from .query_classifier import QueryClassifier
from .clean_query import CleanQuery
from .query_extractor import QueryExtractor
from .gazetteer import GERMAN_STATES  # re-exported: query_controller imports it from here
from .query_params import (QueryParams,
                           DEFAULT_ENTITY_TYPE, RELATIONSHIP_TYPES, SpatialRelationship,
                           VALID_ATTRS, VALID_ENTITY_TYPES, VALID_OPERATIONS,
                           VALID_QUERY_TYPES, default_attribute_for)

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
    range is already expanded.

    A year the model actually extracted is used exactly as asked, never
    rewritten to a different one. "What was Bayern's population in 1985?" is
    a well-formed question even though neither agent's partition reaches that
    far back; the honest answer is that nobody holds it, discovered the same
    way any other gap is (the SQL lookup returns no row, the peer is asked,
    the peer has none either, status resolves to not_found - Scenario 8).

    An empty result here is not an error to paper over with a guess - it
    means the query gave no year and implied none either (the extraction
    prompt already resolves "now"/"this year" to a real number, so what
    reaches this function empty is genuinely unspecified). The caller decides
    whether that empty list is fatal: harmless for a pure spatial question,
    but an incomplete request wherever attributes were actually asked for -
    see the "no year, but data was requested" check in parse_query()."""
    raw = data.get("temporal", [])
    if not isinstance(raw, list):
        raw = [raw]
    try:
        years = sorted({int(year) for year in raw if year is not None})
    except (TypeError, ValueError):
        log.warning("       | Non-numeric temporal values %s - treating as unspecified", raw)
        return []
    return [year for year in years if 0 < year < 10000]


def _attributes_from(data: Dict[str, Any], query_type: str, entity_type: str) -> List[str]:
    """Every declared attribute column, across every entity, is accepted —
    not just the three state ones (see VALID_ATTRS). A pure spatial question
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
        default_attr = default_attribute_for(entity_type)
        log.warning("       | No valid attributes in %s - defaulting to %s", raw, default_attr)
        return [default_attr]
    return attributes


def parse_query(query: str, model: str | None = None) -> Tuple[QueryParams, int]:
    """Returns (QueryParams, tokens_consumed).

    `model` overrides both the classify and extract stages for this one call
    (the caller — see query_controller.py's UserQuery.model — asked for a
    specific model). Leave it None to use each stage's own default, which is
    CLASSIFY_MODEL/EXTRACT_MODEL — themselves overridable per deployment via
    the env vars of the same name, not a per-request choice.

    This function only sequences clean -> classify -> extract -> shape; the
    shaping step (turning the raw extracted fields into a validated
    QueryParams, including the malformed-extraction fallbacks) is
    QueryParamsBuilder.build() below, kept separate so pipeline_main.py —
    which drives clean/classify/extract itself, one call at a time, for
    visibility — can shape its own already-extracted `data` the exact same
    way this function does, without a second, redundant classify+extract
    round trip."""
    original_query = query
    log.info("       | Input (raw)     : %r", query)
    query = CleanQuery(query).cleaned
    log.info("       | Input (cleaned) : %r", query)

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    classifier = QueryClassifier(client, model=model) if model else QueryClassifier(client)
    extractor = QueryExtractor(client, model=model) if model else QueryExtractor(client)

    query_type, tokens_classify = classifier.classify(query)
    log.info("       | STEP 1a | Classified as: %s", query_type)

    # ── Unrelated — refuse before spending an extraction call ────────────────
    if query_type == "UNRELATED":
        log.info("       | UNRELATED query - rejecting without an extraction call")
        return QueryParams(
            query_type="UNRELATED", spatial=[], temporal=[], attributes=[],
            raw_query=original_query, classify_tokens=tokens_classify,
            extract_tokens=0, extracted_data={}, classify_model=classifier.model,
        ), tokens_classify

    data, tokens_extract = extractor.extract(query, query_type)
    tokens_consumed = tokens_classify + tokens_extract

    return QueryParamsBuilder(extractor).build(
        cleaned_query=query, query_type=query_type, data=data,
        original_query=original_query, tokens_classify=tokens_classify,
        tokens_consumed=tokens_consumed, classify_model=classifier.model,
    )


class QueryParamsBuilder:
    """Turns one extract() call's raw fields into a validated QueryParams.

    Split out of parse_query() so a caller that already has `data` in hand
    (parse_query() itself, or pipeline_main.py driving QueryExtractor
    directly) reaches the same result through the same code, including the
    malformed-extraction fallbacks in build(), which call `extractor` again."""

    def __init__(self, extractor: QueryExtractor):
        self.extractor = extractor

    def build(
        self, cleaned_query: str, query_type: str, data: Dict[str, Any],
        original_query: str, tokens_classify: int, tokens_consumed: int,
        classify_model: str,
    ) -> Tuple[QueryParams, int]:
        extractor = self.extractor

        def _trace() -> Dict[str, Any]:
            """Bundle this call's token counts, raw extracted fields, and the
            actual model each stage used into the kwargs every early-return
            QueryParams(...) below needs to supply."""
            return {"classify_tokens": tokens_classify,
                    "extract_tokens": tokens_consumed - tokens_classify,
                    "extracted_data": data,
                    "classify_model": classify_model,
                    "extract_model": extractor.model}

        query = cleaned_query

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
                entity_type = data.get("entity_type") or DEFAULT_ENTITY_TYPE
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
        # Only DIRECT_LOOKUP's own extraction prompt (config/prompts/
        # direct_lookup.yaml) asks the model for entity_type at all — the
        # relationship types are inherently about states specifically (a
        # fact about what those categories mean, not something extracted),
        # so entity_type stays unset for them and agent2/retrieval/
        # local_store.py never looks at it there either. Validated against
        # VALID_ENTITY_TYPES the same way GEOMETRY_LOOKUP's own entities are
        # (_sanitize_entities above) — an untrusted model output, not
        # assumed correct just because it parsed as a string. Determined
        # before _attributes_from() so a query with no valid attribute
        # named defaults to *this* entity's own first attribute, not a
        # blind "population".
        entity_type = None
        if query_type == "DIRECT_LOOKUP":
            raw_entity_type = data.get("entity_type")
            if raw_entity_type in VALID_ENTITY_TYPES:
                entity_type = raw_entity_type
            elif raw_entity_type:
                log.warning("       | Unknown entity_type %r for DIRECT_LOOKUP - "
                            "falling back to the default", raw_entity_type)

        temporal = _years_from(data)
        attributes = _attributes_from(data, query_type, entity_type or DEFAULT_ENTITY_TYPE)
        if not attributes:
            log.info("       | Pure spatial question - no data attributes requested")

        # Data was asked for but no year was named or implied. Guessing one
        # (the system used to silently substitute 2021) answers a question the
        # user never asked; rejecting outright and saying what is missing is the
        # honest response, the same way an UNRELATED question is rejected rather
        # than forced into a category it doesn't fit.
        if attributes and not temporal:
            log.warning("       | %s requested but no year was named or implied - rejecting",
                        ", ".join(attributes))
            return QueryParams(
                query_type="NEEDS_YEAR", spatial=[], temporal=[], attributes=attributes,
                raw_query=original_query, **_trace(),
            ), tokens_consumed

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
        log.info("       | Entity type : %s", entity_type or f"(default: {DEFAULT_ENTITY_TYPE})")
        log.info("       | Tokens      : classify=%d  extract=%d  total=%d",
                 tokens_classify, tokens_consumed - tokens_classify, tokens_consumed)

        return QueryParams(
            query_type=query_type, spatial=spatial, temporal=temporal,
            attributes=attributes, spatial_relationship=relationship,
            entity_type=entity_type, raw_query=original_query, **_trace(),
        ), tokens_consumed
