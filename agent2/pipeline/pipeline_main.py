"""
Pipeline entry point — constructs each pipeline class and calls its method,
in order: CleanQuery -> QueryClassifier -> QueryExtractor -> QueryParamsBuilder
-> validate_spatial. query_controller.py's /query endpoint calls run() here
directly instead of sequencing these calls itself.

QueryParamsBuilder is the same class parse_query() uses internally; it's
constructed and called directly here so this file doesn't duplicate its
per-category shaping logic and doesn't run classify()/extract() a second time.

validate_spatial() only runs for DIRECT_LOOKUP and the three relationship
types — the same categories query_controller.py used to gate it for itself —
since it no-ops for DIRECT_LOOKUP anyway but has nothing at all to do for
GEOMETRY_LOOKUP/SPATIAL_OPERATION/SPATIAL_RELATIONSHIP_BUFFER/NEEDS_YEAR, and
those shouldn't pay for an unused DB round trip.
"""
from __future__ import annotations

import os
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from .query_classifier import QueryClassifier
from .clean_query import CleanQuery
from .query_extractor import QueryExtractor
from .query_params import QueryParams, RELATIONSHIP_TYPES
from .query_parser import QueryParamsBuilder
from .spatial_validator import validate_spatial


def run(query: str, model: str | None = None) -> QueryParams:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    clean_query = CleanQuery(query)
    cleaned = clean_query.cleaned

    classifier = QueryClassifier(client, model=model) if model else QueryClassifier(client)
    query_type, tokens_classify = classifier.classify(cleaned)

    if query_type == "UNRELATED":
        return QueryParams(
            query_type="UNRELATED", spatial=[], temporal=[], attributes=[],
            raw_query=query, classify_tokens=tokens_classify,
            extract_tokens=0, extracted_data={}, classify_model=classifier.model,
        )

    extractor = QueryExtractor(client, model=model) if model else QueryExtractor(client)
    data, tokens_extract = extractor.extract(cleaned, query_type)
    tokens_consumed = tokens_classify + tokens_extract

    params, tokens_consumed = QueryParamsBuilder(extractor).build(
        cleaned_query=cleaned, query_type=query_type, data=data,
        original_query=query, tokens_classify=tokens_classify, tokens_consumed=tokens_consumed,
        classify_model=classifier.model,
    )

    if params.query_type == "DIRECT_LOOKUP" or params.query_type in RELATIONSHIP_TYPES:
        params = validate_spatial(params)

    return params
