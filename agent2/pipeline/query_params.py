"""
The parsed form of a user query, and the vocabulary the pipeline is allowed to
produce.

Every stage of the pipeline speaks in terms of these types: the classifier
picks one of VALID_QUERY_TYPES, the extractor fills the raw fields, and
QueryParams is what the controller dispatches on. Keeping the shape here means
no stage has to import another stage just to know what a query looks like.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The eight categories a query can be classified into. Seven are answerable
# and map onto the document's scenarios; UNRELATED is the refusal.
VALID_QUERY_TYPES = {
    "DIRECT_LOOKUP", "SPATIAL_ADJACENCY", "SPATIAL_DIRECTION", "SPATIAL_DISTANCE",
    "GEOMETRY_LOOKUP", "SPATIAL_OPERATION", "SPATIAL_RELATIONSHIP_BUFFER", "UNRELATED",
}

# The three verdict relationships share one extraction template and one output
# shape (a spatial_relationship object).
RELATIONSHIP_TYPES = {"SPATIAL_ADJACENCY", "SPATIAL_DIRECTION", "SPATIAL_DISTANCE"}

VALID_OPERATIONS = {"Union", "Intersection", "Difference", "SymDifference", "BufferWithin"}
VALID_ATTRS = {"population", "marriages", "live_births"}
VALID_ENTITY_TYPES = {"city", "state"}

MIN_YEAR = 1990
MAX_YEAR = 2030
DEFAULT_YEAR = 2021


@dataclass
class SpatialRelationship:
    """A relationship question: which kind, and between what.

    `subject` is what is being asked *about*; `refs` is what it is measured
    against. Its presence is what separates the two forms of the question:

        "Does Sachsen lie north of Bayern?"   subject=Sachsen, refs=[Bayern]
            -> a verdict about one named pair

        "Which states are north of Bayern?"   subject=None,    refs=[Bayern]
            -> the set of states for which the verdict holds

    Both are answered from the same computation; the subject only decides
    whether the answer is a yes/no or the set itself. Order matters, as the
    specification notes for north_of: the reference is what the bearing is
    measured from.
    """
    type: str
    refs: List[str] = field(default_factory=list)
    distance_km: Optional[float] = None
    subject: Optional[str] = None


@dataclass
class QueryParams:
    """Everything the rest of the pipeline needs to answer one query."""
    query_type: str
    spatial: List[str]
    temporal: List[int]
    attributes: List[str]
    spatial_relationship: Optional[SpatialRelationship] = None
    # Set only when the question named a subject, i.e. asked for a yes/no about
    # one pair. None means the question asked for the set, not a verdict.
    verdict: Optional[bool] = None
    raw_query: str = ""
    # GEOMETRY_LOOKUP only
    entities: Optional[List[Dict[str, str]]] = None
    # SPATIAL_OPERATION / SPATIAL_RELATIONSHIP_BUFFER
    distance_km: Optional[float] = None
    target_entity: Optional[str] = None
    # SPATIAL_OPERATION only
    operation: Optional[str] = None
    entity_type: Optional[str] = None
    # Step-by-step trace, for evaluation logging — see metrics_logger.py
    classify_tokens: int = 0
    extract_tokens: int = 0
    extracted_data: Dict[str, Any] = field(default_factory=dict)
