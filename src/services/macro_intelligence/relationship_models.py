from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class RelationshipType(Enum):
    SUPERSEDES = "SUPERSEDES"
    AMENDS = "AMENDS"
    WITHDRAWS = "WITHDRAWS"
    REFERENCES = "REFERENCES"
    RELATED = "RELATED"
    IMPLEMENTS = "IMPLEMENTS"
    CLARIFIES = "CLARIFIES"
    REPLACES = "REPLACES"
    DUPLICATES = "DUPLICATES"

class RelationshipConfidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass(frozen=True)
class RuleResult:
    relationship_type: RelationshipType
    score: float
    rule_name: str
    matched_terms: List[str]
    signals: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class RelationshipCandidate:
    source_event_id: str
    target_event_id: str
    generator_score: float
    generator_signals: List[str]

@dataclass(frozen=True)
class Relationship:
    id: str  # Canonical SHA256 ID
    source_event_id: str
    target_event_id: str
    type: RelationshipType
    confidence: RelationshipConfidence
    provenance: dict  # Contains rule, matched_terms, signals, weights
    rule_version: str
    resolver_version: str
    created_at: str
