from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str  # "Event"
    label: str
    category: str
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class GraphEdge:
    id: str
    relationship_id: str
    source: str
    target: str
    weight: float
    relationship_type: str
    confidence: str

@dataclass(frozen=True)
class GraphStatistics:
    node_count: int
    edge_count: int
    connected_components: int
    average_degree: float
    relationship_distribution: Dict[str, int]

@dataclass(frozen=True)
class GraphReadModel:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    statistics: GraphStatistics

@dataclass(frozen=True)
class GraphViewModel:
    schema_version: str
    generated_at: str
    pipeline_version: str
    statistics: dict
    nodes: List[dict]
    edges: List[dict]
