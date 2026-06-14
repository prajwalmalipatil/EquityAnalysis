import uuid
import hashlib
from typing import List, Dict, Set
from collections import defaultdict
from datetime import datetime

from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.relationship_models import Relationship
from src.services.macro_intelligence.graph_models import (
    GraphNode, GraphEdge, GraphStatistics, GraphReadModel, GraphViewModel
)

class GraphBuilder:
    """Computes the GraphReadModel from events and relationships."""
    
    def _compute_components(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> int:
        adj = defaultdict(list)
        for e in edges:
            adj[e.source].append(e.target)
            adj[e.target].append(e.source)
            
        visited = set()
        components = 0
        
        for n in nodes:
            if n.id not in visited:
                components += 1
                queue = [n.id]
                while queue:
                    curr = queue.pop(0)
                    if curr not in visited:
                        visited.add(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                queue.append(neighbor)
        return components
        
    def build_read_model(self, events: List[MacroEvent], relationships: List[Relationship]) -> GraphReadModel:
        nodes = []
        for e in events:
            nodes.append(GraphNode(
                id=e.event_id,
                type="Event",
                label=e.official_data.title,
                category=e.official_data.category,
                metadata={
                    "date": e.official_data.publication_date,
                    "source": e.official_data.source
                }
            ))
            
        edges = []
        rel_dist = defaultdict(int)
        
        for r in relationships:
            rel_dist[r.type.value] += 1
            
            # Simple weight mapping: HIGH=1.0, MEDIUM=0.7, LOW=0.4
            weight = 1.0
            if r.confidence.name == "MEDIUM": weight = 0.7
            if r.confidence.name == "LOW": weight = 0.4
            
            raw_id = f"edge:{r.id}"
            edge_id = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:16]
            
            edges.append(GraphEdge(
                id=edge_id,
                relationship_id=r.id,
                source=r.source_event_id,
                target=r.target_event_id,
                weight=weight,
                relationship_type=r.type.value,
                confidence=r.confidence.value
            ))
            
        # Stats
        node_count = len(nodes)
        edge_count = len(edges)
        components = self._compute_components(nodes, edges) if nodes else 0
        avg_degree = (2 * edge_count / node_count) if node_count > 0 else 0.0
        
        stats = GraphStatistics(
            node_count=node_count,
            edge_count=edge_count,
            connected_components=components,
            average_degree=round(avg_degree, 2),
            relationship_distribution=dict(rel_dist)
        )
        
        return GraphReadModel(nodes=nodes, edges=edges, statistics=stats)

class GraphViewModelBuilder:
    """Converts the GraphReadModel into a pure serializable DTO (ViewModel)."""
    
    @staticmethod
    def build(read_model: GraphReadModel, pipeline_version: str) -> GraphViewModel:
        return GraphViewModel(
            schema_version="1",
            generated_at=datetime.utcnow().isoformat() + "Z",
            pipeline_version=pipeline_version,
            statistics={
                "node_count": read_model.statistics.node_count,
                "edge_count": read_model.statistics.edge_count,
                "connected_components": read_model.statistics.connected_components,
                "average_degree": read_model.statistics.average_degree,
                "relationship_distribution": read_model.statistics.relationship_distribution
            },
            nodes=[
                {
                    "id": n.id,
                    "type": n.type,
                    "label": n.label,
                    "category": n.category,
                    "metadata": n.metadata
                } for n in read_model.nodes
            ],
            edges=[
                {
                    "id": e.id,
                    "relationship_id": e.relationship_id,
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "relationship_type": e.relationship_type,
                    "confidence": e.confidence
                } for e in read_model.edges
            ]
        )
