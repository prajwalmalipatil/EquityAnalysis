import pytest
from src.services.macro_intelligence.graph_models import GraphNode, GraphEdge
from src.services.macro_intelligence.models import MacroEvent, OfficialData
from src.services.macro_intelligence.relationship_models import Relationship, RelationshipType, RelationshipConfidence
from src.services.macro_intelligence.graph_builder import GraphBuilder

def test_graph_builder():
    e1 = MacroEvent(
        event_id="e1",
        official_data=OfficialData(title="A", publication_date="2026-01-01", category="Banking", source="RBI", official_url="", content="", attachments=[], pdf_url=""),
        derived_data=None, metadata=None
    )
    e2 = MacroEvent(
        event_id="e2",
        official_data=OfficialData(title="B", publication_date="2026-02-01", category="Banking", source="RBI", official_url="", content="", attachments=[], pdf_url=""),
        derived_data=None, metadata=None
    )
    e3 = MacroEvent(
        event_id="e3",
        official_data=OfficialData(title="C", publication_date="2026-03-01", category="Monetary", source="RBI", official_url="", content="", attachments=[], pdf_url=""),
        derived_data=None, metadata=None
    )
    
    # One relationship: e1 supersedes e2
    r1 = Relationship(
        id="rel-123", source_event_id="e1", target_event_id="e2",
        type=RelationshipType.SUPERSEDES, confidence=RelationshipConfidence.HIGH,
        provenance={}, rule_version="1", resolver_version="1", created_at=""
    )
    
    events = [e1, e2, e3]
    relationships = [r1]
    
    builder = GraphBuilder()
    graph_rm = builder.build_read_model(events, relationships)
    
    assert graph_rm.statistics.node_count == 3
    assert graph_rm.statistics.edge_count == 1
    
    # e1 and e2 are connected (1 component), e3 is isolated (1 component) -> total 2
    assert graph_rm.statistics.connected_components == 2
    
    # Avg degree: 2 * 1 / 3 = 0.67
    assert graph_rm.statistics.average_degree == 0.67
    
    # Node mapping verification
    assert len(graph_rm.nodes) == 3
    n1 = next(n for n in graph_rm.nodes if n.id == "e1")
    assert n1.label == "A"
    
    # Edge mapping verification
    assert len(graph_rm.edges) == 1
    assert graph_rm.edges[0].relationship_type == "SUPERSEDES"
    assert graph_rm.edges[0].weight == 1.0
