import pytest
from src.services.macro_intelligence.relationship_models import RelationshipType, RelationshipConfidence
from src.services.macro_intelligence.relationship_engine import RelationshipCandidateGenerator, RelationshipResolver
from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData

def create_event(id: str, title: str, content: str, date: str, cat: str) -> MacroEvent:
    return MacroEvent(
        event_id=id,
        official_data=OfficialData(
            title=title,
            publication_date=date,
            category=cat,
            source="RBI",
            official_url="",
            content=content,
            attachments=[],
            pdf_url=""
        ),
        derived_data=None,
        metadata=None
    )

def test_relationship_engine():
    # Setup Events
    e1 = create_event("ev-1", "Master Circular on Liquidity", "Original rules on liquidity coverage.", "2026-01-01", "Banking")
    e2 = create_event("ev-2", "Revised Master Circular on Liquidity", "This supersedes the original master circular on liquidity.", "2026-06-01", "Banking")
    e3 = create_event("ev-3", "Amendment to KYC rules", "Amends the previous KYC rules.", "2026-06-15", "Banking")
    
    events = [e1, e2, e3]
    
    # 1. Test Candidate Generation
    generator = RelationshipCandidateGenerator()
    candidates = generator.generate(events)
    
    # ev-1 and ev-2 should be candidates because they share a category
    # ev-2 and ev-3 should be candidates (same category)
    # Since ev-1, ev-2, ev-3 are all "Banking", we have 3 candidates (1-2, 1-3, 2-3)
    assert len(candidates) == 3
    
    # 2. Test Rule Engine & Resolver
    resolver = RelationshipResolver()
    relationships = resolver.resolve(candidates, events)
    
    # Expect 1 strong SUPERSEDES relationship (ev-2 supersedes ev-1)
    # We might also see AMENDS if there's text but e3 amends nothing explicitly in our mock. 
    # Wait, e3 has "amends" in content. Let's see if e3 creates a relationship.
    # The rule compares two documents. If source=e3, target=e1, it sees "amends" in e3. 
    # Our simple AmendsRule currently just checks if the source has "amends". It doesn't check if the target is mentioned.
    # This is a basic rule. It should create an AMENDS relationship.
    
    assert len(relationships) >= 1
    
    # Check that ev-2 supersedes ev-1
    supersedes = [r for r in relationships if r.type == RelationshipType.SUPERSEDES]
    assert len(supersedes) > 0
    assert supersedes[0].source_event_id == "ev-2"
    assert supersedes[0].target_event_id == "ev-1"
    assert supersedes[0].confidence == RelationshipConfidence.HIGH
    
    # Check new refined fields
    assert len(supersedes[0].id) == 16  # SHA256 truncated
    assert supersedes[0].rule_version == "1.0.0"
    assert supersedes[0].provenance["rule"] == "SupersedesRule"
    assert "explicit keyword 'supersedes'" in supersedes[0].provenance["matched_terms"]
    
    # Check that ev-3 amends something (since "amends" is in the text)
    amends = [r for r in relationships if r.type == RelationshipType.AMENDS]
    assert len(amends) > 0
    assert amends[0].source_event_id == "ev-3"
