import pytest
from src.services.macro_intelligence.enrichment_service import EnrichmentService, DummyProvider
from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData, EventMetadata

def test_enrichment_service_dummy_provider():
    # Setup dummy provider
    provider = DummyProvider()
    service = EnrichmentService(provider)
    
    # Create a mock event
    event = MacroEvent(
        event_id="evt-1",
        official_data=OfficialData(
            title="Policy Announcement",
            publication_date="2026-06-15T12:00:00Z",
            category="Press Release",
            source="RBI",
            official_url="http://rbi/1",
            content="Mock content detailing policy rate cut."
        ),
        derived_data=DerivedData(),
        metadata=EventMetadata(
            processing_state="NEW",
            lifecycle_status="ACTIVE",
            created_at="2026-06-15T12:00:00Z",
            updated_at="2026-06-15T12:00:00Z"
        )
    )
    
    # Process the event
    enriched_event = service.process(event)
    
    # Verify that the event is enriched
    assert enriched_event.metadata.lifecycle_status == "ENRICHED"
    assert enriched_event.metadata.processing_state == "ENRICHED"
    assert len(enriched_event.derived_data.ai_snapshots) == 1
    
    snapshot = enriched_event.derived_data.ai_snapshots[0]
    
    # Verify the dummy provider provenance values
    assert snapshot.provider == "dummy"
    assert snapshot.model == "dummy"
    assert snapshot.confidence == 0
    assert "Please configure an API key" in snapshot.summary
