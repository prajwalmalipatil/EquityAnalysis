import pytest
import tempfile
from pathlib import Path
from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData, EventMetadata, ImpactAnalysis
from src.services.macro_intelligence.event_repository import JSONEventWriteRepository, JSONEventReadRepository
from src.services.macro_intelligence.config import StorageConfig, MacroConfig, DeduplicationConfig, CollectionConfig, AIEnrichmentConfig

@pytest.fixture
def macro_config(tmp_path):
    storage = StorageConfig(
        base_path=tmp_path,
        history_file=tmp_path / "rbi_events.jsonl",
        attachments_dir=tmp_path / "attachments",
        metadata_dir=tmp_path / "metadata",
        logs_dir=tmp_path / "logs",
        cache_dir=tmp_path / "cache",
        state_file=tmp_path / "collector_state.json"
    )
    dedup = DeduplicationConfig(
        similarity_threshold=0.85,
        weights={"title": 0.4, "publication_date": 0.2, "official_url": 0.1, "document_hash": 0.2, "attachment_hash": 0.1}
    )
    coll = CollectionConfig(retry_limit=3, base_backoff_seconds=1, max_backoff_seconds=10, timeout_seconds=30)
    ai = AIEnrichmentConfig(provider="gemini", model_name="gemini-1.5-pro", confidence_threshold=80, prompt_version="1")
    return MacroConfig(version="2.1.0", collection=coll, deduplication=dedup, ai_enrichment=ai, storage=storage)

@pytest.fixture
def mock_event_1():
    return MacroEvent(
        event_id="evt-1",
        official_data=OfficialData(
            title="Title 1",
            publication_date="2026-06-10T10:00:00Z",
            category="Test",
            source="RBI",
            official_url="http://rbi/1"
        ),
        derived_data=DerivedData(
            impact=ImpactAnalysis(direction="Neutral", severity="Low", confidence=90, asset_classes=[], sectors=[], securities=[], horizon="Short", importance=5)
        ),
        metadata=EventMetadata(processing_state="COMPLETED", lifecycle_status="ACTIVE", created_at="2026-06-10T10:00:00Z", updated_at="2026-06-10T10:00:00Z")
    )

@pytest.fixture
def mock_event_2():
    return MacroEvent(
        event_id="evt-2",
        official_data=OfficialData(
            title="Title 2",
            publication_date="2026-06-11T10:00:00Z",
            category="Test",
            source="RBI",
            official_url="http://rbi/2"
        ),
        derived_data=DerivedData(
            impact=ImpactAnalysis(direction="Neutral", severity="Low", confidence=90, asset_classes=[], sectors=[], securities=[], horizon="Short", importance=5)
        ),
        metadata=EventMetadata(processing_state="COMPLETED", lifecycle_status="ACTIVE", created_at="2026-06-10T10:00:00Z", updated_at="2026-06-10T10:00:00Z")
    )

def test_save_and_retrieve_event(macro_config, mock_event_1):
    write_repo = JSONEventWriteRepository(macro_config)
    read_repo = JSONEventReadRepository(macro_config.storage)
    
    assert write_repo.save_event(mock_event_1) == True
    
    events = read_repo.get_all_events()
    assert len(events) == 1
    assert events[0].event_id == "evt-1"
    assert events[0].official_data.title == "Title 1"

def test_deduplication(macro_config, mock_event_1):
    write_repo = JSONEventWriteRepository(macro_config)
    
    assert write_repo.save_event(mock_event_1) == True
    assert write_repo.save_event(mock_event_1) == False # Exact duplicate should be blocked
    
    read_repo = JSONEventReadRepository(macro_config.storage)
    assert len(read_repo.get_all_events()) == 1

def test_active_events_only(macro_config, mock_event_1, mock_event_2):
    mock_event_2.metadata.lifecycle_status = "WITHDRAWN"
    
    write_repo = JSONEventWriteRepository(macro_config)
    write_repo.save_event(mock_event_1)
    write_repo.save_event(mock_event_2)
    
    read_repo = JSONEventReadRepository(macro_config.storage)
    active_events = read_repo.get_active_events()
    
    assert len(active_events) == 1
    assert active_events[0].event_id == "evt-1"
