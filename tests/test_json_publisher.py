import pytest
import json
import shutil
from pathlib import Path
from unittest.mock import patch
from src.services.reporting.json_publisher import JSONPublisher
from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData, EventMetadata

@pytest.fixture
def clean_dirs():
    base = Path("test_publisher_output")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_json_publisher_archives_analytics(clean_dirs):
    # Setup dummy directories
    output_dir = clean_dirs / "dashboard"
    output_dir.mkdir(parents=True)
    output_file = output_dir / "data.json"
    
    # Create a dummy analytics.json file
    analytics_file = output_dir / "analytics.json"
    with open(analytics_file, 'w') as f:
        json.dump({"dummy": "analytics"}, f)
        
    publisher = JSONPublisher(clean_dirs, output_file)
    publisher.publish()
    
    # Assert data.json was created
    assert output_file.exists()
    
    # Assert historical archive directory exists
    history_dir = output_dir / "history"
    assert history_dir.exists()
    
    # Assert history file lists
    history_files = list(history_dir.glob("data_*.json"))
    assert len(history_files) == 1
    
    # Assert analytics history file exists
    analytics_history_files = list(history_dir.glob("analytics_*.json"))
    assert len(analytics_history_files) == 1
    
    # Verify the contents of the archived analytics file
    with open(analytics_history_files[0], 'r') as f:
        archived_analytics = json.load(f)
    assert archived_analytics == {"dummy": "analytics"}


def test_json_publisher_validation_blocker(clean_dirs):
    # Setup dummy directories
    output_dir = clean_dirs / "dashboard"
    output_dir.mkdir(parents=True)
    output_file = output_dir / "data.json"
    
    publisher = JSONPublisher(clean_dirs, output_file)
    
    invalid_event = MacroEvent(
        event_id="mock-123",  # Mock ID -> raises ValueError
        official_data=OfficialData(
            title="Test Event",
            publication_date="2026-06-15T00:00:00Z",
            category="Test",
            source="Test",
            official_url="http://example.com"
        ),
        derived_data=DerivedData(),
        metadata=EventMetadata(
            processing_state="NEW",
            lifecycle_status="ACTIVE",
            created_at="2026-06-15T00:00:00Z",
            updated_at="2026-06-15T00:00:00Z"
        )
    )
    
    with patch("src.services.macro_intelligence.query_service.MacroQueryService.get_all_events", return_value=[invalid_event]):
        with pytest.raises(ValueError, match="Release Blocker: Mock ID found"):
            publisher.publish()
