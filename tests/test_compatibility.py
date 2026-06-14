import pytest
import json
from pathlib import Path
from src.services.macro_intelligence.config import load_config
from src.services.macro_intelligence.event_repository import JSONEventReadRepository
from src.services.macro_intelligence.dashboard_mapper import DashboardMapper
from src.services.macro_intelligence.builders import ManifestBuilder
from src.services.macro_intelligence.release_validator import ReleaseValidator
from src.services.macro_intelligence.read_models import DashboardBundle
import dataclasses

def test_regression_compatibility(tmp_path):
    """
    Test that the Read Model, Builders, and Validator can successfully parse 
    and process legacy JSONL events without breaking.
    """
    # 1. Create a dummy legacy event inside a mock repository
    legacy_event_dict = {
      "event_id": "mock-rbi-1",
      "official_data": {
        "title": "Legacy Test Event",
        "publication_date": "2023-01-01T12:00:00Z",
        "category": "Monetary Policy",
        "official_url": "http://mock/rbi",
        "content": "Test content",
        "attachments": [],
        "source": "RBI",
        "pdf_url": None
      },
      "derived_data": {
        "impact": {
          "direction": "Neutral",
          "severity": "Informational",
          "confidence": 85
        },
        "ai_summary": "Legacy summary",
        "keywords": ["test"],
        "ai_theme": "Testing"
      },
      "metadata": {
        "processing_state": "COMPLETED",
        "lifecycle_status": "ACTIVE",
        "related_event_ids": [],
        "supersedes_event_id": None
      }
    }
    
    # Write to a tmp history file
    history_file = tmp_path / "history" / "rbi_events.jsonl"
    history_file.parent.mkdir(parents=True)
    with open(history_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(legacy_event_dict) + "\n")
        
    # Configure read repo
    real_config_path = Path("src/services/macro_intelligence/config.yaml")
    config = load_config(real_config_path)
    new_storage = dataclasses.replace(
        config.storage, 
        base_path=tmp_path / "history", 
        history_file=history_file
    )
    config = dataclasses.replace(config, storage=new_storage)
    
    read_repo = JSONEventReadRepository(config.storage)
    
    # 2. Extract and Build 
    # Must correctly parse schema version differences if any exist
    events = read_repo.get_active_events()
    dashboard_events = DashboardMapper.map_events(events)
    manifest = ManifestBuilder.build(len(events), {})
    
    bundle = DashboardBundle(
        events=dashboard_events,
        analytics=None,
        manifest=manifest
    )
    
    # 3. Assertions
    assert len(bundle.events) == 1
    assert bundle.events[0].title == "Legacy Test Event"
    assert bundle.manifest.event_count == 1
    
    # 4. Validation
    # Normally mock- IDs fail release validation. We can expect the ValueError to be raised here
    # to prove the validator is actually working on the legacy event.
    with pytest.raises(ValueError, match="Release Blocker: Mock ID found"):
        ReleaseValidator.validate(bundle)
