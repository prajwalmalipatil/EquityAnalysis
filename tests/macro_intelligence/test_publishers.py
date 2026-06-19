import json
import pytest
from pathlib import Path
from src.services.macro_intelligence.publishers import ManifestPublisher
from src.services.macro_intelligence.read_models import ManifestViewModel

def test_manifest_publisher_merge(tmp_path):
    output_dir = tmp_path / "dashboard"
    output_dir.mkdir()
    
    # 1. Create a pre-existing manifest.json with ETE stats and files
    existing_manifest = {
        "schema_version": "1.0",
        "engine_version": "1.0.0",
        "generated_at": "2026-06-16T10:00:00Z",
        "research_events": 12,
        "active_sequences": 3,
        "completed_sequences": 5,
        "failed_sequences": 2,
        "last_market_date": "2026-06-16",
        "files": {
            "summary": "summary.123456.json",
            "indicator_catalog": "indicator_catalog.abc.json"
        }
    }
    
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(existing_manifest, f)
        
    # 2. Initialize publisher and publish new Macro manifest
    publisher = ManifestPublisher(output_dir)
    new_manifest_vm = ManifestViewModel(
        schema_version=2,
        generated_at="2026-06-16T12:00:00Z",
        generator="Macro Intelligence Pipeline",
        pipeline_version="2.1.0",
        repository_version="v1",
        event_count=54,
        analytics_version=1,
        search_index_version=1,
        artifacts=["data.json"]
    )
    
    publisher.publish(new_manifest_vm)
    
    # 3. Read back the manifest and verify merged content
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        merged_data = json.load(f)
        
    # Check that new macro manifest keys are present
    assert merged_data["schema_version"] == 2
    assert merged_data["generated_at"] == "2026-06-16T12:00:00Z"
    assert merged_data["generator"] == "Macro Intelligence Pipeline"
    assert merged_data["event_count"] == 54
    assert merged_data["artifacts"] == ["data.json"]
    
    # Check that ETE keys were successfully merged and NOT clobbered
    assert merged_data["engine_version"] == "1.0.0"
    assert merged_data["research_events"] == 12
    assert merged_data["active_sequences"] == 3
    assert merged_data["completed_sequences"] == 5
    assert merged_data["failed_sequences"] == 2
    assert merged_data["last_market_date"] == "2026-06-16"
    assert merged_data["files"] == {
        "summary": "summary.123456.json",
        "indicator_catalog": "indicator_catalog.abc.json"
    }
    
    # Check that historical manifest file was written
    history_manifest = output_dir / "history" / "manifest_2026-06-16.json"
    assert history_manifest.exists()
    with open(history_manifest, "r", encoding="utf-8") as f:
        history_manifest_data = json.load(f)
    assert history_manifest_data["schema_version"] == 2
    assert history_manifest_data["active_sequences"] == 3

def test_analytics_publisher_history(tmp_path):
    from src.services.macro_intelligence.publishers import AnalyticsPublisher
    from src.services.macro_intelligence.read_models import AnalyticsViewModel
    
    output_dir = tmp_path / "dashboard"
    output_dir.mkdir()
    
    publisher = AnalyticsPublisher(output_dir)
    analytics_vm = AnalyticsViewModel(
        schema_version=1,
        generated_at="2026-06-16T12:00:00Z",
        pipeline_version="2.1.0",
        event_count=5,
        analytics={"quality": {"avg_quality_score": 85.5}}
    )
    
    publisher.publish(analytics_vm)
    
    # Check regular analytics.json
    analytics_file = output_dir / "analytics.json"
    assert analytics_file.exists()
    
    # Check archived historical analytics.json
    history_analytics = output_dir / "history" / "analytics_2026-06-16.json"
    assert history_analytics.exists()
    with open(history_analytics, "r", encoding="utf-8") as f:
        history_data = json.load(f)
        
    assert history_data["schema_version"] == 1
    assert history_data["event_count"] == 5
    assert history_data["analytics"]["quality"]["avg_quality_score"] == 85.5

