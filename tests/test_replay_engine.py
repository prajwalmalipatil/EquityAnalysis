import pytest
import shutil
import json
from pathlib import Path
from src.services.orchestration.replay_engine import ReplayEngine

@pytest.fixture
def test_dir():
    base = Path("test_replay_data")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_replay_engine_filters_future_events(test_dir, monkeypatch):
    monkeypatch.setattr("src.services.vsa.eigen_transition_engine_service.EigenTransitionEngineService._calculate_hash", lambda self, ev: ev.get("hash"))
    
    source_dir = test_dir / "source_events"
    source_dir.mkdir()
    
    out_dir = test_dir / "output"
    out_dir.mkdir()
    
    # Create mock events
    events = [
        {"timestamp": "2023-11-10T00:00:00Z", "id": 1, "action": "INIT"},
        {"timestamp": "2023-11-14T00:00:00Z", "id": 2, "action": "ADVANCE"},
        {"timestamp": "2023-11-15T00:00:00Z", "id": 3, "action": "ADVANCE"},
        {"timestamp": "2023-11-16T00:00:00Z", "id": 4, "action": "COMPLETE"} # Future
    ]
    
    with open(source_dir / "daily_events.jsonl", "w") as f:
        last_h = ""
        for i, ev in enumerate(events):
            # Add missing fields required by parser just in case
            ev["sequence_id"] = "seq_1"
            ev["symbol"] = "AAPL"
            ev["timeframe"] = "daily"
            ev["hash"] = f"h{i}"
            ev["previous_hash"] = last_h
            last_h = f"h{i}"
            f.write(json.dumps(ev) + "\n")
            
    engine = ReplayEngine(output_dir=out_dir, source_events_dir=source_dir)
    engine.run_replay("2023-11-15")
    
    # Replay outputs to out_dir/dashboard_next then renames to out_dir/dashboard
    dashboard_dir = out_dir / "dashboard"
    assert dashboard_dir.exists()
    
    # Check manifest logic
    manifest_path = dashboard_dir / "manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    # We should have 3 events, because the 4th was in the future
    assert manifest["research_events"] == 3
