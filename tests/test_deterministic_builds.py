import pytest
import shutil
import json
import pandas as pd
from pathlib import Path
from src.services.vsa.eigen_transition_engine_service import EigenTransitionEngineService

@pytest.fixture
def test_dir():
    base = Path("test_determinism_data")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_deterministic_hashes(test_dir, monkeypatch):
    """Ensure two runs on identical data produce byte-for-byte identical event hashes."""
    # Monkeypatch to use local test dir for events
    monkeypatch.setattr("src.services.vsa.eigen_transition_engine_service.ETE_EVENTS_DIR", str(test_dir))
    monkeypatch.setattr("src.services.vsa.eigen_transition_engine_service.ETE_SNAPSHOTS_DIR", str(test_dir))
    
    df1 = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Open": [100, 101, 102],
        "High": [105, 106, 107],
        "Low": [95, 96, 97],
        "Close": [102, 103, 104],
        "Volume": [1000, 1050, 1100]
    })
    
    # --- RUN 1 ---
    engine1 = EigenTransitionEngineService("daily")
    engine1.events_file = test_dir / "run1_events.jsonl"
    engine1.snapshots_file = test_dir / "run1_snap.json"
    
    engine1.detect_triggers("AAPL", df1.iloc[:1], True) # Init
    engine1.update_active_sequences("AAPL", df1.iloc[:2]) # Advance
    
    run1_lines = engine1.events_file.read_text().strip().split("\n")
    run1_hashes = [json.loads(line)["hash"] for line in run1_lines]
    
    # --- RUN 2 (Different time, same data) ---
    engine2 = EigenTransitionEngineService("daily")
    engine2.events_file = test_dir / "run2_events.jsonl"
    engine2.snapshots_file = test_dir / "run2_snap.json"
    
    engine2.detect_triggers("AAPL", df1.iloc[:1], True) # Init
    engine2.update_active_sequences("AAPL", df1.iloc[:2]) # Advance
    
    run2_lines = engine2.events_file.read_text().strip().split("\n")
    run2_hashes = [json.loads(line)["hash"] for line in run2_lines]
    
    assert len(run1_hashes) == len(run2_hashes)
    assert len(run1_hashes) > 0
    assert run1_hashes == run2_hashes
