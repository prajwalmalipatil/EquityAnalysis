import pytest
import json
import shutil
from pathlib import Path
from src.services.reporting.data_contract_validator import DataContractValidator

@pytest.fixture
def test_dir():
    base = Path("test_contract_data")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_valid_contracts(test_dir):
    # Setup valid mock files
    manifest = {
        "schema_version": "1.0",
        "engine_version": "1.0.0",
        "generated_at": "2023-01-01T00:00:00Z",
        "research_events": 100,
        "active_sequences": 1,
        "completed_sequences": 0,
        "failed_sequences": 0,
        "last_market_date": "2023-01-01",
        "files": {
            "summary": "summary.123456.json"
        }
    }
    health = {
        "schema_version": "1.0",
        "generated_at": "2023-01-01T00:00:00Z",
        "research_engine": "1.0.0",
        "last_run": "2023-01-01T00:00:00Z",
        "integrity": "PASS",
        "hash_failures": 0,
        "active_sequences": 1,
        "pipeline_seconds": 1.0,
        "reconstruction_seconds": 0.5,
        "publish_seconds": 0.5
    }
    summary = {
        "active": [{
            "sequence_id": "seq_1",
            "symbol": "AAPL",
            "timeframe": "daily",
            "state": "Triggered",
            "current_stage": "T+1",
            "confidence": 95.5,
            "progress": []
        }],
        "completed": [],
        "failed": []
    }
    
    with open(test_dir / "manifest.json", 'w') as f: json.dump(manifest, f)
    with open(test_dir / "system_health.json", 'w') as f: json.dump(health, f)
    with open(test_dir / "summary.123456.json", 'w') as f: json.dump(summary, f)
    
    validator = DataContractValidator(test_dir)
    assert validator.validate_all() == True

def test_invalid_manifest(test_dir):
    # Missing required fields
    with open(test_dir / "manifest.json", 'w') as f: json.dump({"engine_version": "1.0.0"}, f)
    
    validator = DataContractValidator(test_dir)
    assert validator.validate_all() == False
