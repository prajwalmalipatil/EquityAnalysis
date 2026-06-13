import pytest
import shutil
import json
from pathlib import Path
from src.services.orchestration.registry import PlatformRegistry, ResearchModule

@pytest.fixture
def test_dir():
    base = Path("test_registry_data")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_research_registry_validation(test_dir):
    registry = PlatformRegistry()
    
    dq = ResearchModule(
        name="DataQualityGate",
        version="1.0.0",
        description="Filters OHLCV anomalies",
        inputs=["CSV"],
        outputs=["CleanCSV"],
        dependencies=[]
    )
    
    vsa = ResearchModule(
        name="VSAProcessor",
        version="1.0.0",
        description="Volume Spread Analysis",
        inputs=["CleanCSV"],
        outputs=["Signals"],
        dependencies=["DataQualityGate"]
    )
    
    registry.register(dq)
    registry.register(vsa)
    
    # Validation should pass
    assert registry.validate_dependencies() == True
    
    # Export
    out_file = test_dir / "registry.json"
    registry.export_manifest(out_file)
    assert out_file.exists()
    
    with open(out_file, "r") as f:
        data = json.load(f)
    assert len(data["modules"]) == 2
    assert data["modules"][0]["name"] == "DataQualityGate"

def test_missing_dependencies():
    registry = PlatformRegistry()
    
    vsa = ResearchModule(
        name="VSAProcessor",
        version="1.0.0",
        description="Volume Spread Analysis",
        inputs=["CleanCSV"],
        outputs=["Signals"],
        dependencies=["DataQualityGate"] # Missing
    )
    registry.register(vsa)
    
    # Validation should fail
    assert registry.validate_dependencies() == False
