import pytest
import os
import shutil
from pathlib import Path
from src.services.reporting.view_builder_service import ViewBuilderService

@pytest.fixture
def clean_dirs():
    base = Path("test_output")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_publish(clean_dirs):
    vb = ViewBuilderService(clean_dirs)
    vb.publish()
    
    # Check outputs
    assert (clean_dirs / "dashboard" / "manifest.json").exists()
    assert (clean_dirs / "dashboard" / "system_health.json").exists()
    
    # Check that dashboard_next was atomically swapped
    assert not (clean_dirs / "dashboard_next").exists()
