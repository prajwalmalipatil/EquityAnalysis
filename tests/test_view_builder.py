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

def test_publish_preserves_macro_assets(clean_dirs):
    # Setup mock live dashboard with existing assets
    live_dashboard = clean_dirs / "dashboard"
    live_dashboard.mkdir(parents=True, exist_ok=True)
    
    # Create dummy static assets and macro assets
    assets = [
        "index.html", "app.js", "styles.css", "data.json", "backtest_results.json",
        "analytics.json", "search-index.json", "relationships.json", "graph.json"
    ]
    for asset in assets:
        with open(live_dashboard / asset, 'w') as f:
            f.write(f"dummy content for {asset}")
            
    vb = ViewBuilderService(clean_dirs)
    vb.publish()
    
    # Check that both ETE outputs and preserved macro assets exist in the live dashboard directory after the swap
    assert (live_dashboard / "manifest.json").exists()
    assert (live_dashboard / "system_health.json").exists()
    
    for asset in assets:
        assert (live_dashboard / asset).exists(), f"{asset} was not preserved"
        with open(live_dashboard / asset, 'r') as f:
            content = f.read()
        assert content == f"dummy content for {asset}"
