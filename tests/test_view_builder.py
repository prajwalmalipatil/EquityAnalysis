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

def test_publish_preserves_history_assets(clean_dirs):
    live_dashboard = clean_dirs / "dashboard"
    history_dir = live_dashboard / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # Create history files
    history_files = [
        "index.json",
        "data_2026-06-16.json",
        "analytics_2026-06-16.json",
        "manifest_2026-06-16.json",
        "rbi_events.jsonl"
    ]
    for h_file in history_files:
        with open(history_dir / h_file, 'w') as f:
            f.write(f"dummy history content for {h_file}")
            
    # Add an unpreserved file to verify it gets filtered out
    with open(history_dir / "unpreserved.json", 'w') as f:
        f.write("should not be preserved")
        
    vb = ViewBuilderService(clean_dirs)
    vb.publish()
    
    # Check that history folder and expected files were preserved in the live dashboard directory after the swap
    new_history_dir = live_dashboard / "history"
    assert new_history_dir.exists()
    
    for h_file in history_files:
        preserved_path = new_history_dir / h_file
        assert preserved_path.exists(), f"History file {h_file} was not preserved"
        with open(preserved_path, 'r') as f:
            content = f.read()
        assert content == f"dummy history content for {h_file}"
        
    assert not (new_history_dir / "unpreserved.json").exists()


def test_publish_preserves_ete_and_metadata_assets(clean_dirs):
    live_dashboard = clean_dirs / "dashboard"
    live_dashboard.mkdir(parents=True, exist_ok=True)
    
    ete_files = [
        "summary.oldhash.json",
        "indicator_catalog.oldhash.json",
        "history_index.oldhash.json"
    ]
    metadata_files = [
        "analytics-history.json",
        "system_health.json"
    ]
    
    all_files = ete_files + metadata_files
    for filename in all_files:
        with open(live_dashboard / filename, 'w') as f:
            f.write(f"dummy content for {filename}")
            
    # Add an unpreserved JSON file in root to verify it gets filtered out
    with open(live_dashboard / "unpreserved_root.json", 'w') as f:
        f.write("should not be preserved")
        
    vb = ViewBuilderService(clean_dirs)
    vb.publish()
    
    # Check that expected files were preserved in the live dashboard directory after the swap
    for filename in all_files:
        preserved_path = live_dashboard / filename
        assert preserved_path.exists(), f"File {filename} was not preserved"
        if filename != "system_health.json":
            with open(preserved_path, 'r') as f:
                content = f.read()
            assert content == f"dummy content for {filename}"
        
    assert not (live_dashboard / "unpreserved_root.json").exists()




