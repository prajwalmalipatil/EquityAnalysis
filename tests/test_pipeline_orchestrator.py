import pytest
import shutil
from pathlib import Path
from src.services.orchestration.pipeline_orchestrator import PipelineOrchestrator

@pytest.fixture
def test_dir():
    base = Path("test_orchestrator_data")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    yield base
    if base.exists():
        shutil.rmtree(base)

def test_pipeline_halts_on_dq_failure(test_dir, monkeypatch):
    """If DQ gate fails on all files, the pipeline should exit"""
    # Create invalid file
    (test_dir / "invalid.csv").write_text("Date,Open,High,Low,Close,Volume\n2023-01-01,1,1,1,1,0\n")
    
    orchestrator = PipelineOrchestrator(test_dir, workers=1)
    
    # Mock sys.exit to avoid killing the test runner
    class ExitException(Exception): pass
    monkeypatch.setattr("sys.exit", lambda code: (_ for _ in ()).throw(ExitException("sys.exit called")))
    
    # Mock registry validation
    from src.services.orchestration.registry import platform_registry
    monkeypatch.setattr(platform_registry, "validate_dependencies", lambda: True)
    
    with pytest.raises(ExitException):
        orchestrator.execute_dag()
        
    assert orchestrator.stats["data_quality"]["passed"] == 0
    assert orchestrator.stats["data_quality"]["quarantined"] == 1

def test_pipeline_vsa_failure_halts(test_dir, monkeypatch):
    # Mock data quality to pass
    monkeypatch.setattr(PipelineOrchestrator, "run_stage_data_quality", lambda self: True)
    # Mock VSA to fail
    monkeypatch.setattr(PipelineOrchestrator, "run_stage_vsa_and_ete", lambda self: False)
    
    orchestrator = PipelineOrchestrator(test_dir, workers=1)
    
    class ExitException(Exception): pass
    monkeypatch.setattr("sys.exit", lambda code: (_ for _ in ()).throw(ExitException("sys.exit called")))
    
    # Mock registry validation
    from src.services.orchestration.registry import platform_registry
    monkeypatch.setattr(platform_registry, "validate_dependencies", lambda: True)
    
    with pytest.raises(ExitException):
        orchestrator.execute_dag()

def test_orchestrator_aggregates_stats(test_dir, monkeypatch):
    """Test that PipelineOrchestrator aggregates file processing stats correctly using ThreadPoolExecutor"""
    from src.services.vsa.processor_service import VSAProcessorService
    
    orchestrator = PipelineOrchestrator(test_dir, workers=2)
    
    # Create two dummy CSV files in test_dir
    (test_dir / "AAPL.csv").write_text("dummy")
    (test_dir / "MSFT.csv").write_text("dummy")
    
    # Mock VSAProcessorService.process_file to return mock stats
    # AAPL returns 5 confirmed, 2 failed
    # MSFT returns 3 confirmed, 1 failed, 1 pending, 1 fire
    mock_stats = {
        "AAPL.csv": {
            "success": True,
            "metadata": {"symbol": "AAPL", "path": Path("AAPL_VSA.xlsx"), "is_trending": False, "is_effort": False, "is_ticker": False, "is_trigger": False, "is_anomaly": False, "vol_pct": 0.0, "vsa_signal": "No Signal", "vsa_confidence": 0.0},
            "stats": {"conf": 5, "fail": 2, "pend": 0, "fire": 0, "total_signals": 7}
        },
        "MSFT.csv": {
            "success": True,
            "metadata": {"symbol": "MSFT", "path": Path("MSFT_VSA.xlsx"), "is_trending": False, "is_effort": False, "is_ticker": False, "is_trigger": False, "is_anomaly": False, "vol_pct": 0.0, "vsa_signal": "No Signal", "vsa_confidence": 0.0},
            "stats": {"conf": 3, "fail": 1, "pend": 1, "fire": 1, "total_signals": 5}
        }
    }
    
    def mock_process_file(self, file_path):
        return mock_stats[file_path.name]
        
    monkeypatch.setattr(VSAProcessorService, "process_file", mock_process_file)
    monkeypatch.setattr(VSAProcessorService, "finalize_run", lambda self: None)
    
    # Run the stage
    success = orchestrator.run_stage_vsa_and_ete()
    
    assert success is True
    vsa_stats = orchestrator.stats["vsa_processor"]
    
    # Assert counts aggregated correctly
    assert vsa_stats["success_files"] == 2
    assert vsa_stats["confirmed"] == 8
    assert vsa_stats["failed"] == 3
    assert vsa_stats["pending"] == 1
    assert vsa_stats["fire"] == 1
    assert vsa_stats["total_signals"] == 12
