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
