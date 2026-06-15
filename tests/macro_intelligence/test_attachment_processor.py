import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.services.macro_intelligence.attachment_processor import AttachmentProcessor
from src.services.macro_intelligence.config import StorageConfig
from src.services.macro_intelligence.models import MacroEvent, OfficialData

@pytest.fixture
def storage_config(tmp_path):
    storage = StorageConfig(
        base_path=tmp_path,
        history_file=tmp_path / "rbi_events.jsonl",
        attachments_dir=tmp_path / "attachments",
        metadata_dir=tmp_path / "metadata",
        logs_dir=tmp_path / "logs",
        cache_dir=tmp_path / "cache",
        state_file=tmp_path / "collector_state.json"
    )
    storage.attachments_dir.mkdir(parents=True, exist_ok=True)
    return storage

def test_validate_download_allowed(storage_config):
    processor = AttachmentProcessor(storage_config)
    
    # Mock response
    response = MagicMock()
    response.headers = {
        "Content-Type": "application/pdf",
        "Content-Length": "1024"
    }
    
    assert processor._validate_download(response, "http://example.com/test.pdf") is True

def test_validate_download_forbidden_content_type(storage_config):
    processor = AttachmentProcessor(storage_config)
    
    response = MagicMock()
    response.headers = {
        "Content-Type": "text/html",
        "Content-Length": "1024"
    }
    
    with pytest.raises(ValueError, match="Forbidden Content-Type"):
        processor._validate_download(response, "http://example.com/test.pdf")

def test_validate_download_forbidden_extension(storage_config):
    processor = AttachmentProcessor(storage_config)
    
    response = MagicMock()
    response.headers = {
        "Content-Type": "application/pdf",
        "Content-Length": "1024"
    }
    
    # PDF Content type, but extension is .exe
    with pytest.raises(ValueError, match="Forbidden file extension"):
        processor._validate_download(response, "http://example.com/test.exe")

def test_validate_download_size_limit_header(storage_config):
    processor = AttachmentProcessor(storage_config)
    
    response = MagicMock()
    response.headers = {
        "Content-Type": "application/pdf",
        "Content-Length": str(60 * 1024 * 1024)  # 60MB, limit is 50MB
    }
    
    with pytest.raises(ValueError, match="exceeds limit"):
        processor._validate_download(response, "http://example.com/test.pdf")

def test_download_size_limit_streaming(storage_config, monkeypatch):
    processor = AttachmentProcessor(storage_config)
    
    # Mock session.get
    mock_response = MagicMock()
    mock_response.headers = {
        "Content-Type": "application/pdf"
        # Content-Length is missing to trigger streaming size check
    }
    
    # Generate 6 chunks of 10MB each (total 60MB, exceeding 50MB limit)
    mock_response.iter_content = lambda chunk_size: [b"A" * (10 * 1024 * 1024)] * 6
    
    monkeypatch.setattr(processor.session, "get", lambda url, stream, timeout: mock_response)
    
    local_path, file_hash = processor._download_attachment("http://example.com/test.pdf", "evt-1")
    
    assert local_path is None
    assert file_hash is None
    
    # Ensure temporary/partial file was deleted
    attachments_dir = storage_config.attachments_dir
    attachments_files = list(attachments_dir.glob("*"))
    assert len(attachments_files) == 0
