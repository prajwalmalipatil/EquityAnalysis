import os
import hashlib
import requests
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.config import StorageConfig
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("attachment-processor")

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv"
}
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".csv", ".xls"}
MAX_ATTACHMENT_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


class AttachmentProcessor:
    """
    Downloads, verifies, and extracts attachments (e.g. PDFs) for Macro Events.
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.attachments_dir = self.config.attachments_dir
        self.session = requests.Session()

    def process(self, event: MacroEvent) -> MacroEvent:
        """Process all attachments for a given event."""
        if not event.official_data.pdf_url and not event.official_data.attachments:
            return event
            
        # Process the main PDF URL if it exists
        if event.official_data.pdf_url:
            local_path, file_hash = self._download_attachment(event.official_data.pdf_url, event.event_id)
            if local_path:
                logger.info("ATTACHMENT_DOWNLOADED", extra={"event_id": event.event_id, "hash": file_hash})
                # Add to attachments list
                if local_path not in event.official_data.attachments:
                    event.official_data.attachments.append(local_path)
                    
        return event

    def _validate_download(self, response: requests.Response, url: str) -> bool:
        """Guard clause: validates content-type and size before saving."""
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError(f"Forbidden Content-Type: {content_type}")
            
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
            except (ValueError, TypeError):
                size = None
            if size is not None and size > MAX_ATTACHMENT_SIZE_BYTES:
                raise ValueError(f"File size {size} exceeds limit of {MAX_ATTACHMENT_SIZE_BYTES} bytes")
                
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Forbidden file extension: {ext}")
            
        return True

    def _download_attachment(self, url: str, event_id: str) -> tuple[Optional[str], Optional[str]]:
        """Downloads the file and returns (local_relative_path, sha256_hash)."""
        local_filepath = None
        try:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"{event_id}_attachment.pdf"
                
            local_filepath = self.attachments_dir / filename
            
            # Avoid re-downloading if it exists
            if local_filepath.exists():
                logger.info("ATTACHMENT_ALREADY_EXISTS", extra={"filepath": str(local_filepath)})
                return str(local_filepath), self._hash_file(local_filepath)
                
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Run validation
            self._validate_download(response, url)
            
            total_bytes = 0
            with open(local_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_ATTACHMENT_SIZE_BYTES:
                        raise ValueError(f"File size exceeds limit of {MAX_ATTACHMENT_SIZE_BYTES} bytes during streaming")
                    f.write(chunk)
                    
            file_hash = self._hash_file(local_filepath)
            return str(local_filepath), file_hash
            
        except (requests.RequestException, OSError, ValueError) as e:
            # Clean up if partial file was written
            if local_filepath is not None and local_filepath.exists():
                local_filepath.unlink(missing_ok=True)
            logger.error("ATTACHMENT_DOWNLOAD_FAILED", extra={"url": url, "error": str(e)})
            return None, None

    def _hash_file(self, filepath: Path) -> str:
        """Generates a SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
