"""
extraction_models.py
Models for the extraction service.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class ExtractionRequest:
    symbol: str
    from_date: str
    to_date: str
    output_dir: Path

@dataclass
class ExtractionResponse:
    symbol: str
    success: bool
    file_path: Optional[Path] = None
    error: Optional[str] = None
    attempts: int = 0
