"""
dependencies.py
Centralized Dependency Injection (DI) factory.
Handles instantiation and lifecycle of clients and services.
"""

from .clients.nse_client import NSEClient
from .services.extraction_service import ExtractionService

from .services.vsa.processor_service import VSAProcessorService
from .services.vsa.eigen_filter_service import EigenFilterService
from .services.reporting.data_aggregator import DataAggregator
from .services.reporting.html_renderer import HTMLRenderer
from .clients.smtp_client import SMTPClient
from pathlib import Path

def get_nse_client(use_selenium: bool = True, headless: bool = True) -> NSEClient:
    """Returns a managed instance of NSEClient."""
    return NSEClient(use_selenium=use_selenium, headless=headless)

def get_extraction_service(nse_client: NSEClient = None) -> ExtractionService:
    """Provides an ExtractionService with its dependencies injected."""
    client = nse_client or get_nse_client()
    return ExtractionService(nse_client=client)

def get_vsa_processor_service(output_base: Path) -> VSAProcessorService:
    """Provides a VSAProcessorService."""
    return VSAProcessorService(output_base=output_base)

def get_eigen_filter_service(base_dir: Path) -> EigenFilterService:
    """Provides an EigenFilterService for post-VSA divergence classification."""
    return EigenFilterService(base_dir=base_dir)

def get_reporting_service(base_dir: Path):
    """Returns a tuple of (Aggregator, Renderer, SMTPClient) for reporting."""
    aggregator = DataAggregator(base_dir)
    renderer = HTMLRenderer()
    smtp = SMTPClient()
    return aggregator, renderer, smtp
