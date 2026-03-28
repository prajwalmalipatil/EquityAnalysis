"""
test_extraction_service.py
Unit tests for the ExtractionService with mocked infrastructure.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import shutil

from src.clients.nse_client import NSEClient
from src.services.extraction_service import ExtractionService
from src.models.extraction_models import ExtractionRequest

class TestExtractionService(unittest.TestCase):
    def setUp(self):
        # Create a mock client
        self.mock_client = MagicMock(spec=NSEClient)
        self.service = ExtractionService(self.mock_client)
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # Cleanup test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_validate_symbols(self):
        """Verify symbol cleaning and validation."""
        raw = "INFY, TCS TCS TCS TCS TCS,, RELIANCE!@# TCS"
        cleaned = self.service.validate_symbols(raw)
        
        # Should deduplicate and clean
        self.assertIn("INFY", cleaned)
        self.assertIn("TCS", cleaned)
        self.assertIn("RELIANCE", cleaned)
        self.assertEqual(len(cleaned), 3)

    def test_extract_symbol_data_success(self):
        """Verify successful extraction and atomic write."""
        # Mock successful response
        mock_resp = MagicMock()
        mock_resp.content = b"Date,Open,High,Low,Close,Volume\n2026-03-27,100,110,90,105,1000"
        self.mock_client.fetch_historical_data.return_value = mock_resp
        
        req = ExtractionRequest(
            symbol="TEST",
            from_date="01-01-2026",
            to_date="01-01-2027",
            output_dir=self.test_dir
        )
        
        result = self.service.extract_symbol_data(req)
        
        self.assertTrue(result.success)
        self.assertTrue(result.file_path.exists())
        self.assertEqual(result.file_path.read_bytes(), b"Date,Open,High,Low,Close,Volume\n2026-03-27,100,110,90,105,1000")

    def test_extract_symbol_data_failure(self):
        """Verify handling of extraction errors."""
        self.mock_client.fetch_historical_data.side_effect = Exception("API_ERROR")
        
        req = ExtractionRequest(
            symbol="FAIL",
            from_date="01-01-2026",
            to_date="01-01-2027",
            output_dir=self.test_dir
        )
        
        result = self.service.extract_symbol_data(req)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "API_ERROR")

if __name__ == "__main__":
    unittest.main()
