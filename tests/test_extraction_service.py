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
        raw = "INFY, TCS, INFY,, RELIANCE!@#, NIFTY 50"
        cleaned = self.service.validate_symbols(raw)
        
        # Should deduplicate, preserve spaces for indices, and discard invalid format symbols
        self.assertIn("INFY", cleaned)
        self.assertIn("TCS", cleaned)
        self.assertIn("NIFTY 50", cleaned)
        self.assertNotIn("RELIANCE!@#", cleaned)
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

    def test_parse_yahoo_chart_json_to_nse_json(self):
        """Verify that Yahoo Finance Chart JSON is correctly parsed to NSE JSON format."""
        from src.clients.nse_client import parse_yahoo_chart_json_to_nse_json
        
        mock_json = {
            "chart": {
                "result": [{
                    "timestamp": [1750098600],
                    "indicators": {
                        "quote": [{
                            "open": [100.0],
                            "high": [105.0],
                            "low": [95.0],
                            "close": [102.0],
                            "volume": [10000]
                        }]
                    }
                }]
            }
        }
        
        parsed = parse_yahoo_chart_json_to_nse_json(mock_json)
        
        self.assertIn("data", parsed)
        self.assertEqual(len(parsed["data"]), 1)
        
        first_row = parsed["data"][0]
        # 1750098600 is 2025-06-17. Let's make sure it matches the converted local format or date format.
        # Since we use datetime.fromtimestamp, we can assert its formatting contains the date components.
        # We can dynamically construct the expected date for the test environment's timezone:
        from datetime import datetime
        expected_date = datetime.fromtimestamp(1750098600).strftime("%d-%m-%Y")
        
        self.assertEqual(first_row["date"], expected_date)
        self.assertEqual(first_row["open"], "100.0")
        self.assertEqual(first_row["high"], "105.0")
        self.assertEqual(first_row["low"], "95.0")
        self.assertEqual(first_row["close"], "102.0")
        self.assertEqual(first_row["sharesTraded"], "10000")
        self.assertEqual(first_row["turnoverInCr"], "0")

    @patch("requests.Session")
    def test_fetch_historical_index_data_yahoo_fallback(self, mock_session_class):
        """Verify fetch_historical_index_data falls back to Yahoo Finance Chart API on NSE failure."""
        from src.clients.nse_client import NSEClient
        
        mock_session = mock_session_class.return_value
        
        # 1. Setup mock responses
        # First request to NSE fails (raises exception or status code 503)
        mock_nse_resp = MagicMock()
        mock_nse_resp.raise_for_status.side_effect = Exception("NSE HTTP 503")
        
        # Second request to Yahoo Finance succeeds
        mock_yahoo_resp = MagicMock()
        mock_yahoo_resp.status_code = 200
        
        mock_json_response = {
            "chart": {
                "result": [{
                    "timestamp": [1750098600],
                    "indicators": {
                        "quote": [{
                            "open": [100.0],
                            "high": [105.0],
                            "low": [95.0],
                            "close": [102.0],
                            "volume": [10000]
                        }]
                    }
                }]
            }
        }
        mock_yahoo_resp.json.return_value = mock_json_response
        
        mock_session.get.side_effect = [mock_nse_resp, mock_yahoo_resp]
        
        # Create client without selenium cookie warmup to avoid browser mocks
        client = NSEClient(use_selenium=False)
        client.session = mock_session
        
        resp = client.fetch_historical_index_data("NIFTY 50", "15-06-2026", "16-06-2026")
        
        self.assertEqual(resp.status_code, 200)
        resp_data = resp.json()
        self.assertEqual(len(resp_data["data"]), 1)
        
        from datetime import datetime
        expected_date = datetime.fromtimestamp(1750098600).strftime("%d-%m-%Y")
        self.assertEqual(resp_data["data"][0]["date"], expected_date)
        self.assertEqual(resp_data["data"][0]["close"], "102.0")

    @patch("requests.Session")
    def test_fetch_historical_data_yahoo_fallback(self, mock_session_class):
        """Verify fetch_historical_data falls back to Yahoo Finance Chart API on stock extraction failure."""
        from src.clients.nse_client import NSEClient
        
        mock_session = mock_session_class.return_value
        
        # 1. Setup mock responses
        # First request to NSE fails (raises exception or status code 503)
        mock_nse_resp = MagicMock()
        mock_nse_resp.raise_for_status.side_effect = Exception("NSE HTTP 503")
        
        # Second request to Yahoo Finance succeeds
        mock_yahoo_resp = MagicMock()
        mock_yahoo_resp.status_code = 200
        
        mock_json_response = {
            "chart": {
                "result": [{
                    "timestamp": [1750098600],
                    "indicators": {
                        "quote": [{
                            "open": [100.0],
                            "high": [105.0],
                            "low": [95.0],
                            "close": [102.0],
                            "volume": [10000]
                        }]
                    }
                }]
            }
        }
        mock_yahoo_resp.json.return_value = mock_json_response
        
        mock_session.get.side_effect = [mock_nse_resp, mock_yahoo_resp]
        
        # Create client without selenium cookie warmup to avoid browser mocks
        client = NSEClient(use_selenium=False)
        client.session = mock_session
        
        resp = client.fetch_historical_data("VEDL", "15-06-2026", "16-06-2026")
        
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Date,Open,High,Low,Close,Volume", resp.text)
        
        # Check that the data rows are correct
        rows = resp.text.strip().split("\n")
        self.assertEqual(len(rows), 2)
        
        from datetime import datetime
        expected_date = datetime.fromtimestamp(1750098600).strftime("%Y-%m-%d")
        self.assertEqual(rows[1], f"{expected_date},100.0,105.0,95.0,102.0,10000")

if __name__ == "__main__":
    unittest.main()
