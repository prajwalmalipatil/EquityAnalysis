"""
test_processor_service.py
Unit tests and integration tests for the refactored VSA Processor Service components.
"""

import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
import pandas as pd

from src.services.vsa.excel_report_generator import VSAExcelReportGenerator
from src.services.vsa.file_loader import VSAFileLoader
from src.services.vsa.indicators_enricher import VSAIndicatorsEnricher
from src.services.vsa.pattern_router_service import VSAPatternRouter
from src.services.vsa.processor_service import VSAProcessorService
from src.services.vsa.signal_applier import VSASignalApplier


class TestProcessorService(unittest.TestCase):
    def test_file_loader_empty_file(self):
        """Verify that file loader returns empty dataframe for invalid formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "empty.csv"
            file_path.write_text("invalid csv format")
            df = VSAFileLoader.load_and_clean(file_path)
            self.assertTrue(df.empty)

    def test_indicators_enricher_and_signal_applier(self):
        """Verify indicator calculations and signal applications on mock data."""
        # Create a mock dataframe mimicking typical cleaned NSE format
        data = {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "Low": [95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
            "Date": pd.date_range("2023-01-01", periods=6),
        }
        df = pd.DataFrame(data)

        # Enrich indicators
        df_enriched = VSAIndicatorsEnricher.enrich(df.copy())
        self.assertIn("Spread", df_enriched.columns)
        self.assertIn("Close_Position", df_enriched.columns)
        self.assertIn("Volume_MA", df_enriched.columns)

        # Apply signals
        df_signals = VSASignalApplier.apply_signals(df_enriched.copy())
        self.assertIn("Signal_Type", df_signals.columns)
        self.assertIn("Validation_Status", df_signals.columns)
        self.assertIn("Anomaly_V2", df_signals.columns)

    def test_excel_generator_and_router_integration(self):
        """Verify end-to-end integration and run finalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Write a mock valid input CSV
            csv_path = tmp_path / "TICKER.csv"
            # 25 rows of data to satisfy indicators like MA(20) and shift(1)
            dates = pd.date_range("2023-01-01", periods=25)
            rows = []
            for i in range(25):
                rows.append(f"{dates[i].strftime('%Y-%m-%d')},100,105,95,102,{1000 + i*10}")
            csv_path.write_text("Date,Open,High,Low,Close,Volume\n" + "\n".join(rows) + "\n")

            # Instantiate service
            service = VSAProcessorService(output_base=tmp_path)

            # Process file
            res = service.process_file(csv_path)
            self.assertTrue(res["success"])
            self.assertEqual(res["metadata"]["symbol"], "TICKER")

            # Verify Excel file was generated
            excel_file = tmp_path / "Results" / "TICKER_VSA.xlsx"
            self.assertTrue(excel_file.exists())

            # Finalize run routing
            service._processed_metadata.append(res["metadata"])

            # Mock the ViewBuilder.publish to avoid creating actual UI output directories
            with mock.patch(
                "src.services.reporting.view_builder_service.ViewBuilderService.publish"
            ) as mock_publish:
                service.finalize_run()
                self.assertTrue(mock_publish.called)


if __name__ == "__main__":
    unittest.main()
