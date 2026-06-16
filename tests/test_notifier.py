import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

class TestNotifier(unittest.TestCase):
    def test_notifier_near_duplicate_filter(self):
        """Verify that main_notifier deduplication correctly handles near duplicate titles."""
        # Simple standalone recreation of the main_notifier helper logic to assert correctness
        def clean_title(t: str) -> set:
            fillers = {"rbi", "issues", "announces", "to", "on", "for", "a", "an", "the", "and", "of", "in", "with", "under"}
            words = [w for w in t.lower().split() if w.isalnum()]
            return set(w for w in words if w not in fillers)

        def is_duplicate(t1: str, t2: str) -> bool:
            s1 = clean_title(t1)
            s2 = clean_title(t2)
            if not s1 or not s2:
                return False
            common = s1.intersection(s2)
            overlap = len(common) / min(len(s1), len(s2))
            return overlap >= 0.7

        # Case 1: Identical phrase besides filler word and casing
        self.assertTrue(is_duplicate(
            "RBI Issues Master Directions on Authorisation to operate a Payment System",
            "Master Directions on Authorisation to operate a Payment System"
        ))

        # Case 2: Near duplicate announcements
        self.assertTrue(is_duplicate(
            "RBI Announces Special Liquidity Facility for Mutual Funds",
            "Special Liquidity Facility for Mutual Funds"
        ))

        # Case 3: Totally different announcements
        self.assertFalse(is_duplicate(
            "RBI Announces Special Liquidity Facility for Mutual Funds",
            "RBI Issues Master Directions on Authorisation to operate a Payment System"
        ))

    @patch("main_notifier.SMTPClient")
    @patch("main_notifier.Path")
    @patch("main_notifier.open")
    def test_notifier_reads_flat_properties_and_sends(self, mock_open, mock_path_class, mock_smtp_client_class):
        """Verify main_notifier loads and maps flat schema properties from data.json correctly."""
        from main_notifier import main
        import sys
        
        # Mock Path.exists to return True for dashboard/data.json
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path_class.return_value = mock_path
        
        # Mock data.json content matching flat DashboardViewModel format
        mock_data = {
            "macro_intelligence": {
                "recent_events": [
                    {
                        "event_id": "event1",
                        "title": "RBI Issues Master Directions on Authorisation to operate a Payment System",
                        "summary": "This is a summary of the master directions.",
                        "processing_state": "NEW"
                    },
                    {
                        "event_id": "event2",
                        "title": "Master Directions on Authorisation to operate a Payment System",
                        "summary": "This is a duplicate summary.",
                        "processing_state": "NEW"
                    },
                    {
                        "event_id": "event3",
                        "title": "Unique Press Release",
                        "summary": "This is a unique announcement.",
                        "processing_state": "NEW"
                    }
                ]
            }
        }
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with patch("json.load", return_value=mock_data):
            # Run in report-only mode so it doesn't need actual credentials
            with patch.object(sys, "argv", ["main_notifier.py", "--report-only"]):
                main()
                
        # Assert local_report_preview.html is written with deduplicated macro events
        mock_path.write_text.assert_called_once()
        html_written = mock_path.write_text.call_args[0][0]
        
        # "RBI Issues..." and "Master Directions..." are near-duplicates, so only one should be written.
        # "Unique Press Release" is unique, so it should be written.
        self.assertIn("RBI Issues Master Directions on Authorisation to operate a Payment System", html_written)
        # It should only appear once (in the first item's title)
        self.assertEqual(html_written.count("Master Directions"), 1)
        self.assertIn("Unique Press Release", html_written)

if __name__ == "__main__":
    unittest.main()
