import sys
import unittest
from unittest.mock import MagicMock, patch

from main_notifier import UNAVAILABLE_MESSAGE, build_email, main, session_date

GAP_MOMENTUM = {"daily": [
    {"symbol": "UPCO", "sentiment": "Bullish", "signal_date": "2026-06-02", "t1_close": 100.0,
     "t_open": 102.0, "t_close": 106.0, "gap_pct": 2.0, "change_pct": 6.0,
     "volume_pct": 100.0, "spread_pct": 60.0},
    {"symbol": "DOWNCO", "sentiment": "Bearish", "signal_date": "2026-06-02", "t1_close": 100.0,
     "t_open": 98.0, "t_close": 94.0, "gap_pct": -2.0, "change_pct": -6.0,
     "volume_pct": 250.0, "spread_pct": 40.0},
]}
# Sections the email used to carry; none of them may appear any more.
OTHER_SECTIONS = {
    "macro_intelligence": {"recent_events": [{"title": "Unique Press Release", "summary": "x.", "processing_state": "NEW"}]},
    "volume_trap_filters": {"daily": [{"symbol": "TRAPCO", "sentiment": "Bullish"}], "weekly": [], "monthly": []},
    "age_again_filters": {"daily": [{"symbol": "AGECO", "sentiment": "Bearish"}], "weekly": [], "monthly": []},
}


class TestNotifier(unittest.TestCase):
    def test_email_contains_only_gap_momentum(self):
        subject, html, text = build_email({**OTHER_SECTIONS, "gap_momentum_filters": GAP_MOMENTUM}, "2026-06-03")
        self.assertEqual(subject, "Gap Momentum Analysis - 2026-06-02 | 2 signals | 0 new expansions | 0 new rejections | V² Money")
        for body in (html, text):
            self.assertIn("UPCO", body)
            self.assertIn("DOWNCO", body)
            self.assertIn("Session 2026-06-02", body)
            for removed in ("Unique Press Release", "TRAPCO", "AGECO", "Macro", "Volume Trap", "AgeAgain", "Dashboard"):
                self.assertNotIn(removed, body)

    def test_no_signals_still_reports_the_session(self):
        subject, html, text = build_email({"gap_momentum_filters": {"daily": []}}, "2026-06-03")
        self.assertEqual(subject, "Gap Momentum Analysis - 2026-06-03 | 0 signals | 0 new expansions | 0 new rejections | V² Money")
        self.assertIn("No stocks met the criteria", html)
        self.assertIn("No stocks met the criteria", text)

    def test_missing_block_is_stated_not_hidden(self):
        _, html, text = build_email({}, "2026-06-03")
        self.assertIn(UNAVAILABLE_MESSAGE, html)
        self.assertIn(UNAVAILABLE_MESSAGE, text)

    def test_email_adds_gap_up_rejection_watch_after_gap_momentum(self):
        rejections = {"daily": [{"symbol": "FADECO", "status": "New", "sessions_since_trigger": 0, "gap_pct": 1.0}]}
        subject, html, text = build_email(
            {"gap_momentum_filters": GAP_MOMENTUM, "gap_up_rejection_filters": rejections}, "2026-06-03")
        self.assertIn("| 1 new rejection |", subject)
        for body in (html, text):
            self.assertIn("Gap-Up Rejection Watch", body)
            self.assertLess(body.index("UPCO"), body.index("FADECO"))

    def test_sections_are_ordered_momentum_expansion_rejection(self):
        expansions = {"daily": [{"symbol": "WIDECO", "status": "New", "sessions_since_trigger": 0, "spread_ratio": 1.4}]}
        rejections = {"daily": [{"symbol": "FADECO", "status": "New", "sessions_since_trigger": 0, "gap_pct": 1.0}]}
        subject, html, text = build_email({"gap_momentum_filters": GAP_MOMENTUM, "gap_up_expansion_filters": expansions,
                                           "gap_up_rejection_filters": rejections}, "2026-06-03")
        self.assertIn("| 1 new expansion | 1 new rejection |", subject)
        for body in (html, text):
            self.assertIn("Gap-Up Expansion Watch", body)
            self.assertLess(body.index("UPCO"), body.index("WIDECO"))
            self.assertLess(body.index("WIDECO"), body.index("FADECO"))

    def test_session_date_uses_latest_signal_date(self):
        self.assertEqual(session_date(GAP_MOMENTUM, "2026-06-03"), "2026-06-02")
        self.assertEqual(session_date(None, "2026-06-03"), "2026-06-03")

    @patch("main_notifier.Path")
    @patch("main_notifier.load_dashboard_data", return_value={"gap_momentum_filters": GAP_MOMENTUM})
    def test_report_only_writes_preview_without_sending(self, _mock_load, mock_path_class):
        mock_path = MagicMock()
        mock_path_class.return_value = mock_path
        with patch("main_notifier.SMTPClient") as mock_smtp, patch.object(sys, "argv", ["main_notifier.py", "--report-only"]):
            main()
        mock_smtp.assert_not_called()
        self.assertIn("UPCO", mock_path.write_text.call_args[0][0])

    @patch.dict("os.environ", {"SENDER_EMAIL": "sender@example.com", "SENDER_PASSWORD": "test-password"})
    @patch("main_notifier.load_dashboard_data", return_value={"gap_momentum_filters": GAP_MOMENTUM})
    def test_send_passes_gap_momentum_subject_and_bodies(self, _mock_load):
        with patch("main_notifier.SMTPClient") as mock_smtp, \
                patch.object(sys, "argv", ["main_notifier.py", "--to", "reader@example.com"]):
            mock_smtp.return_value.send_email.return_value = True
            main()
        kwargs = mock_smtp.return_value.send_email.call_args.kwargs
        self.assertEqual(kwargs["recipient_email"], "reader@example.com")
        self.assertTrue(kwargs["subject"].startswith("Gap Momentum Analysis - 2026-06-02"))
        self.assertIn("UPCO", kwargs["html_body"])
        self.assertIn("DOWNCO", kwargs["text_body"])


if __name__ == "__main__":
    unittest.main()
