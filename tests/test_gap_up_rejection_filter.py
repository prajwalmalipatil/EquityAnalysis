"""
test_gap_up_rejection_filter.py
Unit tests for the Gap-Up Rejection watch: detection, tracking window, publishing and email rendering.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.constants import vsa_constants as const
from src.services.reporting.data_aggregator import DataAggregator
from src.services.reporting.gap_up_rejection_email_renderer import (
    EMPTY_MESSAGE,
    count_new,
    render_gap_up_rejection_html,
    render_gap_up_rejection_text,
)
from src.services.reporting.json_publisher import JSONPublisher
from src.services.vsa.gap_up_rejection_filter_service import TRACK_SESSIONS, GapUpRejectionFilterService


def _bar(day, open_, high, low, close, volume):
    return {"Date": f"2026-06-{day:02d}", "Open": open_, "High": high, "Low": low, "Close": close,
            "Volume": volume, "Spread": high - low}


# T-1, T (Gap-Up Continuation: gap above 100, green, volume and spread expand), T+1 (opens above 106, closes red)
PREV = _bar(1, 98.0, 101.0, 96.0, 100.0, 1000)
SIGNAL = _bar(2, 102.0, 107.0, 99.0, 106.0, 2000)
TRIGGER = _bar(3, 108.0, 109.0, 103.0, 104.0, 1500)


def _frame(*bars):
    return pd.DataFrame(list(bars))


def _later(count, close=102.0):
    """Quiet sessions after the trigger that cannot start a new setup."""
    return [_bar(4 + i, close, close + 1.0, close - 1.0, close, 100) for i in range(count)]


def _classify(*bars):
    return GapUpRejectionFilterService.classify_frame("TEST", _frame(*bars))


def _write_results(base_dir: Path, symbol: str, bars: list) -> None:
    target = base_dir / const.RESULTS_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target / f"{symbol}_VSA.xlsx") as writer:
        pd.DataFrame(bars).to_excel(writer, sheet_name="VSA_Analysis", index=False)


class TestGapUpRejectionDetection(unittest.TestCase):
    def test_trigger_day_is_reported_as_new(self):
        result = _classify(PREV, SIGNAL, TRIGGER)
        self.assertEqual((result.status, result.sessions_since_trigger), ("New", 0))
        self.assertEqual((result.signal_date, result.trigger_date), ("2026-06-02", "2026-06-03"))
        self.assertAlmostEqual(result.gap_pct, (108.0 - 106.0) / 106.0 * 100)
        self.assertAlmostEqual(result.trigger_change_pct, (104.0 - 108.0) / 108.0 * 100)
        self.assertEqual((result.return_pct, result.max_gain_pct, result.max_drawdown_pct), (0.0, 0.0, 0.0))
        self.assertFalse(result.above_signal_close)

    def test_green_close_on_t1_is_not_a_rejection(self):
        self.assertIsNone(_classify(PREV, SIGNAL, _bar(3, 108.0, 111.0, 107.0, 110.0, 1500)))

    def test_lower_open_on_t1_is_not_a_rejection(self):
        self.assertIsNone(_classify(PREV, SIGNAL, _bar(3, 105.0, 105.5, 102.0, 103.0, 1500)))

    def test_requires_gap_up_continuation_on_t(self):
        weak_volume = dict(SIGNAL, Volume=900)
        self.assertIsNone(_classify(PREV, weak_volume, TRIGGER))
        gap_down = _bar(2, 98.0, 99.0, 91.0, 94.0, 2000)
        self.assertIsNone(_classify(PREV, gap_down, _bar(3, 95.0, 96.0, 92.0, 93.0, 1500)))

    def test_too_few_rows_returns_none(self):
        self.assertIsNone(_classify(SIGNAL, TRIGGER))

    def test_tracking_measures_from_the_trigger_close(self):
        later = [_bar(4, 104.0, 106.08, 101.0, 102.0, 100), _bar(5, 102.0, 103.0, 98.8, 109.2, 100)]
        result = _classify(PREV, SIGNAL, TRIGGER, *later)
        self.assertEqual((result.status, result.sessions_since_trigger), ("Tracking", 2))
        self.assertAlmostEqual(result.return_pct, 5.0)
        self.assertAlmostEqual(result.max_gain_pct, 2.0)
        self.assertAlmostEqual(result.max_drawdown_pct, -5.0)
        self.assertTrue(result.above_signal_close)

    def test_setup_expires_after_the_tracking_window(self):
        self.assertEqual(_classify(PREV, SIGNAL, TRIGGER, *_later(TRACK_SESSIONS)).sessions_since_trigger, TRACK_SESSIONS)
        self.assertIsNone(_classify(PREV, SIGNAL, TRIGGER, *_later(TRACK_SESSIONS + 1)))


class TestGapUpRejectionPipeline(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(tempfile.mkdtemp())
        # All three histories end on 2026-06-06; FADECO's setup is shifted three sessions later.
        shifted = [dict(bar, Date=f"2026-06-{4 + i:02d}") for i, bar in enumerate([PREV, SIGNAL, TRIGGER])]
        _write_results(self.base_dir, "FADECO", shifted)
        _write_results(self.base_dir, "OLDCO", [PREV, SIGNAL, TRIGGER] + _later(3))
        _write_results(self.base_dir, "QUIETCO", [PREV, SIGNAL] + _later(3))

    def tearDown(self):
        shutil.rmtree(self.base_dir)

    def test_scan_copies_only_qualifying_files(self):
        results = GapUpRejectionFilterService(self.base_dir).scan_and_classify()
        self.assertEqual({r.symbol: r.status for r in results}, {"FADECO": "New", "OLDCO": "Tracking"})
        copied = sorted(p.name for p in (self.base_dir / const.GAP_UP_REJECTION_FILTER_DIR_NAME).glob("*.xlsx"))
        self.assertEqual(copied, ["FADECO_VSA.xlsx", "OLDCO_VSA.xlsx"])

    def test_scan_skips_stocks_whose_history_is_stale(self):
        _write_results(self.base_dir, "FRESHCO", [PREV, SIGNAL, TRIGGER] + _later(5))
        results = GapUpRejectionFilterService(self.base_dir).scan_and_classify()
        self.assertEqual([r.symbol for r in results], ["FRESHCO"])

    def test_scan_without_results_dir_returns_empty(self):
        empty_base = self.base_dir / "empty"
        empty_base.mkdir()
        self.assertEqual(GapUpRejectionFilterService(empty_base).scan_and_classify(), [])

    def test_publisher_emits_gap_up_rejection_filters(self):
        GapUpRejectionFilterService(self.base_dir).scan_and_classify()
        self.assertEqual(DataAggregator(self.base_dir).aggregate_pipeline_stats()["gap_up_rejection"], 2)
        output_file = self.base_dir / "dashboard" / "data.json"
        JSONPublisher(self.base_dir, output_file).publish()
        by_symbol = {d["symbol"]: d for d in json.loads(output_file.read_text())["gap_up_rejection_filters"]["daily"]}
        self.assertEqual(set(by_symbol), {"FADECO", "OLDCO"})
        self.assertEqual(by_symbol["OLDCO"]["sessions_since_trigger"], 3)
        self.assertEqual(by_symbol["OLDCO"]["track_sessions"], TRACK_SESSIONS)
        self.assertEqual(by_symbol["FADECO"]["gap_pct"], 1.89)


class TestGapUpRejectionEmail(unittest.TestCase):
    ITEMS = {"daily": [
        {"symbol": "M&M<b>", "status": "New", "sessions_since_trigger": 0, "track_sessions": 10,
         "trigger_date": "2026-06-03", "signal_close": 106.0, "trigger_open": 108.0, "trigger_close": 104.0,
         "gap_pct": 1.89, "trigger_change_pct": -3.7, "last_close": 104.0, "return_pct": 0.0,
         "max_gain_pct": 0.0, "max_drawdown_pct": 0.0, "above_signal_close": False},
        {"symbol": "OLDCO", "status": "Tracking", "sessions_since_trigger": 4, "track_sessions": 10,
         "trigger_date": "2026-05-28", "signal_close": 50.0, "trigger_open": 51.0, "trigger_close": 49.0,
         "gap_pct": 2.0, "trigger_change_pct": -3.92, "last_close": 47.5, "return_pct": -3.06,
         "max_gain_pct": 1.2, "max_drawdown_pct": None, "above_signal_close": False},
    ]}

    def test_html_splits_new_and_tracking_and_escapes_symbols(self):
        html = render_gap_up_rejection_html(self.ITEMS)
        self.assertIn("<strong>1</strong> new today, 1 still tracking", html)
        self.assertIn("M&amp;M&lt;b&gt;", html)
        self.assertNotIn("M&M<b>", html)
        self.assertIn("4/10", html)
        self.assertIn("-3.06%", html)

    def test_text_lists_rows_with_missing_values(self):
        text = render_gap_up_rejection_text(self.ITEMS)
        self.assertIn("New today (1)", text)
        self.assertIn("Still tracking (1)", text)
        self.assertIn("Below", text)
        self.assertIn("--", text)

    def test_absent_block_renders_nothing(self):
        self.assertEqual(render_gap_up_rejection_html(None), "")
        self.assertEqual(render_gap_up_rejection_text(None), "")
        self.assertEqual(count_new(None), 0)

    def test_empty_block_states_no_setups(self):
        self.assertIn(EMPTY_MESSAGE, render_gap_up_rejection_html({"daily": []}))
        self.assertIn(EMPTY_MESSAGE, render_gap_up_rejection_text({"daily": []}))

    def test_count_new_ignores_tracking_rows(self):
        self.assertEqual(count_new(self.ITEMS), 1)


if __name__ == "__main__":
    unittest.main()
