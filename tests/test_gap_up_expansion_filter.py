"""
test_gap_up_expansion_filter.py
Unit tests for the Gap-Up Expansion watch: detection, tracking window, publishing and email rendering.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.constants import vsa_constants as const
from src.services.reporting.data_aggregator import DataAggregator
from src.services.reporting.gap_up_expansion_email_renderer import (
    EMPTY_MESSAGE,
    count_new,
    render_gap_up_expansion_html,
    render_gap_up_expansion_text,
)
from src.services.reporting.json_publisher import JSONPublisher
from src.services.vsa.gap_up_expansion_filter_service import TRACK_SESSIONS, GapUpExpansionFilterService


def _bar(day, open_, high, low, close, volume):
    return {"Date": f"2026-06-{day:02d}", "Open": open_, "High": high, "Low": low, "Close": close,
            "Volume": volume, "Spread": high - low}


# T-1, T (Gap-Up Continuation, spread 8), T+1 (opens above 106, spread 10 > 8)
PREV = _bar(1, 98.0, 101.0, 96.0, 100.0, 1000)
SIGNAL = _bar(2, 102.0, 107.0, 99.0, 106.0, 2000)
GREEN_TRIGGER = _bar(3, 107.0, 115.0, 105.0, 113.0, 1500)
RED_TRIGGER = _bar(3, 108.0, 112.0, 102.0, 103.0, 1500)


def _later(count, close=110.0):
    """Quiet sessions after the trigger that cannot start a new setup."""
    return [_bar(4 + i, close, close + 1.0, close - 1.0, close, 100) for i in range(count)]


def _classify(*bars):
    return GapUpExpansionFilterService.classify_frame("TEST", pd.DataFrame(list(bars)))


def _write_results(base_dir: Path, symbol: str, bars: list) -> None:
    target = base_dir / const.RESULTS_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target / f"{symbol}_VSA.xlsx") as writer:
        pd.DataFrame(bars).to_excel(writer, sheet_name="VSA_Analysis", index=False)


class TestGapUpExpansionDetection(unittest.TestCase):
    def test_green_trigger_is_reported_as_new(self):
        result = _classify(PREV, SIGNAL, GREEN_TRIGGER)
        self.assertEqual((result.status, result.sessions_since_trigger, result.trigger_candle), ("New", 0, "Green"))
        self.assertEqual((result.signal_spread, result.trigger_spread), (8.0, 10.0))
        self.assertAlmostEqual(result.spread_ratio, 1.25)
        self.assertAlmostEqual(result.gap_pct, (107.0 - 106.0) / 106.0 * 100)
        self.assertTrue(result.above_signal_close)

    def test_red_trigger_still_qualifies(self):
        self.assertEqual(_classify(PREV, SIGNAL, RED_TRIGGER).trigger_candle, "Red")

    def test_equal_or_narrower_spread_does_not_qualify(self):
        self.assertIsNone(_classify(PREV, SIGNAL, _bar(3, 107.0, 112.0, 104.0, 110.0, 1500)))

    def test_lower_open_on_t1_does_not_qualify(self):
        self.assertIsNone(_classify(PREV, SIGNAL, _bar(3, 105.0, 115.0, 100.0, 112.0, 1500)))

    def test_requires_gap_up_continuation_on_t(self):
        self.assertIsNone(_classify(PREV, dict(SIGNAL, Volume=900), GREEN_TRIGGER))

    def test_spread_falls_back_to_high_minus_low(self):
        bars = [{k: v for k, v in bar.items()} for bar in (PREV, SIGNAL, GREEN_TRIGGER)]
        bars[2]["Spread"] = None
        self.assertAlmostEqual(_classify(*bars).trigger_spread, 10.0)

    def test_tracking_measures_from_the_trigger_close(self):
        later = [_bar(4, 113.0, 115.26, 110.0, 112.0, 100), _bar(5, 112.0, 113.0, 107.35, 118.65, 100)]
        result = _classify(PREV, SIGNAL, GREEN_TRIGGER, *later)
        self.assertEqual((result.status, result.sessions_since_trigger), ("Tracking", 2))
        self.assertAlmostEqual(result.return_pct, 5.0)
        self.assertAlmostEqual(result.max_gain_pct, 2.0)
        self.assertAlmostEqual(result.max_drawdown_pct, -5.0)

    def test_setup_expires_after_the_tracking_window(self):
        self.assertEqual(_classify(PREV, SIGNAL, GREEN_TRIGGER, *_later(TRACK_SESSIONS)).sessions_since_trigger,
                         TRACK_SESSIONS)
        self.assertIsNone(_classify(PREV, SIGNAL, GREEN_TRIGGER, *_later(TRACK_SESSIONS + 1)))


class TestGapUpExpansionPipeline(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(tempfile.mkdtemp())
        # All histories end on 2026-06-06.
        shifted = [dict(bar, Date=f"2026-06-{4 + i:02d}") for i, bar in enumerate([PREV, SIGNAL, GREEN_TRIGGER])]
        _write_results(self.base_dir, "WIDECO", shifted)
        _write_results(self.base_dir, "OLDCO", [PREV, SIGNAL, RED_TRIGGER] + _later(3))
        _write_results(self.base_dir, "QUIETCO", [PREV, SIGNAL] + _later(3))
        _write_results(self.base_dir, "STALECO", [PREV, SIGNAL, GREEN_TRIGGER])

    def tearDown(self):
        shutil.rmtree(self.base_dir)

    def test_scan_copies_only_current_qualifying_files(self):
        results = GapUpExpansionFilterService(self.base_dir).scan_and_classify()
        self.assertEqual({r.symbol: r.status for r in results}, {"WIDECO": "New", "OLDCO": "Tracking"})
        copied = sorted(p.name for p in (self.base_dir / const.GAP_UP_EXPANSION_FILTER_DIR_NAME).glob("*.xlsx"))
        self.assertEqual(copied, ["OLDCO_VSA.xlsx", "WIDECO_VSA.xlsx"])

    def test_publisher_emits_gap_up_expansion_filters(self):
        GapUpExpansionFilterService(self.base_dir).scan_and_classify()
        self.assertEqual(DataAggregator(self.base_dir).aggregate_pipeline_stats()["gap_up_expansion"], 2)
        output_file = self.base_dir / "dashboard" / "data.json"
        JSONPublisher(self.base_dir, output_file).publish()
        by_symbol = {d["symbol"]: d for d in json.loads(output_file.read_text())["gap_up_expansion_filters"]["daily"]}
        self.assertEqual(set(by_symbol), {"WIDECO", "OLDCO"})
        self.assertEqual(by_symbol["WIDECO"]["spread_ratio"], 1.25)
        self.assertEqual(by_symbol["OLDCO"]["trigger_candle"], "Red")
        self.assertEqual(by_symbol["OLDCO"]["track_sessions"], TRACK_SESSIONS)


class TestGapUpExpansionEmail(unittest.TestCase):
    ITEMS = {"daily": [
        {"symbol": "A&B", "status": "New", "sessions_since_trigger": 0, "track_sessions": 10,
         "signal_close": 106.0, "trigger_open": 107.0, "trigger_close": 113.0, "gap_pct": 0.94,
         "spread_ratio": 1.25, "trigger_candle": "Green"},
        {"symbol": "OLDCO", "status": "Tracking", "sessions_since_trigger": 3, "track_sessions": 10,
         "trigger_date": "2026-06-03", "trigger_candle": "Red", "trigger_close": 103.0, "last_close": 110.0,
         "return_pct": 6.8, "max_gain_pct": 7.1, "max_drawdown_pct": None, "above_signal_close": True},
    ]}

    def test_html_shows_both_tables_with_spread_and_candle(self):
        html = render_gap_up_expansion_html(self.ITEMS)
        self.assertIn("<strong>1</strong> new today, 1 still tracking", html)
        self.assertIn("A&amp;B", html)
        self.assertIn("1.25x", html)
        self.assertIn("3/10", html)
        self.assertIn("Red", html)

    def test_text_lists_rows_with_missing_values(self):
        text = render_gap_up_expansion_text(self.ITEMS)
        self.assertIn("New today (1)", text)
        self.assertIn("Still tracking (1)", text)
        self.assertIn("+6.80%", text)
        self.assertIn("--", text)

    def test_absent_and_empty_blocks(self):
        self.assertEqual(render_gap_up_expansion_html(None), "")
        self.assertEqual(count_new(None), 0)
        self.assertIn(EMPTY_MESSAGE, render_gap_up_expansion_text({"daily": []}))
        self.assertEqual(count_new(self.ITEMS), 1)


if __name__ == "__main__":
    unittest.main()
