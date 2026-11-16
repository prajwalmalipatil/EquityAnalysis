"""
test_gap_momentum_filter.py
Unit tests for the Gap Momentum indicator: classification, publishing and email rendering.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.constants import vsa_constants as const
from src.services.reporting.data_aggregator import DataAggregator
from src.services.reporting.gap_momentum_email_renderer import (
    EMPTY_MESSAGE,
    render_gap_momentum_html,
    render_gap_momentum_text,
)
from src.services.reporting.json_publisher import JSONPublisher
from src.services.vsa.gap_momentum_filter_service import GapMomentumFilterService


def _bar(open_, close, volume, spread, date="2026-06-02"):
    return pd.Series(
        {"Date": date, "Open": open_, "Close": close, "Volume": volume, "Spread": spread}
    )


PREV = _bar(open_=98.0, close=100.0, volume=1000, spread=5.0, date="2026-06-01")


def _evaluate(latest, prev=PREV):
    return GapMomentumFilterService.evaluate_stock("TEST", latest, prev)


def _write_results(base_dir: Path, folder: str, symbol: str, rows: list) -> None:
    target = base_dir / folder
    target.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target / f"{symbol}_VSA.xlsx") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="VSA_Analysis", index=False)


class TestGapMomentumEvaluation(unittest.TestCase):
    def test_gap_up_green_close_is_bullish(self):
        result = _evaluate(_bar(open_=102.0, close=106.0, volume=2000, spread=8.0))
        self.assertEqual(result.scenario, "Gap_Up_Green_Close")
        self.assertEqual(result.sentiment, "Bullish")
        self.assertEqual(result.signal_date, "2026-06-02")
        self.assertAlmostEqual(result.gap_pct, 2.0)
        self.assertAlmostEqual(result.change_pct, 6.0)
        self.assertAlmostEqual(result.volume_pct, 100.0)
        self.assertAlmostEqual(result.spread_pct, 60.0)

    def test_gap_down_red_close_is_bearish(self):
        result = _evaluate(_bar(open_=98.0, close=94.0, volume=1500, spread=6.0))
        self.assertEqual(result.scenario, "Gap_Down_Red_Close")
        self.assertEqual(result.sentiment, "Bearish")
        self.assertAlmostEqual(result.gap_pct, -2.0)

    def test_gap_not_confirmed_by_candle_is_rejected(self):
        self.assertIsNone(_evaluate(_bar(open_=102.0, close=101.0, volume=2000, spread=8.0)))
        self.assertIsNone(_evaluate(_bar(open_=98.0, close=99.0, volume=2000, spread=8.0)))

    def test_requires_both_volume_and_spread_expansion(self):
        self.assertIsNone(_evaluate(_bar(open_=102.0, close=106.0, volume=1000, spread=8.0)))
        self.assertIsNone(_evaluate(_bar(open_=102.0, close=106.0, volume=2000, spread=5.0)))
        self.assertIsNone(_evaluate(_bar(open_=102.0, close=106.0, volume=900, spread=4.0)))

    def test_flat_open_or_doji_is_rejected(self):
        self.assertIsNone(_evaluate(_bar(open_=100.0, close=106.0, volume=2000, spread=8.0)))
        self.assertIsNone(_evaluate(_bar(open_=102.0, close=102.0, volume=2000, spread=8.0)))

    def test_invalid_previous_bar_is_rejected(self):
        latest = _bar(open_=102.0, close=106.0, volume=2000, spread=8.0)
        for field in ("Volume", "Spread", "Close"):
            for bad in (0, float("nan")):
                prev = PREV.copy()
                prev[field] = bad
                self.assertIsNone(_evaluate(latest, prev), f"{field}={bad}")

    def test_single_row_frame_is_rejected(self):
        frame = pd.DataFrame([PREV])
        self.assertIsNone(GapMomentumFilterService.classify_frame("TEST", frame))


class TestGapMomentumPipeline(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(tempfile.mkdtemp())
        prev = PREV.to_dict()
        self.bars = {
            "UPCO": [prev, _bar(102.0, 106.0, 2000, 8.0).to_dict()],
            "DOWNCO": [prev, _bar(98.0, 94.0, 1500, 6.0).to_dict()],
            "FLATCO": [prev, _bar(102.0, 106.0, 500, 8.0).to_dict()],
        }
        for symbol, rows in self.bars.items():
            _write_results(self.base_dir, const.RESULTS_DIR_NAME, symbol, rows)

    def tearDown(self):
        shutil.rmtree(self.base_dir)

    def test_scan_copies_only_qualifying_files(self):
        results = GapMomentumFilterService(self.base_dir).scan_and_classify()
        self.assertEqual({r.symbol: r.sentiment for r in results},
                         {"UPCO": "Bullish", "DOWNCO": "Bearish"})
        copied = sorted(p.name for p in (self.base_dir / const.GAP_MOMENTUM_FILTER_DIR_NAME).glob("*.xlsx"))
        self.assertEqual(copied, ["DOWNCO_VSA.xlsx", "UPCO_VSA.xlsx"])

    def test_scan_without_results_dir_returns_empty(self):
        empty_base = self.base_dir / "empty"
        empty_base.mkdir()
        self.assertEqual(GapMomentumFilterService(empty_base).scan_and_classify(), [])

    def test_publisher_emits_gap_momentum_filters(self):
        GapMomentumFilterService(self.base_dir).scan_and_classify()
        aggregator = DataAggregator(self.base_dir)
        self.assertEqual(aggregator.aggregate_pipeline_stats()["gap_momentum"], 2)
        self.assertIsNone(aggregator.get_gap_momentum_details("FLATCO"))

        output_file = self.base_dir / "dashboard" / "data.json"
        JSONPublisher(self.base_dir, output_file).publish()
        daily = json.loads(output_file.read_text())["gap_momentum_filters"]["daily"]
        by_symbol = {d["symbol"]: d for d in daily}
        self.assertEqual(set(by_symbol), {"UPCO", "DOWNCO"})
        self.assertEqual(by_symbol["UPCO"]["label"], "Gap-Up Continuation")
        self.assertEqual(by_symbol["DOWNCO"]["gap_pct"], -2.0)


class TestGapMomentumEmail(unittest.TestCase):
    ITEMS = {"daily": [
        {"symbol": "UPCO", "sentiment": "Bullish", "signal_date": "2026-06-02", "t1_close": 100.0,
         "t_open": 102.0, "t_close": 106.0, "gap_pct": 2.0, "change_pct": 6.0,
         "volume_pct": 100.0, "spread_pct": 60.0},
        {"symbol": "M&M<b>", "sentiment": "Bearish", "signal_date": "2026-06-02", "t1_close": 100.0,
         "t_open": 98.0, "t_close": 94.0, "gap_pct": -2.0, "change_pct": -6.0,
         "volume_pct": 250.0, "spread_pct": None},
    ]}

    def test_html_lists_both_scenarios_and_escapes_symbols(self):
        html = render_gap_momentum_html(self.ITEMS)
        self.assertIn("2</strong> stocks detected for session 2026-06-02 (Gap-Up: 1, Gap-Down: 1)", html)
        self.assertIn("UPCO", html)
        self.assertIn("+2.00%", html)
        self.assertIn("M&amp;M&lt;b&gt;", html)
        self.assertNotIn("M&M<b>", html)

    def test_text_lists_rows_with_missing_values(self):
        text = render_gap_momentum_text(self.ITEMS)
        self.assertIn("Gap-Up Continuation (1)", text)
        self.assertIn("Gap-Down Continuation (1)", text)
        self.assertIn("-6.00%", text)
        self.assertIn("--", text)

    def test_absent_block_renders_nothing(self):
        self.assertEqual(render_gap_momentum_html(None), "")
        self.assertEqual(render_gap_momentum_text(None), "")

    def test_empty_block_states_no_matches(self):
        self.assertIn(EMPTY_MESSAGE, render_gap_momentum_html({"daily": []}))
        self.assertIn(EMPTY_MESSAGE, render_gap_momentum_text({"daily": []}))


if __name__ == "__main__":
    unittest.main()
