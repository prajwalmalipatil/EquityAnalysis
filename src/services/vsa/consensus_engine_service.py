"""
consensus_engine_service.py
Multi-Timeframe Consensus Engine.
Synthesizes Daily, Weekly, and Monthly EigenFilter signals into a single
institutional-grade consensus rating using a 40/35/25 weighting model.
"""

import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from src.constants import vsa_constants as const
from src.models.consensus_models import ConsensusRating
from src.models.vsa_models import EigenClassification
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("consensus-engine-service")

# Institutional Weightings
WEIGHT_MONTHLY = 40.0
WEIGHT_WEEKLY = 35.0
WEIGHT_DAILY = 25.0


class ConsensusEngineService:
    """
    Reads EigenFilter classifications across Daily, Weekly, and Monthly 
    timeframes and computes a weighted consensus score and rating.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.daily_dir = base_dir / const.EIGEN_FILTER_DIR_NAME
        self.weekly_dir = base_dir / const.WEEKLY_EIGEN_FILTER_DIR_NAME
        self.monthly_dir = base_dir / const.MONTHLY_EIGEN_FILTER_DIR_NAME
        
        self.consensus_dir = base_dir / const.CONSENSUS_RESULTS_DIR_NAME
        self.consensus_dir.mkdir(parents=True, exist_ok=True)

    def compute_consensus(self, daily_results: List[EigenClassification], weekly_results: List[EigenClassification], monthly_results: List[EigenClassification]) -> List[ConsensusRating]:
        """Calculates ratings based on provided classification results and exports them."""
        # 1. Gather all available signals
        symbols_data: Dict[str, Dict[str, str]] = {}
        
        self._populate_signals(daily_results, "daily", symbols_data)
        self._populate_signals(weekly_results, "weekly", symbols_data)
        self._populate_signals(monthly_results, "monthly", symbols_data)
        
        if not symbols_data:
            logger.info("No signals found across any timeframe to generate consensus.")
            return []

        # 2. Compute ratings
        ratings: List[ConsensusRating] = []
        for symbol, signals in symbols_data.items():
            rating = self._evaluate_symbol(symbol, signals)
            if rating:
                ratings.append(rating)

        # 3. Export to ConsensusResults directory
        self._export_results(ratings)
        
        self._log_summary(ratings)
        return ratings

    def _populate_signals(self, results: List[EigenClassification], timeframe: str, symbols_data: Dict[str, Dict[str, str]]):
        """Populates the symbols_data dictionary with sentiments from the provided results."""
        for r in results:
            if r.symbol not in symbols_data:
                symbols_data[r.symbol] = {
                    "daily": "None",
                    "weekly": "None",
                    "monthly": "None"
                }
            symbols_data[r.symbol][timeframe] = r.sentiment

    def _evaluate_symbol(self, symbol: str, signals: Dict[str, str]) -> Optional[ConsensusRating]:
        """Calculates the weighted score and assigns the star rating/label."""
        m_sent = signals["monthly"]
        w_sent = signals["weekly"]
        d_sent = signals["daily"]
        
        total_weight_available = 0.0
        score = 0.0
        
        m_score, w_score, d_score = 0.0, 0.0, 0.0
        
        if m_sent != "None":
            m_score = WEIGHT_MONTHLY if m_sent == "Bullish" else -WEIGHT_MONTHLY
            score += m_score
            
        if w_sent != "None":
            w_score = WEIGHT_WEEKLY if w_sent == "Bullish" else -WEIGHT_WEEKLY
            score += w_score
            
        if d_sent != "None":
            d_score = WEIGHT_DAILY if d_sent == "Bullish" else -WEIGHT_DAILY
            score += d_score
            
        if m_sent == "None" and w_sent == "None" and d_sent == "None":
            return None
            
        # Divide by the absolute total possible weight (100) to ensure only 
        # full alignment across all timeframes results in 100%.
        total_weight_absolute = WEIGHT_MONTHLY + WEIGHT_WEEKLY + WEIGHT_DAILY
        score_percentage = (score / total_weight_absolute) * 100
        
        # Classification Matrix
        if score_percentage == 100.0:
            star = 5
            label = "Strong Buy"
        elif 10.0 < score_percentage < 100.0:
            star = 4
            label = "Cautious Bullish"
        elif -10.0 <= score_percentage <= 10.0:
            star = 3
            label = "Mixed Trend"
        elif -100.0 < score_percentage < -10.0:
            star = 2
            label = "Cautious Bearish"
        elif score_percentage == -100.0:
            star = 1
            label = "Strong Sell"
        else:
            # Fallback
            star = 3
            label = "Mixed Trend"

        return ConsensusRating(
            symbol=symbol,
            daily_sentiment=d_sent,
            weekly_sentiment=w_sent,
            monthly_sentiment=m_sent,
            score_percentage=score_percentage,
            star_rating=star,
            consensus_label=label,
            monthly_score=m_score,
            weekly_score=w_score,
            daily_score=d_score
        )

    def _export_results(self, ratings: List[ConsensusRating]):
        """Exports the consensus ratings to an Excel file for the renderer."""
        if not ratings:
            return
            
        data = []
        for r in ratings:
            data.append({
                "Symbol": r.symbol,
                "Monthly_Sentiment": r.monthly_sentiment,
                "Weekly_Sentiment": r.weekly_sentiment,
                "Daily_Sentiment": r.daily_sentiment,
                "Score_Pct": r.score_percentage,
                "Stars": r.star_rating,
                "Label": r.consensus_label
            })
            
        df = pd.DataFrame(data)
        # Sort so Strong Buys are first, then down to Strong Sells
        df = df.sort_values(by="Score_Pct", ascending=False)
        
        out_path = self.consensus_dir / "consensus_ratings.xlsx"
        df.to_excel(out_path, index=False)

    @staticmethod
    def _log_summary(ratings: List[ConsensusRating]):
        """Logs the distribution of consensus labels."""
        strong_buys = len([r for r in ratings if r.star_rating == 5])
        caut_bulls = len([r for r in ratings if r.star_rating == 4])
        mixed = len([r for r in ratings if r.star_rating == 3])
        caut_bears = len([r for r in ratings if r.star_rating == 2])
        strong_sells = len([r for r in ratings if r.star_rating == 1])
        
        logger.info(
            f"CONSENSUS_COMPLETE: {len(ratings)} symbols rated "
            f"(5★: {strong_buys}, 4★: {caut_bulls}, 3★: {mixed}, "
            f"2★: {caut_bears}, 1★: {strong_sells})"
        )
