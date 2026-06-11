"""
consensus_engine_service.py
Multi-Timeframe Consensus Engine.
Synthesizes Daily, Weekly, and Monthly EigenFilter signals into a single
institutional-grade consensus rating using a 40/35/25 weighting model.
"""

import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.constants import vsa_constants as const
from src.models.consensus_models import ConsensusRating
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

    def compute_consensus(self) -> List[ConsensusRating]:
        """Scans all three directories, calculates ratings, and exports results."""
        # 1. Gather all available signals
        symbols_data: Dict[str, Dict[str, str]] = {}
        
        self._load_signals(self.daily_dir, "daily", symbols_data)
        self._load_signals(self.weekly_dir, "weekly", symbols_data)
        self._load_signals(self.monthly_dir, "monthly", symbols_data)
        
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

    def _load_signals(self, dir_path: Path, timeframe: str, symbols_data: Dict[str, Dict[str, str]]):
        """Reads the Excel files in the given directory to determine sentiment."""
        if not dir_path.exists():
            return
            
        for xlsx_path in dir_path.glob("*.xlsx"):
            symbol = xlsx_path.stem.replace("_VSA", "")
            if symbol not in symbols_data:
                symbols_data[symbol] = {
                    "daily": "None",
                    "weekly": "None",
                    "monthly": "None"
                }
                
            try:
                # We can deduce sentiment by looking at the specific label/column, 
                # but since EigenFilter services only copy files into these dirs if they qualify,
                # we must read the sentiment from the dataframe or infer it from the close position.
                # All these dataframes have a 'Close_Position' and 'Gap_Direction' somewhere, but
                # we didn't inject 'Sentiment' directly into the Excel. 
                # Wait, the simplest way to know sentiment is to check if T_CP > T1_CP and Gap-Up vs Gap-Down
                # Actually, the quickest way is checking gap direction since Bullish = Gap-Up, Bearish = Gap-Down in EigenFilter.
                df = pd.read_excel(xlsx_path, sheet_name="VSA_Analysis")
                if len(df) < 2:
                    continue
                
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                t_open = float(latest.get("Open", 0))
                t1_close = float(prev.get("Close", 0))
                t_cp = float(latest.get("Close_Position", 0.5))
                t1_cp = float(prev.get("Close_Position", 0.5))
                
                sentiment = "None"
                if t_open > t1_close and t_cp >= t1_cp:
                    sentiment = "Bullish"
                elif t_open < t1_close and t_cp <= t1_cp:
                    sentiment = "Bearish"
                
                if sentiment != "None":
                    symbols_data[symbol][timeframe] = sentiment
                
            except Exception:
                continue

    def _evaluate_symbol(self, symbol: str, signals: Dict[str, str]) -> Optional[ConsensusRating]:
        """Calculates the weighted score and assigns the star rating/label."""
        m_sent = signals["monthly"]
        w_sent = signals["weekly"]
        d_sent = signals["daily"]
        
        total_weight_available = 0.0
        score = 0.0
        
        m_score, w_score, d_score = 0.0, 0.0, 0.0
        
        if m_sent != "None":
            total_weight_available += WEIGHT_MONTHLY
            m_score = WEIGHT_MONTHLY if m_sent == "Bullish" else -WEIGHT_MONTHLY
            score += m_score
            
        if w_sent != "None":
            total_weight_available += WEIGHT_WEEKLY
            w_score = WEIGHT_WEEKLY if w_sent == "Bullish" else -WEIGHT_WEEKLY
            score += w_score
            
        if d_sent != "None":
            total_weight_available += WEIGHT_DAILY
            d_score = WEIGHT_DAILY if d_sent == "Bullish" else -WEIGHT_DAILY
            score += d_score
            
        if total_weight_available == 0:
            return None
            
        score_percentage = (score / total_weight_available) * 100
        
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
