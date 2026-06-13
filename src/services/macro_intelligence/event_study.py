import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import dateutil.parser
from typing import Optional, Dict, List

from src.services.macro_intelligence.models import (
    EventStudy, 
    PreEventRegime, 
    PostEventOutcome, 
    ReturnWindow, 
    MacroEvent
)
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("event-study-engine")

class EventStudyEngine:
    """
    Layer 3: Quantitative Correlation.
    Measures T-windows and regime shifts around macro events using local CSV data.
    """
    
    def __init__(self, equity_data_dir: Path):
        self.data_dir = Path(equity_data_dir)
        # In-memory cache for DataFrames to avoid repeated disk I/O
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _load_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
            
        # NIFTY 50 is saved as NIFTY50 in extract_symbol_data (spaces stripped)
        safe_symbol = symbol.replace(" ", "")
        files = list(self.data_dir.glob(f"{safe_symbol}_1Y_*.csv"))
        if not files:
            return None
            
        latest_file = sorted(files)[-1]
        try:
            df = pd.read_csv(latest_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Ensure Close is numeric
            df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce')
            
            self._price_cache[symbol] = df
            return df
        except Exception as e:
            logger.error("FAILED_TO_LOAD_PRICE_DATA", extra={"symbol": symbol, "error": str(e)})
            return None

    def _calculate_return_window(self, df: pd.DataFrame, t0_date: datetime) -> ReturnWindow:
        """Calculates returns for T-5 to T+20."""
        # Find exact or nearest next trading day for T0
        t0_date_ts = pd.Timestamp(t0_date.date())
        
        if df.empty:
            return ReturnWindow()
            
        # Get index location of T0 (or first date >= T0)
        future_dates = df.index[df.index >= t0_date_ts]
        if len(future_dates) == 0:
            return ReturnWindow() # Event is too recent or out of bounds
            
        t0_idx = df.index.get_loc(future_dates[0])
        p_t0 = df['Close'].iloc[t0_idx]
        
        def safe_return(offset: int) -> Optional[float]:
            target_idx = t0_idx + offset
            if 0 <= target_idx < len(df):
                p_target = df['Close'].iloc[target_idx]
                if pd.notna(p_t0) and pd.notna(p_target) and p_t0 > 0:
                    if offset < 0:
                        # T-minus return: (P_T0 - P_T-N) / P_T-N
                        p_t_minus = df['Close'].iloc[target_idx]
                        return round(((p_t0 / p_t_minus) - 1.0) * 100, 2) if p_t_minus > 0 else None
                    else:
                        # T-plus return: (P_T+N - P_T0) / P_T0
                        return round(((p_target / p_t0) - 1.0) * 100, 2)
            return None

        return ReturnWindow(
            t_minus_5=safe_return(-5),
            t_minus_3=safe_return(-3),
            t_minus_1=safe_return(-1),
            t_plus_1=safe_return(1),
            t_plus_3=safe_return(3),
            t_plus_5=safe_return(5),
            t_plus_10=safe_return(10),
            t_plus_20=safe_return(20)
        )

    def process(self, event: MacroEvent) -> EventStudy:
        """Generates an EventStudy for the given event."""
        pub_dt = dateutil.parser.parse(event.published_at)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            
        # 1. Determine index returns
        indices = ["NIFTY 50", "NIFTY BANK"]
        index_returns = {}
        for idx in indices:
            df = self._load_price_data(idx)
            if df is not None:
                index_returns[idx] = self._calculate_return_window(df, pub_dt)
                
        # 2. Determine sector returns (placeholder for now, mapping from indices)
        sector_returns = {}
        
        # 3. Determine specific stock returns if impacted
        stock_returns = {}
        if event.impact and event.impact.securities:
            for sec in event.impact.securities:
                df = self._load_price_data(sec)
                if df is not None:
                    stock_returns[sec] = self._calculate_return_window(df, pub_dt)

        # 4. Regime shifts (Stubbed until we have historical Eigen storage)
        # For now, we assume Neutral to Neutral unless we can compute it on the fly.
        pre = PreEventRegime(daily_eigen="Neutral", weekly_eigen="Neutral", monthly_eigen="Neutral")
        post = PostEventOutcome(daily_eigen="Neutral", weekly_eigen="Neutral", monthly_eigen="Neutral")

        return EventStudy(
            pre_event_regime=pre,
            post_event_outcome=post,
            index_returns=index_returns,
            sector_returns=sector_returns,
            stock_returns=stock_returns
        )
