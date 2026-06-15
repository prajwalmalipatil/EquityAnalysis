"""
pattern_router_service.py
Routes processed files to categorized pattern folders and runs downstream tasks.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from src.constants import vsa_constants as const
from src.services.orchestration.registry import ResearchModule, platform_registry
from src.services.reporting.view_builder_service import ViewBuilderService
from src.utils.observability import get_tenant_logger

from .age_again_filter_service import AgeAgainFilterService
from .consensus_engine_service import ConsensusEngineService
from .eigen_filter_service import EigenFilterService
from .eigen_transition_engine_service import EigenTransitionEngineService
from .monthly_eigen_filter_service import MonthlyEigenFilterService
from .weekly_eigen_filter_service import WeeklyEigenFilterService

logger = get_tenant_logger("vsa-pattern-router")


class VSAPatternRouter:
    """Routes/distributes processed files to categorized folders based on VSA/Anomaly patterns."""

    def __init__(self, output_base: Path):
        self.output_base = output_base

    def route_pattern_folders(self, processed_metadata: List[Dict[str, Any]]) -> None:
        """Copies files into Trending, Efforts, Ticker, Triggers, and Anomaly folders."""
        mapping = [
            ("is_trending", const.TRENDING_DIR_NAME, "Trending"),
            ("is_effort", const.EFFORTS_DIR_NAME, "Efforts"),
            ("is_ticker", const.TICKER_DIR_NAME, "Ticker"),
            ("is_trigger", const.TRIGGERS_DIR_NAME, "Triggers"),
            ("is_anomaly", const.ANOMALY_DIR_NAME, "Anomaly"),
        ]

        for key, dir_name, label in mapping:
            targets = [m for m in processed_metadata if m[key]]
            dest_dir = self.output_base / dir_name
            for m in targets:
                shutil.copy(m["path"], dest_dir)
            logger.info(f"POST_PROCESS: {label} Filtered {len(targets)} symbols")

    def register_module(self) -> None:
        """Registers the VSAProcessorService with the platform registry."""
        # Registered at module-level import to support early pipeline DAG validation.
        pass

    def run_filters(self) -> Dict[str, Any]:
        """Runs the four Eigen/Age filters and consensus engine."""
        eigen_results = EigenFilterService(self.output_base).scan_and_classify()
        logger.info(f"POST_PROCESS: EigenFilter Classified {len(eigen_results)} symbols")

        age_again_results = AgeAgainFilterService(self.output_base).scan_and_classify()
        logger.info(f"POST_PROCESS: AgeAgain Classified {len(age_again_results)} symbols")

        monthly_eigen_results = MonthlyEigenFilterService(
            self.output_base
        ).consolidate_and_classify()
        logger.info(
            f"POST_PROCESS: MonthlyEigenFilter Classified {len(monthly_eigen_results)} symbols"
        )

        weekly_eigen_results = WeeklyEigenFilterService(
            self.output_base
        ).consolidate_and_classify()
        logger.info(
            f"POST_PROCESS: WeeklyEigenFilter Classified {len(weekly_eigen_results)} symbols"
        )

        consensus_results = ConsensusEngineService(self.output_base).compute_consensus(
            eigen_results, weekly_eigen_results, monthly_eigen_results
        )
        logger.info(
            f"POST_PROCESS: ConsensusEngine Computed {len(consensus_results)} consensus ratings"
        )

        return {
            "eigen": eigen_results,
            "weekly": weekly_eigen_results,
            "monthly": monthly_eigen_results,
            "consensus": consensus_results,
        }

    def run_ete(
        self, processed_metadata: List[Dict[str, Any]], filter_results: Dict[str, Any]
    ) -> None:
        """Runs the Eigen Transition Engine (ETE) on all timeframes."""
        logger.info("POST_PROCESS: Running Eigen Transition Engine (ETE)")

        ete_daily = EigenTransitionEngineService(timeframe="daily")
        ete_weekly = EigenTransitionEngineService(timeframe="weekly")
        ete_monthly = EigenTransitionEngineService(timeframe="monthly")

        eigen_symbols_daily = {r.symbol: r.sentiment for r in filter_results["eigen"]}
        eigen_symbols_weekly = {r.symbol: r.sentiment for r in filter_results["weekly"]}
        eigen_symbols_monthly = {r.symbol: r.sentiment for r in filter_results["monthly"]}

        for m in processed_metadata:
            self._process_ete_symbol(
                m["symbol"],
                m["path"],
                ete_daily,
                ete_weekly,
                ete_monthly,
                eigen_symbols_daily,
                eigen_symbols_weekly,
                eigen_symbols_monthly,
            )

    def _process_ete_symbol(
        self,
        symbol: str,
        path: Path,
        ete_daily,
        ete_weekly,
        ete_monthly,
        eigen_symbols_daily,
        eigen_symbols_weekly,
        eigen_symbols_monthly,
    ) -> None:
        """Processes daily, weekly, and monthly ETE sequences for a single symbol."""
        try:
            df = pd.read_excel(path, sheet_name="VSA_Analysis")
            if df.empty:
                return

            # Daily ETE
            ete_daily.update_active_sequences(symbol, df)
            ete_daily.detect_triggers(
                symbol,
                df,
                symbol in eigen_symbols_daily,
                eigen_symbols_daily.get(symbol, "Neutral"),
            )

            # Weekly ETE
            weekly_df = WeeklyEigenFilterService._consolidate_to_weekly(df)
            if weekly_df is not None and not weekly_df.empty:
                ete_weekly.update_active_sequences(symbol, weekly_df)
                ete_weekly.detect_triggers(
                    symbol,
                    weekly_df,
                    symbol in eigen_symbols_weekly,
                    eigen_symbols_weekly.get(symbol, "Neutral"),
                )

            # Monthly ETE
            monthly_df = MonthlyEigenFilterService._consolidate_to_monthly(df)
            if monthly_df is not None and not monthly_df.empty:
                ete_monthly.update_active_sequences(symbol, monthly_df)
                ete_monthly.detect_triggers(
                    symbol,
                    monthly_df,
                    symbol in eigen_symbols_monthly,
                    eigen_symbols_monthly.get(symbol, "Neutral"),
                )

        except Exception as e:
            logger.error(f"ETE_PROCESS_FAILED for {symbol}: {e}")

    def publish_views(self) -> None:
        """Publishes the ETE UI views using the ViewBuilderService."""
        view_builder = ViewBuilderService(self.output_base.parent)
        view_builder.publish(pipeline_seconds=12.5)  # Example pipeline seconds
        logger.info("POST_PROCESS: View Builder Published ETE UI artifacts")

    def finalize(self, processed_metadata: List[Dict[str, Any]]) -> None:
        """Orchestrates all post-processing tasks for a completed run."""
        self.route_pattern_folders(processed_metadata)
        self.register_module()
        filter_results = self.run_filters()
        self.run_ete(processed_metadata, filter_results)
        self.publish_views()


# Module-level platform registration to ensure early DAG validation success
platform_registry.register(
    ResearchModule(
        name="VSAProcessorService",
        version="1.0.0",
        description="Core Volume Spread Analysis processing engine.",
        inputs=["CleanCSV"],
        outputs=["Signals"],
        dependencies=["DataQualityGate"],
    )
)
