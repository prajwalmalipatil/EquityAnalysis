import json
from pathlib import Path
from datetime import datetime, timezone
from src.services.reporting.data_aggregator import DataAggregator
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("json-publisher")

class JSONPublisher:
    """
    Publishes the pipeline results to a static JSON file for the Web UI Dashboard.
    Implements the schema contract outlined in Phase 2.
    """
    def __init__(self, base_dir: Path, output_file: Path):
        self.base_dir = base_dir
        self.output_file = output_file
        self.aggregator = DataAggregator(base_dir)

    def publish(self):
        logger.info("PUBLISHING_JSON_DATA", extra={"output_file": str(self.output_file)})
        stats = self.aggregator.aggregate_pipeline_stats()
        symbol_data = self.aggregator.get_symbol_lists()

        # Gather consensus
        consensus = self.aggregator.get_consensus_details()

        # Gather eigen details
        daily_eigen = []
        for sym in symbol_data.get("eigen_filter", []):
            details = self.aggregator.get_eigen_details(sym)
            if details: daily_eigen.append(details)

        weekly_eigen = []
        for sym in symbol_data.get("weekly_eigen", []):
            details = self.aggregator.get_weekly_eigen_details(sym)
            if details: weekly_eigen.append(details)

        monthly_eigen = []
        for sym in symbol_data.get("monthly_eigen", []):
            details = self.aggregator.get_monthly_eigen_details(sym)
            if details: monthly_eigen.append(details)

        eigen_filters = {
            "daily": daily_eigen,
            "weekly": weekly_eigen,
            "monthly": monthly_eigen
        }

        # Gather ticker alerts
        ticker_alerts = []
        for sym in symbol_data.get("ticker", []):
            details = self.aggregator.get_ticker_details(sym)
            if details: ticker_alerts.append(details)

        # Build schema payload
        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "stage_statistics": stats,
            "consensus": consensus,
            "eigen_filters": eigen_filters,
            "ticker_alerts": ticker_alerts
        }

        # Write to JSON
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        logger.info("JSON_PUBLISHED_SUCCESSFULLY", extra={"output_file": str(self.output_file)})
