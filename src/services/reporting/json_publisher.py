import json
from pathlib import Path
from datetime import datetime, timezone
from src.services.reporting.data_aggregator import DataAggregator
from src.utils.observability import get_tenant_logger
from src.services.macro_intelligence.config import load_config
from src.services.macro_intelligence.event_repository import JSONEventReadRepository
from src.services.macro_intelligence.query_service import MacroQueryService
from src.services.macro_intelligence.dashboard_mapper import DashboardMapper

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

        # Gather macro intelligence
        macro_intelligence = {
            "last_event_at": None,
            "total_events": 0,
            "recent_events": []
        }
        
        history_dir = self.output_file.parent / "history"
        index_file = history_dir / "index.json"
        last_session_date_str = None
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                try:
                    history_index = json.load(f)
                    if history_index:
                        last_session_date_str = history_index[-1]
                except json.JSONDecodeError:
                    pass

        # Load Macro events via Query Service
        config_path = Path("src/services/macro_intelligence/config.yaml")
        macro_config = load_config(config_path)
        
        import dataclasses
        new_storage = dataclasses.replace(
            macro_config.storage, 
            base_path=history_dir, 
            history_file=history_dir / "rbi_events.jsonl"
        )
        
        repo = JSONEventReadRepository(new_storage)
        query_service = MacroQueryService(repo)
        
        all_macro_events = query_service.get_all_events()
        
        # --- Release Blocker Validations ---
        event_ids = set()
        title_dates = set()
        urls = set()
        
        for e in all_macro_events:
            # 1. No mock IDs
            if "mock-" in e.event_id.lower():
                raise ValueError(f"Release Blocker: Mock ID found: {e.event_id}")
                
            # 2. Unique event_ids
            if e.event_id in event_ids:
                raise ValueError(f"Release Blocker: Duplicate event_id found: {e.event_id}")
            event_ids.add(e.event_id)
            
            # 3. No duplicate titles with identical dates
            title_date_key = f"{e.official_data.title}_{e.official_data.publication_date[:10]}"
            if title_date_key in title_dates:
                raise ValueError(f"Release Blocker: Duplicate title+date found: {title_date_key}")
            title_dates.add(title_date_key)
            
            # 4. No duplicate URLs (if valid URL provided)
            url = e.official_data.official_url
            if url and url != "Unknown" and not url.startswith("http://mock"):
                if url in urls:
                    raise ValueError(f"Release Blocker: Duplicate URL found: {url}")
                urls.add(url)
                
            # 5. Required Fields
            if not e.official_data.title:
                raise ValueError(f"Release Blocker: Missing title for event: {e.event_id}")
                
            # 6. AI Validations
            if e.derived_data and e.derived_data.impact:
                conf = e.derived_data.impact.confidence
                if not (0 <= conf <= 100):
                    raise ValueError(f"Release Blocker: Confidence out of bounds (0-100) for {e.event_id}: {conf}")
                    
        # --- Mapping via DashboardMapper ---
        mapped_events = DashboardMapper.map_events(all_macro_events)
        
        import dataclasses
        mapped_events_dicts = [dataclasses.asdict(e) for e in mapped_events]
        
        macro_intelligence["total_events"] = len(mapped_events_dicts)
        if mapped_events_dicts:
            macro_intelligence["last_event_at"] = max(e["published"] for e in mapped_events_dicts)
            
            # Assign Trading Day Queue status
            for e in mapped_events_dicts:
                is_new = False
                if last_session_date_str:
                    if e["published"][:10] > last_session_date_str:
                        is_new = True
                else:
                    is_new = True
                e["is_new_since_last_session"] = is_new
                
            macro_intelligence["recent_events"] = mapped_events_dicts[:20]

        # Build schema payload
        payload = {
            "schema_version": "2",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "extraction_list": symbol_data.get("extraction", []),
            "vsa_list": symbol_data.get("vsa", []),
            "trending_list": symbol_data.get("trending", []),
            "anomaly_list": symbol_data.get("anomaly", []),
            "triggers_list": symbol_data.get("triggers", []),
            "stage_statistics": stats,
            "consensus": consensus,
            "eigen_filters": eigen_filters,
            "ticker_alerts": ticker_alerts,
            "macro_intelligence": macro_intelligence
        }

        # Write to JSON
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        # Historical Archiving
        history_dir = self.output_file.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        history_file = history_dir / f"data_{date_str}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
            
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                try:
                    history_index = json.load(f)
                except json.JSONDecodeError:
                    history_index = []
        else:
            history_index = []
            
        if date_str not in history_index:
            history_index.append(date_str)
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(history_index, f, indent=2)

        logger.info("JSON_PUBLISHED_SUCCESSFULLY", extra={"output_file": str(self.output_file)})
