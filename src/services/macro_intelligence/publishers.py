import json
from pathlib import Path
from dataclasses import asdict
from src.services.macro_intelligence.read_models import DashboardBundle
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("publishers")


class AnalyticsPublisher:
    """Pure serializer that writes the AnalyticsViewModel to analytics.json."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def publish(self, analytics_view_model: 'AnalyticsViewModel') -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        analytics_dict = asdict(analytics_view_model)
        
        analytics_file = self.output_dir / "analytics.json"
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics_dict, f, indent=2)
            
        # Update analytics-history.json
        history_file = self.output_dir / "analytics-history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                pass
                
        # Use YYYY-MM-DD from generated_at
        current_date = analytics_view_model.generated_at[:10]
        
        # Archive analytics json historically
        history_dir = self.output_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_analytics_file = history_dir / f"analytics_{current_date}.json"
        with open(history_analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics_dict, f, indent=2)
        
        # Check if today's entry exists, if so update it, else append
        existing_entry = next((entry for entry in history if entry.get('date') == current_date), None)
        
        new_entry = {
            "date": current_date,
            "events": analytics_view_model.event_count,
            "business_score": analytics_dict['analytics']['quality'].get('avg_quality_score', 0)
        }
        
        if existing_entry:
            existing_entry.update(new_entry)
        else:
            history.append(new_entry)
            
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)


class ManifestPublisher:
    """Pure serializer that writes the manifest.json."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def publish(self, manifest_view_model: 'ManifestViewModel') -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = self.output_dir / "manifest.json"
        
        existing_data = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing manifest for merge: {e}")
                
        manifest_dict = asdict(manifest_view_model)
        
        # Merge ETE keys if present in existing manifest to prevent clobbering ETE UI tab
        ete_keys = ["engine_version", "research_events", "active_sequences", "completed_sequences", "failed_sequences", "last_market_date", "files"]
        for key in ete_keys:
            if key in existing_data:
                if key == "files" and isinstance(existing_data["files"], dict):
                    if "files" not in manifest_dict or not isinstance(manifest_dict["files"], dict):
                        manifest_dict["files"] = {}
                    manifest_dict["files"].update(existing_data["files"])
                else:
                    manifest_dict[key] = existing_data[key]
                    
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_dict, f, indent=2)
            
        # Archive manifest json historically
        current_date = manifest_view_model.generated_at[:10]
        history_dir = self.output_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_manifest_file = history_dir / f"manifest_{current_date}.json"
        with open(history_manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_dict, f, indent=2)

class SearchPublisher:
    """Pure serializer that writes the inverted search index to search-index.json."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def publish(self, search_index: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        search_file = self.output_dir / "search-index.json"
        
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(search_index, f, indent=0, sort_keys=True)

class RelationshipPublisher:
    """Serializes the explicitly generated relationships to relationships.json."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def publish(self, relationships: list, metadata: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rel_file = self.output_dir / "relationships.json"
        
        payload = {
            "schema_version": 1,
            "generated_at": metadata.get("generated_at"),
            "pipeline_version": metadata.get("pipeline_version"),
            "relationship_count": len(relationships),
            "relationships": relationships
        }
        
        with open(rel_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

class GraphPublisher:
    """Serializes the GraphViewModel into graph.json."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def publish(self, graph_view_model) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        graph_file = self.output_dir / "graph.json"
        
        # graph_view_model is a dataclass, but we need to dictify it 
        # Actually since we mapped it in GraphViewModelBuilder, we can just use dataclasses.asdict
        payload = asdict(graph_view_model)
        
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
