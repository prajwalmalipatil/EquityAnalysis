from pathlib import Path
from src.services.macro_intelligence.config import load_config
from src.services.macro_intelligence.config import load_config
from src.services.macro_intelligence.event_repository import JSONEventReadRepository
from src.services.macro_intelligence.builders import ManifestBuilder, AnalyticsBuilder, RelationshipBuilder
from src.services.macro_intelligence.dashboard_mapper import DashboardMapper
from src.services.macro_intelligence.analytics_provider import AnalyticsProvider
from src.services.macro_intelligence.release_validator import ReleaseValidator
from src.services.macro_intelligence.publishers import JSONPublisher, ManifestPublisher, AnalyticsPublisher, SearchPublisher, RelationshipPublisher, GraphPublisher
from src.services.macro_intelligence.search_indexer import SearchIndexer, SearchDocumentBuilder
from src.services.macro_intelligence.relationship_engine import RelationshipCandidateGenerator, RelationshipResolver
from src.services.macro_intelligence.graph_builder import GraphBuilder, GraphViewModelBuilder
from src.services.macro_intelligence.read_models import DashboardBundle
from src.utils.observability import get_tenant_logger, get_metrics_tracker

logger = get_tenant_logger("publish-pipeline")
metrics = get_metrics_tracker()

def run_publish_pipeline(output_dir: Path) -> bool:
    """
    Executes Pipeline B (Publish Pipeline).
    Strict flow: Query -> Build -> Validate -> Publish
    """
    logger.info("PUBLISH_PIPELINE_STARTED", extra={"output_dir": str(output_dir)})
    
    try:
        with metrics.time_block("pipeline_total_duration_ms"):
            # 1. Setup
            config = load_config()
            read_repo = JSONEventReadRepository(config.storage)
            
            # 2. Query
            events = read_repo.get_active_events()
            events.sort(key=lambda x: x.official_data.publication_date, reverse=True)
            metrics.gauge("publish_events_count", len(events))
            
            # 3. Build Models
            dashboard_events = DashboardMapper.map_events(events)
            
            with metrics.time_block("analytics_generation_ms"):
                analytics_provider = AnalyticsProvider()
                analytics_read_model = analytics_provider.compute(events, run_stats={})
                analytics_view_model = AnalyticsBuilder.build(analytics_read_model)
            
            with metrics.time_block("search_index_generation_ms"):
                search_doc_builder = SearchDocumentBuilder()
                search_docs = [search_doc_builder.build(e) for e in events]
                search_indexer = SearchIndexer()
                search_index = search_indexer.build_index(search_docs)
            
            # 3.4 Relationship Intelligence
            with metrics.time_block("relationships_generation_ms"):
                candidate_generator = RelationshipCandidateGenerator()
                candidates = candidate_generator.generate(events)
                
                resolver = RelationshipResolver()
                relationships = resolver.resolve(candidates, events)
                
                relationships_view_model = RelationshipBuilder.build(relationships)
            metrics.gauge("relationships_count", len(relationships))
            
            # 3.5 Knowledge Graph
            with metrics.time_block("graph_generation_ms"):
                graph_builder = GraphBuilder()
                graph_read_model = graph_builder.build_read_model(events, relationships)
                graph_view_model = GraphViewModelBuilder.build(graph_read_model, "1")
            metrics.gauge("graph_nodes_count", len(graph_read_model.nodes))
            
            # 3.6 Manifest
            artifacts = {
                "dashboard": {"version": "2", "file": "data.json", "record_count": len(events)},
                "analytics": {"version": "1", "file": "analytics.json", "record_count": len(analytics_view_model.get("metrics", {}))},
                "analytics_history": {"version": "1", "file": "analytics-history.json"},
                "search": {"version": "1", "file": "search-index.json", "record_count": len(search_index)},
                "relationships": {"version": "1", "file": "relationships.json", "record_count": len(relationships)},
                "graph": {"version": "1", "file": "graph.json", "record_count": len(graph_read_model.nodes)}
            }
            manifest_view_model = ManifestBuilder.build(len(events), artifacts=artifacts)
            
            bundle = DashboardBundle(
                events=dashboard_events,
                analytics=analytics_view_model,
                manifest=manifest_view_model
            )
            
            # 4. Release Validation
            ReleaseValidator.validate(bundle)
            
            # 5. Publish
            with metrics.time_block("publish_io_ms"):
                JSONPublisher(output_dir).publish(dashboard_events)
                AnalyticsPublisher(output_dir).publish(analytics_view_model)
                SearchPublisher(output_dir).publish(search_index)
                
                metadata_dict = {
                    "generated_at": manifest_view_model.generated_at,
                    "pipeline_version": manifest_view_model.pipeline_version
                }
                RelationshipPublisher(output_dir).publish(relationships_view_model, metadata_dict)
                GraphPublisher(output_dir).publish(graph_view_model)
                
                ManifestPublisher(output_dir).publish(manifest_view_model)
        
        logger.info("PUBLISH_PIPELINE_SUCCESSFUL", extra={
            "event_count": bundle.manifest.event_count,
            "schema_version": bundle.manifest.schema_version
        })
        return True
        
    except Exception as e:
        logger.error("PUBLISH_PIPELINE_FAILED", extra={"error": str(e)})
        return False

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Example execution
    project_root = Path(__file__).parent.parent.parent.parent
    dashboard_dir = project_root / "dashboard"
    
    success = run_publish_pipeline(dashboard_dir)
    sys.exit(0 if success else 1)
