import sys
import time
from pathlib import Path
from typing import Dict, Any

from src.utils.observability import get_tenant_logger
from src.services.vsa.data_quality_service import DataQualityService
from src.services.vsa.processor_service import VSAProcessorService
from src.services.reporting.json_publisher import JSONPublisher

logger = get_tenant_logger("orchestrator")

class PipelineOrchestrator:
    """
    Explicit DAG Orchestrator for the Quantitative Research Platform.
    Ensures sequential reliability and fail-fast dependencies.
    """
    def __init__(self, data_dir: Path, workers: int = 4):
        self.data_dir = data_dir
        self.workers = workers
        self.stats = {}

    def run_stage_data_quality(self) -> bool:
        logger.info("DAG_STAGE: Data Quality Gate")
        dq_service = DataQualityService(self.data_dir)
        stats = dq_service.run_gate()
        self.stats["data_quality"] = stats
        
        if stats["passed"] == 0 and stats["total_files"] > 0:
            logger.error("All files failed Data Quality Gate. Halting pipeline.")
            return False
        return True

    def run_stage_vsa_and_ete(self) -> bool:
        logger.info("DAG_STAGE: VSA -> Eigen -> ETE -> ViewBuilder -> DataContract")
        try:
            import concurrent.futures
            
            service = VSAProcessorService(output_base=self.data_dir)
            csv_files = list(self.data_dir.glob("*.csv"))
            
            # Filter out quarantine
            csv_files = [f for f in csv_files if f.parent.name != "quarantine"]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(service.process_file, f): f for f in csv_files}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res.get("success"):
                        service.stats["success_files"] += 1
                        service._processed_metadata.append(res["metadata"])
            
            service.finalize_run()
            self.stats["vsa_processor"] = service.stats
            return True
        except Exception as e:
            logger.error(f"DAG_STAGE_FAILED: VSA/ETE Pipeline: {e}")
            return False

    def run_stage_macro(self) -> bool:
        logger.info("DAG_STAGE: Macro Intelligence")
        try:
            import main_macro
            # We call the main method directly. It uses Path(__file__).parent.
            main_macro.main()
            return True
        except Exception as e:
            logger.error(f"DAG_STAGE_FAILED: Macro Intelligence: {e}")
            return False

    def run_stage_publisher(self) -> bool:
        logger.info("DAG_STAGE: Legacy JSON Publisher")
        try:
            # Using dashboard/data.json as legacy path
            output_file = self.data_dir.parent / "dashboard" / "data.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            publisher = JSONPublisher(self.data_dir, output_file)
            publisher.publish()
            return True
        except Exception as e:
            logger.error(f"DAG_STAGE_FAILED: Legacy Publisher: {e}")
            return False

    def get_dir_size_mb(self, directory: Path) -> float:
        if not directory.exists(): return 0.0
        total_size = sum(f.stat().st_size for f in directory.glob('**/*') if f.is_file())
        return round(total_size / (1024 * 1024), 2)

    def execute_dag(self):
        t0 = time.time()
        logger.info("STARTING_PIPELINE_ORCHESTRATOR")
        
        # 1. Platform Discovery & Validation
        try:
            import main_macro # Import to trigger module registration
        except ImportError:
            pass # Gracefully skip if dependencies (e.g. google.generativeai) are missing in tests
            
        from src.services.orchestration.registry import platform_registry
        if not platform_registry.validate_dependencies():
            logger.error("DAG_VALIDATION_FAILED: Unmet dependencies. Halting Pipeline.")
            sys.exit(1)
            
        registry_path = self.data_dir.parent / "dashboard_next" / "research_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        platform_registry.export_manifest(registry_path)
        
        self.stats["stages"] = {}

        t_start = time.time()
        if not self.run_stage_data_quality():
            sys.exit(1)
        self.stats["stages"]["data_quality_sec"] = round(time.time() - t_start, 2)

        t_start = time.time()
        if not self.run_stage_vsa_and_ete():
            sys.exit(1)
        self.stats["stages"]["vsa_ete_sec"] = round(time.time() - t_start, 2)

        # Macro Intelligence operates independently of VSA so it can fail without halting dashboard
        t_start = time.time()
        self.run_stage_macro()
        self.stats["stages"]["macro_sec"] = round(time.time() - t_start, 2)
        
        # Calculate Event Repo Size
        from src.constants.vsa_constants import ETE_EVENTS_DIR
        self.stats["repository_size_mb"] = self.get_dir_size_mb(Path(ETE_EVENTS_DIR))

        t_start = time.time()
        if not self.run_stage_publisher():
            sys.exit(1)
        self.stats["stages"]["publisher_sec"] = round(time.time() - t_start, 2)

        duration = time.time() - t0
        self.stats["total_duration_sec"] = round(duration, 2)
        
        metrics_file = self.data_dir.parent / "orchestrator_metrics.json"
        import json
        with open(metrics_file, "w") as f:
            json.dump(self.stats, f)
            
        logger.info("PIPELINE_ORCHESTRATOR_COMPLETE", extra={"duration_seconds": duration, "stats": self.stats})
        return True
