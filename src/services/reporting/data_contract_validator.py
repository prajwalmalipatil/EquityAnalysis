import json
from pathlib import Path
from pydantic import ValidationError, BaseModel
from typing import List, Dict, Any
from src.utils.observability import get_tenant_logger
from src.models.ete_models import DashboardManifest, SystemHealth, SequenceSummary

logger = get_tenant_logger("data-contract")

class SummarySchema(BaseModel):
    active: List[SequenceSummary]
    completed: List[SequenceSummary]
    failed: List[SequenceSummary]

class DataContractValidator:
    """
    Validates JSON artifacts against Pydantic schemas before they are published to the UI.
    """
    def __init__(self, publish_dir: Path):
        self.publish_dir = Path(publish_dir)

    def validate_all(self) -> bool:
        """
        Validates manifest.json, system_health.json, and the files referenced in manifest.
        Returns True if all contracts are fulfilled.
        """
        logger.info(f"Validating JSON contracts in {self.publish_dir}...")
        
        # 1. Manifest
        manifest_path = self.publish_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("CONTRACT_FAILED: manifest.json is missing.")
            return False
            
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            manifest = DashboardManifest(**manifest_data)
        except json.JSONDecodeError as e:
            logger.error(f"CONTRACT_FAILED: manifest.json is invalid JSON: {e}")
            return False
        except ValidationError as e:
            logger.error(f"CONTRACT_FAILED: manifest.json violates schema: {e}")
            return False

        # 2. System Health
        health_path = self.publish_dir / "system_health.json"
        if not health_path.exists():
            logger.error("CONTRACT_FAILED: system_health.json is missing.")
            return False
            
        try:
            with open(health_path, 'r') as f:
                health_data = json.load(f)
            SystemHealth(**health_data)
        except Exception as e:
            logger.error(f"CONTRACT_FAILED: system_health.json violates schema: {e}")
            return False

        # 3. Summary (Referenced in Manifest)
        summary_filename = manifest.files.get("summary")
        if not summary_filename:
            logger.error("CONTRACT_FAILED: Manifest does not contain a 'summary' pointer.")
            return False
            
        summary_path = self.publish_dir / summary_filename
        if not summary_path.exists():
            logger.error(f"CONTRACT_FAILED: Summary file {summary_filename} is missing.")
            return False
            
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            SummarySchema(**summary_data)
        except Exception as e:
            logger.error(f"CONTRACT_FAILED: {summary_filename} violates schema: {e}")
            return False

        logger.info("ALL_CONTRACTS_PASSED")
        return True
