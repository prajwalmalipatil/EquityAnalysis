import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("research-registry")

class ResearchModule(BaseModel):
    name: str = Field(..., description="Unique name of the analytical module")
    version: str = Field(..., description="Semantic version of the module")
    owner: str = Field(default="CoreQuant", description="Team or system that owns the module")
    maturity: str = Field(default="Production", description="Lifecycle stage (e.g., Alpha, Beta, Production)")
    description: str = Field(..., description="Human-readable description of capabilities")
    capabilities: List[str] = Field(default_factory=list, description="List of platform capabilities advertised by this module")
    inputs: List[str] = Field(default_factory=list, description="Data inputs required (e.g. OHLCV)")
    outputs: List[str] = Field(default_factory=list, description="Data outputs produced (e.g. ResearchEvents)")
    dependencies: List[str] = Field(default_factory=list, description="Module names this module strictly depends on")
    health_checks: List[str] = Field(default_factory=list, description="Identifiers of health checks to execute")
    contracts: List[str] = Field(default_factory=list, description="Identifiers for data contracts enforced by this module")

class PlatformRegistry:
    """
    Central discovery registry for all analytical modules in the Quantitative Research Platform.
    Enables true inversion of control and dynamic DAG orchestration.
    """
    def __init__(self):
        self._modules: Dict[str, ResearchModule] = {}

    def register(self, module: ResearchModule):
        if module.name in self._modules:
            logger.warning(f"MODULE_OVERWRITE: {module.name} is already registered. Overwriting.")
        self._modules[module.name] = module
        logger.info(f"MODULE_REGISTERED: {module.name} v{module.version}")

    def get_module(self, name: str) -> ResearchModule:
        return self._modules.get(name)

    def get_all_modules(self) -> List[ResearchModule]:
        return list(self._modules.values())

    def validate_dependencies(self) -> bool:
        """
        Ensures all declared dependencies are actually registered.
        Returns True if the DAG is structurally sound.
        """
        valid = True
        for name, mod in self._modules.items():
            for dep in mod.dependencies:
                if dep not in self._modules:
                    logger.error(f"MISSING_DEPENDENCY: '{name}' requires '{dep}' which is not registered.")
                    valid = False
        return valid

    def export_manifest(self, output_path: Path):
        """Exports the registry to a JSON payload for the ViewBuilder."""
        payload = {
            "registry_version": "1.0",
            "modules": [m.model_dump() for m in self.get_all_modules()]
        }
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"REGISTRY_MANIFEST_EXPORTED to {output_path}")

# Global singleton for easy injection
platform_registry = PlatformRegistry()
