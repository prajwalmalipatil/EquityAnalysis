import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass

@dataclass(frozen=True)
class CollectionConfig:
    retry_limit: int
    base_backoff_seconds: int
    max_backoff_seconds: int
    timeout_seconds: int

@dataclass(frozen=True)
class DeduplicationConfig:
    similarity_threshold: float
    weights: Dict[str, float]

@dataclass(frozen=True)
class AIEnrichmentConfig:
    provider: str
    model_name: str
    confidence_threshold: int
    prompt_version: str

@dataclass(frozen=True)
class StorageConfig:
    base_path: Path
    attachments_dir: Path
    metadata_dir: Path
    logs_dir: Path
    cache_dir: Path
    history_file: Path
    state_file: Path

@dataclass(frozen=True)
class MacroConfig:
    version: str
    collection: CollectionConfig
    deduplication: DeduplicationConfig
    ai_enrichment: AIEnrichmentConfig
    storage: StorageConfig

def load_config(config_path: Path) -> MacroConfig:
    """Loads and validates the macro intelligence configuration from a YAML file."""
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse YAML configuration: {str(e)}")
        
    if not raw_config:
        raise ConfigurationError("Configuration file is empty")
        
    try:
        # Resolve paths relative to the project root (assuming script runs from root)
        project_root = Path.cwd()
        base_storage = Path(raw_config["storage"]["base_path"])
        if not base_storage.is_absolute():
            base_storage = project_root / base_storage
            
        storage_config = StorageConfig(
            base_path=base_storage,
            attachments_dir=base_storage / raw_config["storage"]["attachments_dir"],
            metadata_dir=base_storage / raw_config["storage"]["metadata_dir"],
            logs_dir=base_storage / raw_config["storage"]["logs_dir"],
            cache_dir=base_storage / raw_config["storage"]["cache_dir"],
            history_file=base_storage / raw_config["storage"]["history_file"],
            state_file=base_storage / raw_config["storage"]["state_file"]
        )
        
        # Validate weights sum to 1.0 (approximate for floating point)
        weights = raw_config["deduplication"]["weights"]
        if abs(sum(weights.values()) - 1.0) > 0.01:
            raise ConfigurationError("Deduplication weights must sum to 1.0")

        return MacroConfig(
            version=str(raw_config.get("version", "1.0")),
            collection=CollectionConfig(**raw_config["collection"]),
            deduplication=DeduplicationConfig(
                similarity_threshold=float(raw_config["deduplication"]["similarity_threshold"]),
                weights=weights
            ),
            ai_enrichment=AIEnrichmentConfig(**raw_config["ai_enrichment"]),
            storage=storage_config
        )
    except KeyError as e:
        raise ConfigurationError(f"Missing required configuration key: {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {str(e)}")
