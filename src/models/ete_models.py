from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class ETEState(str, Enum):
    TRIGGERED = "Triggered"
    WAITING = "Waiting"
    TRACKING = "Tracking"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    FAILED = "Failed"
    EXPIRED = "Expired"
    ARCHIVED = "Archived"
    INTEGRITY_FAILED = "Integrity_Failed"

class FailureReason(str, Enum):
    VOLUME_MISMATCH = "Volume Mismatch"
    SPREAD_MISMATCH = "Spread Mismatch"
    MISSING_CANDLE = "Missing Candle"
    DATA_UNAVAILABLE = "Data Unavailable"
    EXPIRED = "Expired"
    CONCURRENT_TRIGGER = "Concurrent Trigger"
    NONE = "None"

class StageMetrics(BaseModel):
    vol_delta_pct: float
    spread_delta_pct: float
    raw_vol: float
    raw_spread: float

class ResearchEvent(BaseModel):
    event_id: str
    hash: str
    previous_hash: str
    schema_version: str
    engine_version: str
    sequence_id: str
    symbol: str
    timeframe: str
    config_id: str
    timestamp: str
    action: str
    stage: str
    rule_evaluated: str
    matched: bool
    failure_reason: FailureReason
    metrics: StageMetrics
    research_id: str
    experiment_id: str
    pipeline_version: str
    dataset_checksum: str
    commit_sha: str

class ETESequence(BaseModel):
    sequence_id: str
    symbol: str
    timeframe: str
    config_id: str
    trigger_date: str
    config_version: str
    state: ETEState
    current_stage_index: int
    events: List[Dict[str, Any]]
    confidence_score: float
    started_at: str
    completed_at: str
    num_failures: int
    research_id: str
    experiment_id: str

class SequenceSummary(BaseModel):
    # Used for Dashboard Active/Completed listings
    sequence_id: str
    symbol: str
    timeframe: str
    state: ETEState
    current_stage: str
    confidence: float
    trigger_date: Optional[str] = None
    progress: List[Dict[str, Any]]

class DashboardManifest(BaseModel):
    schema_version: Union[str, int]
    engine_version: str
    generated_at: str
    research_events: int
    active_sequences: int
    completed_sequences: int
    failed_sequences: int = 0
    last_market_date: str
    files: Dict[str, str]

class SystemHealth(BaseModel):
    schema_version: str
    generated_at: str
    research_engine: str
    last_run: str
    integrity: str
    hash_failures: int
    active_sequences: int
    pipeline_seconds: float
    reconstruction_seconds: float
    publish_seconds: float
    # Operational Telemetry
    avg_confidence: float = 0.0
    repo_size_mb: float = 0.0
    dq_pass_rate: float = 1.0
    stage_durations: Dict[str, float] = {}
