import json
import uuid
import hashlib
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from src.constants.vsa_constants import (
    ETE_EVENTS_DIR, ETE_SNAPSHOTS_DIR, CHECKPOINT_INTERVAL, ENGINE_VERSION,
    RULE_HVHS, RULE_LVLS, RULE_HVLS, RULE_LVHS
)
from src.models.ete_models import (
    ETEState, FailureReason, ResearchEvent, ETESequence, StageMetrics
)

class EigenTransitionEngineService:
    def __init__(self, timeframe: str, config_path: str = "src/constants/ete_sequences.json", events_dir: Optional[Path] = None):
        self.timeframe = timeframe.lower()
        self.events_dir = events_dir if events_dir else Path(ETE_EVENTS_DIR)
        self.snapshots_dir = self.events_dir / "snapshots" if events_dir else Path(ETE_SNAPSHOTS_DIR)
        self.events_file = self.events_dir / f"{self.timeframe}_events.jsonl"
        self.snapshots_file = self.snapshots_dir / f"{self.timeframe}_snapshots.json"
        
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequences_config = self._load_configs(config_path)

    def _load_configs(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _calculate_hash(self, event_dict: Dict[str, Any]) -> str:
        hashable = {k: v for k, v in event_dict.items() if k != 'hash'}
        canonical = json.dumps(hashable, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def _get_last_hash(self) -> str:
        if not self.events_file.exists() or self.events_file.stat().st_size == 0:
            return ""
        try:
            with open(self.events_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return ""
                return json.loads(lines[-1]).get('hash', "")
        except Exception:
            return ""

    def _append_event(self, event: ResearchEvent) -> None:
        event_dict = event.model_dump()
        if not event_dict.get('hash'):
            event_dict['hash'] = self._calculate_hash(event_dict)
            
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event_dict) + "\n")

    def _create_event(self, sequence_id: str, symbol: str, action: str, 
                      stage: str, rule_evaluated: str, matched: bool, 
                      failure_reason: FailureReason, metrics: StageMetrics, 
                      research_id: str, experiment_id: str,
                      timestamp: str, dataset_checksum: str = "NA") -> ResearchEvent:
        
        event_id_seed = f"{sequence_id}_{timestamp}_{action}_{stage}"
        deterministic_event_id = hashlib.sha256(event_id_seed.encode('utf-8')).hexdigest()

        event = ResearchEvent(
            event_id=deterministic_event_id,
            hash="",
            previous_hash=self._get_last_hash(),
            schema_version="1.0",
            engine_version=ENGINE_VERSION,
            sequence_id=sequence_id,
            symbol=symbol,
            timeframe=self.timeframe,
            timestamp=timestamp,
            action=action,
            stage=stage,
            rule_evaluated=rule_evaluated,
            matched=matched,
            failure_reason=failure_reason,
            metrics=metrics,
            research_id=research_id,
            experiment_id=experiment_id,
            pipeline_version="1.0.0",  # Mocked pipeline version
            dataset_checksum=dataset_checksum,
            commit_sha="UNKNOWN"  # Mocked commit sha
        )
        return event

    def detect_triggers(self, symbol: str, df: pd.DataFrame, eigen_filter_matched: bool):
        """Idempotent trigger generation."""
        if not eigen_filter_matched or df.empty:
            return
            
        # Determine the date column flexibly
        if df.index.name:
            trigger_date = str(df.iloc[-1].name).split()[0]
        elif 'Date' in df.columns:
            trigger_date = str(df['Date'].iloc[-1]).split()[0]
        elif 'YearWeek' in df.columns:
            trigger_date = str(df['YearWeek'].iloc[-1]).split()[0]
        elif 'YearMonth' in df.columns:
            trigger_date = str(df['YearMonth'].iloc[-1]).split()[0]
        else:
            trigger_date = "UNKNOWN_DATE"
        
        state_map, _ = self.reconstruct_state()
        
        for config_id, config in self.sequences_config.items():
            if not config.get('enabled', True):
                continue
            
            # Deterministic sequence_id
            seq_seed = f"{symbol}_{self.timeframe}_{trigger_date}_{config_id}"
            sequence_id = hashlib.sha256(seq_seed.encode('utf-8')).hexdigest()
            
            if sequence_id in state_map:
                continue # Idempotent
                
            metrics = StageMetrics(vol_delta_pct=0.0, spread_delta_pct=0.0, raw_vol=0.0, raw_spread=0.0)
            
            # Deterministic research ID
            research_id = hashlib.sha256(f"RESEARCH_{sequence_id}".encode()).hexdigest()
            experiment_id = "LIVE"
            
            event = self._create_event(
                sequence_id=sequence_id,
                symbol=symbol,
                action="INIT",
                stage="T",
                rule_evaluated="EIGEN_TRIGGER",
                matched=True,
                failure_reason=FailureReason.NONE,
                metrics=metrics,
                research_id=research_id,
                experiment_id=experiment_id,
                timestamp=f"{trigger_date}T00:00:00Z" # Deterministic trigger timestamp
            )
            self._append_event(event)

    def reconstruct_state(self) -> Tuple[Dict[str, ETESequence], int]:
        """Reads snapshots and tails events log."""
        sequences: Dict[str, ETESequence] = {}
        last_offset = 0
        last_hash = ""
        
        if self.snapshots_file.exists():
            try:
                with open(self.snapshots_file, 'r') as f:
                    snap = json.load(f)
                    last_offset = snap.get('last_processed_line_offset', 0)
                    for k, v in snap.get('sequences', {}).items():
                        sequences[k] = ETESequence(**v)
                    last_hash = snap.get('last_hash', "")
            except Exception:
                pass # Fallback to 0

        if not self.events_file.exists():
            return sequences, last_offset
            
        processed_lines = 0
        with open(self.events_file, 'r') as f:
            for _ in range(last_offset):
                f.readline()
                
            for line in f:
                if not line.strip():
                    continue
                processed_lines += 1
                ev_dict = json.loads(line)
                
                computed_hash = self._calculate_hash(ev_dict)
                seq_id = ev_dict.get('sequence_id')
                
                if computed_hash != ev_dict.get('hash') or (last_hash and ev_dict.get('previous_hash') != last_hash):
                    if seq_id in sequences:
                        sequences[seq_id].state = ETEState.INTEGRITY_FAILED
                    last_hash = ev_dict.get('hash', "")
                    continue
                    
                last_hash = computed_hash
                self._apply_event_to_state(ev_dict, sequences)
                
        new_offset = last_offset + processed_lines
        if processed_lines >= CHECKPOINT_INTERVAL:
            self._save_snapshot(sequences, new_offset, last_hash)
            
        return sequences, new_offset

    def _save_snapshot(self, sequences: Dict[str, ETESequence], offset: int, last_hash: str):
        snap = {
            "last_processed_line_offset": offset,
            "last_hash": last_hash,
            "sequences": {k: v.model_dump() for k, v in sequences.items()}
        }
        with open(self.snapshots_file, 'w') as f:
            json.dump(snap, f)

    def _apply_event_to_state(self, ev_dict: Dict[str, Any], sequences: Dict[str, ETESequence]):
        seq_id = ev_dict['sequence_id']
        action = ev_dict['action']
        
        if action == "INIT":
            sequences[seq_id] = ETESequence(
                sequence_id=seq_id,
                symbol=ev_dict['symbol'],
                timeframe=ev_dict['timeframe'],
                trigger_date=ev_dict['timestamp'].split('T')[0],
                config_version="1.0",
                state=ETEState.TRIGGERED,
                current_stage_index=0,
                events=[ev_dict],
                confidence_score=0.0,
                started_at=ev_dict['timestamp'],
                completed_at="",
                num_failures=0,
                research_id=ev_dict.get('research_id', ''),
                experiment_id=ev_dict.get('experiment_id', '')
            )
        elif seq_id in sequences:
            seq = sequences[seq_id]
            if seq.state == ETEState.INTEGRITY_FAILED:
                return # Halts
            
            seq.events.append(ev_dict)
            
            if action == "ADVANCE":
                seq.current_stage_index += 1
                seq.state = ETEState.WAITING
            elif action == "FAIL":
                seq.state = ETEState.FAILED
                seq.num_failures += 1
                seq.completed_at = ev_dict['timestamp']
            elif action == "COMPLETE":
                seq.state = ETEState.COMPLETED
                seq.completed_at = ev_dict['timestamp']
            elif action == "PAUSE":
                seq.state = ETEState.PAUSED
            elif action == "EXPIRE":
                seq.state = ETEState.EXPIRED
                seq.completed_at = ev_dict['timestamp']

    def update_active_sequences(self, symbol: str, df: pd.DataFrame):
        sequences, _ = self.reconstruct_state()
        
        active_seqs = [s for s in sequences.values() if s.symbol == symbol and s.state in (ETEState.TRIGGERED, ETEState.WAITING, ETEState.PAUSED)]
        if not active_seqs:
            return
            
        if len(df) < 2:
            for seq in active_seqs:
                self._handle_pause(seq, FailureReason.DATA_UNAVAILABLE)
            return
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Determine the date column flexibly
        if df.index.name:
            current_date = str(current.name).split()[0]
        elif 'Date' in current:
            current_date = str(current['Date']).split()[0]
        elif 'YearWeek' in current:
            current_date = str(current['YearWeek']).split()[0]
        elif 'YearMonth' in current:
            current_date = str(current['YearMonth']).split()[0]
        else:
            current_date = "UNKNOWN_DATE"
        
        for seq in active_seqs:
            self._evaluate_sequence(seq, current, previous, current_date)

    def _handle_pause(self, seq: ETESequence, reason: FailureReason):
        metrics = StageMetrics(vol_delta_pct=0.0, spread_delta_pct=0.0, raw_vol=0.0, raw_spread=0.0)
        event = self._create_event(
            sequence_id=seq.sequence_id,
            symbol=seq.symbol,
            action="PAUSE",
            stage=f"T+{seq.current_stage_index+1}",
            rule_evaluated="NONE",
            matched=False,
            failure_reason=reason,
            metrics=metrics,
            research_id=seq.research_id,
            experiment_id=seq.experiment_id,
            timestamp=f"UNKNOWN_T00:00:00Z" # Pauses happen when data is unavailable, hard to determine time, but typically we want deterministic
        )
        self._append_event(event)

    def _evaluate_sequence(self, seq: ETESequence, current: pd.Series, previous: pd.Series, current_date: str):
        config = self.sequences_config.get("AVE_1", {}) # Currently hardcoded to single test config
        target_rules = config.get("sequence", ["HVHS", "LVLS", "HVHS", "LVLS"])
        
        if seq.current_stage_index >= len(target_rules):
            return
            
        target_rule = target_rules[seq.current_stage_index]
        stage_name = f"T+{seq.current_stage_index+1}"
        
        prev_vol = max(float(previous.get("Volume", 1.0)), 1.0)
        curr_vol = float(current.get("Volume", 1.0))
        vol_delta_pct = ((curr_vol - prev_vol) / prev_vol) * 100.0
        
        prev_spread = max(float(previous.get("High", 1.0) - previous.get("Low", 1.0)), 0.01)
        curr_spread = float(current.get("High", 1.0) - current.get("Low", 1.0))
        spread_delta_pct = ((curr_spread - prev_spread) / prev_spread) * 100.0
        
        metrics = StageMetrics(
            vol_delta_pct=vol_delta_pct,
            spread_delta_pct=spread_delta_pct,
            raw_vol=curr_vol,
            raw_spread=curr_spread
        )
        
        # Rule Evaluation logic
        matched = False
        failure_reason = FailureReason.NONE
        
        if target_rule == RULE_HVHS:
            if vol_delta_pct > 0 and spread_delta_pct > 0:
                matched = True
            else:
                failure_reason = FailureReason.VOLUME_MISMATCH if vol_delta_pct <= 0 else FailureReason.SPREAD_MISMATCH
        elif target_rule == RULE_LVLS:
            if vol_delta_pct < 0 and spread_delta_pct < 0:
                matched = True
            else:
                failure_reason = FailureReason.VOLUME_MISMATCH if vol_delta_pct >= 0 else FailureReason.SPREAD_MISMATCH
        
        # Create event
        action = "ADVANCE" if matched else "FAIL"
        is_complete = matched and (seq.current_stage_index + 1 == len(target_rules))
        if is_complete:
            action = "COMPLETE"
            
        event = self._create_event(
            sequence_id=seq.sequence_id,
            symbol=seq.symbol,
            action=action,
            stage=stage_name,
            rule_evaluated=target_rule,
            matched=matched,
            failure_reason=failure_reason,
            metrics=metrics,
            research_id=seq.research_id,
            experiment_id=seq.experiment_id,
            timestamp=f"{current_date}T00:00:00Z"
        )
        self._append_event(event)

from src.services.orchestration.registry import platform_registry, ResearchModule
platform_registry.register(ResearchModule(
    name="EigenTransitionEngine",
    version="1.0.0",
    description="Event-sourced state machine for quantitative transitions.",
    inputs=["Signals"],
    outputs=["ResearchEvents"],
    dependencies=["VSAProcessorService"]
))
