import os
import json
import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.constants.vsa_constants import ENGINE_VERSION
from src.models.ete_models import (
    ETESequence, SequenceSummary, DashboardManifest, SystemHealth, ETEState
)
from src.services.vsa.eigen_transition_engine_service import EigenTransitionEngineService

class ViewBuilderService:
    def __init__(self, base_output_dir: Path, events_dir: Optional[Path] = None):
        self.base_output = base_output_dir
        self.events_dir = events_dir
        self.dashboard_live = self.base_output / "dashboard"
        self.dashboard_next = self.base_output / "dashboard_next"

    def _get_content_hash(self, data: str) -> str:
        return hashlib.md5(data.encode('utf-8')).hexdigest()[:6]

    def _write_hashed_json(self, dir_path: Path, prefix: str, data: Any) -> str:
        json_str = json.dumps(data, separators=(',', ':'))
        h = self._get_content_hash(json_str)
        filename = f"{prefix}.{h}.json"
        with open(dir_path / filename, 'w') as f:
            f.write(json_str)
        return filename

    def publish(self, pipeline_seconds: float = 0.0):
        t0 = time.time()
        
        # 1. Reconstruct all timeframes
        sequences = {}
        for tf in ["daily", "weekly", "monthly"]:
            engine = EigenTransitionEngineService(tf, events_dir=self.events_dir)
            seqs, _ = engine.reconstruct_state()
            sequences.update(seqs)

        reconstruction_seconds = time.time() - t0
        t1 = time.time()

        # 2. Setup dashboard_next
        if self.dashboard_next.exists():
            shutil.rmtree(self.dashboard_next)
        self.dashboard_next.mkdir(parents=True, exist_ok=True)
        (self.dashboard_next / "history").mkdir(parents=True, exist_ok=True)

        # Copy static UI assets from dashboard_live if they exist, to prevent destroying the UI
        if self.dashboard_live.exists():
            for asset in [
                "index.html", "app.js", "styles.css", "data.json", "backtest_results.json",
                "analytics.json", "search-index.json", "relationships.json", "graph.json",
                "analytics-history.json", "system_health.json"
            ]:
                asset_path = self.dashboard_live / asset
                if asset_path.exists():
                    shutil.copy(asset_path, self.dashboard_next / asset)
            
            # Preserve ETE content-hashed files (Task 6.8)
            for item in self.dashboard_live.iterdir():
                if item.is_file():
                    is_ete_file = (
                        item.name.startswith("summary.") or
                        item.name.startswith("indicator_catalog.") or
                        item.name.startswith("history_index.")
                    ) and item.name.endswith(".json")
                    if is_ete_file:
                        shutil.copy(item, self.dashboard_next / item.name)
            
            # Preserve JSONPublisher history (Time Travel)
            history_dir = self.dashboard_live / "history"
            if history_dir.exists():
                next_history_dir = self.dashboard_next / "history"
                next_history_dir.mkdir(parents=True, exist_ok=True)
                for item in history_dir.iterdir():
                    if item.is_file() and (item.name == "index.json" or item.name.startswith("data_") or item.name.startswith("analytics_") or item.name.startswith("manifest_") or item.name == "rbi_events.jsonl"):
                        shutil.copy(item, next_history_dir / item.name)

        # 3. Build data structures
        summary_data = {
            "active": [],
            "completed": [],
            "failed": []
        }
        
        history_map = {}
        integrity_failures = 0

        for seq_id, seq in sequences.items():
            if seq.state == ETEState.INTEGRITY_FAILED:
                integrity_failures += 1
                continue
                
            sym = seq.symbol
            if sym not in history_map:
                history_map[sym] = []
            
            history_map[sym].append(seq.model_dump())

            summary = SequenceSummary(
                sequence_id=seq.sequence_id,
                symbol=seq.symbol,
                timeframe=seq.timeframe,
                state=seq.state,
                current_stage=f"T+{seq.current_stage_index}" if seq.state != ETEState.COMPLETED else "Completed",
                confidence=seq.confidence_score,
                progress=seq.events
            ).model_dump()

            if seq.state in (ETEState.TRIGGERED, ETEState.WAITING, ETEState.PAUSED):
                summary_data["active"].append(summary)
            elif seq.state == ETEState.COMPLETED:
                summary_data["completed"].append(summary)
            elif seq.state in (ETEState.FAILED, ETEState.EXPIRED):
                summary_data["failed"].append(summary)

        # 4. Write Content-Hashed Files
        files_manifest = {}
        
        # Summary
        files_manifest["summary"] = self._write_hashed_json(self.dashboard_next, "summary", summary_data)
        
        # Indicator Catalog
        engine_tmp = EigenTransitionEngineService("daily")
        files_manifest["indicator_catalog"] = self._write_hashed_json(self.dashboard_next, "indicator_catalog", engine_tmp.sequences_config)
        
        # History
        history_manifest = {}
        for sym, seqs in history_map.items():
            first_letter = sym[0].upper() if sym else "_"
            sym_dir = self.dashboard_next / "history" / first_letter
            sym_dir.mkdir(parents=True, exist_ok=True)
            
            h_filename = self._write_hashed_json(sym_dir, sym, seqs)
            history_manifest[sym] = f"history/{first_letter}/{h_filename}"
            
        files_manifest["history_index"] = self._write_hashed_json(self.dashboard_next, "history_index", history_manifest)

        publish_seconds = time.time() - t1

        # Calculate avg confidence
        conf_scores = [s.confidence_score for s in sequences.values() if s.state in (ETEState.TRIGGERED, ETEState.WAITING, ETEState.PAUSED)]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        # Read Orchestrator Metrics
        metrics_file = self.base_output.parent / "orchestrator_metrics.json"
        repo_size = 0.0
        dq_rate = 1.0
        stages = {}
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                repo_size = metrics.get("repository_size_mb", 0.0)
                stages = metrics.get("stages", {})
                dq_stats = metrics.get("data_quality", {})
                total_files = dq_stats.get("total_files", 0)
                if total_files > 0:
                    dq_rate = dq_stats.get("passed", 0) / total_files
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        # 5. Write Health and Manifest (Stable filenames)
        health = SystemHealth(
            schema_version="1.0",
            generated_at=datetime.now(timezone.utc).isoformat(),
            research_engine=ENGINE_VERSION,
            last_run=datetime.now(timezone.utc).isoformat(),
            integrity="PASS" if integrity_failures == 0 else "WARNING",
            hash_failures=integrity_failures,
            active_sequences=len(summary_data["active"]),
            pipeline_seconds=pipeline_seconds,
            reconstruction_seconds=reconstruction_seconds,
            publish_seconds=publish_seconds,
            avg_confidence=round(avg_conf, 2),
            repo_size_mb=repo_size,
            dq_pass_rate=round(dq_rate, 4),
            stage_durations=stages
        )
        with open(self.dashboard_next / "system_health.json", "w") as f:
            f.write(json.dumps(health.model_dump()))

        manifest = DashboardManifest(
            schema_version="1.0",
            engine_version=ENGINE_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            research_events=sum(len(s.events) for s in sequences.values()),
            active_sequences=len(summary_data["active"]),
            completed_sequences=len(summary_data["completed"]),
            failed_sequences=len(summary_data["failed"]),
            last_market_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            files=files_manifest
        )
        with open(self.dashboard_next / "manifest.json", "w") as f:
            f.write(json.dumps(manifest.model_dump()))

        # Validate Data Contracts before swapping
        from src.services.reporting.data_contract_validator import DataContractValidator
        validator = DataContractValidator(self.dashboard_next)
        if not validator.validate_all():
            raise Exception("DATA_CONTRACT_VIOLATION: JSON Schema validation failed. Aborting atomic swap.")

        # 6. Atomic Swap
        try:
            if self.dashboard_live.exists():
                # On some OS rename to existing directory fails, better to remove or move old
                backup = self.base_output / "dashboard_bak"
                if backup.exists():
                    shutil.rmtree(backup)
                self.dashboard_live.rename(backup)
            
            self.dashboard_next.rename(self.dashboard_live)
        except OSError as e:
            # If rename fails, we keep what we had (or restore)
            if 'backup' in locals() and backup.exists() and not self.dashboard_live.exists():
                backup.rename(self.dashboard_live)
            raise Exception(f"Atomic swap failed: {e}")
