from typing import List, Dict, Any
from datetime import datetime

from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.read_models import AnalyticsReadModel
from src.services.macro_intelligence.analytics_calculators import (
    BusinessCalculator, AICalculator, OperationalCalculator, QualityCalculator, CoverageCalculator
)

class AnalyticsProvider:
    """
    Orchestrates the individual calculators to compute the overarching AnalyticsReadModel.
    Contains no formatting or JSON output logic.
    """
    def __init__(self):
        # We could instantiate these dynamically from MetricsRegistry, 
        # but for strong typing in AnalyticsReadModel we keep direct references.
        self.business_calc = BusinessCalculator()
        self.ai_calc = AICalculator()
        self.ops_calc = OperationalCalculator()
        self.quality_calc = QualityCalculator()
        self.coverage_calc = CoverageCalculator()
        self.version = 1
        self._cache = {}

    def _compute_hash(self, events: List[MacroEvent]) -> str:
        import hashlib
        # Hash based on event IDs and their updated_at timestamps (if available)
        hash_input = "".join(f"{e.event_id}" for e in events)
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def compute(self, events: List[MacroEvent], run_stats: Dict[str, Any] = None) -> AnalyticsReadModel:
        if run_stats is None:
            run_stats = {}

        event_hash = self._compute_hash(events)
        if event_hash in self._cache:
            return self._cache[event_hash]

        business_metrics = self.business_calc.calculate(events)
        ai_metrics = self.ai_calc.calculate(events)
        ops_metrics = self.ops_calc.calculate(events, run_stats)
        quality_metrics = self.quality_calc.calculate(events)

        ops_metrics = self.ops_calc.calculate(events, run_stats)
        quality_metrics = self.quality_calc.calculate(events)
        coverage_metrics = self.coverage_calc.calculate(events)

        model = AnalyticsReadModel(
            version=self.version,
            generated_at=datetime.utcnow().isoformat() + "Z",
            total_events=len(events),
            business=business_metrics,
            ai=ai_metrics,
            operational=ops_metrics,
            quality=quality_metrics,
            coverage=coverage_metrics
        )
        self._cache[event_hash] = model
        return model
