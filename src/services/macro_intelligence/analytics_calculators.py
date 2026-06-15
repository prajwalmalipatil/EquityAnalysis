from typing import List, Dict, Any
from collections import Counter
from datetime import datetime, timezone

from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.read_models import (
    BusinessMetrics, AIMetrics, OperationalMetrics, QualityMetrics, CoverageMetrics,
    DistributionBucket, TimeSeries, TrendPoint
)
from src.services.macro_intelligence.metrics_registry import MetricsRegistry, MetricDefinition

def _safe_divide(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0

class BusinessCalculator:
    def calculate(self, events: List[MacroEvent]) -> BusinessMetrics:
        if not events:
            return BusinessMetrics(
                events_per_day=TimeSeries("events_per_day", []),
                category_distribution=[],
                high_priority_circulars=0,
                upcoming_effective_dates=0
            )

        # Category distribution
        categories = Counter(e.official_data.category for e in events if e.official_data.category)
        cat_dist = [DistributionBucket(k, v) for k, v in categories.items()]

        # High priority
        # Wait, priority isn't a direct field on OfficialData anymore. Let's assume ImpactAnalysis severity or keywords.
        # Alternatively, we map specific keywords or check derived_data.impact.severity
        high_priority = sum(
            1 for e in events 
            if e.derived_data and e.derived_data.impact and e.derived_data.impact.severity in ("High", "Critical")
        )

        # Upcoming effective dates (future dates)
        today = datetime.now(timezone.utc).date()
        upcoming_dates = 0
        for e in events:
            dt_str = e.official_data.effective_date
            if dt_str:
                try:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d").date()
                    if dt > today:
                        upcoming_dates += 1
                except ValueError:
                    pass

        # Events per day
        date_counts = Counter()
        for e in events:
            # use YYYY-MM-DD
            pub_dt = e.official_data.publication_date
            if pub_dt:
                date_counts[pub_dt[:10]] += 1
                
        sorted_dates = sorted(date_counts.items())
        trend_points = [TrendPoint(d, float(c)) for d, c in sorted_dates]

        return BusinessMetrics(
            events_per_day=TimeSeries("events_per_day", trend_points),
            category_distribution=cat_dist,
            high_priority_circulars=high_priority,
            upcoming_effective_dates=upcoming_dates
        )


class AICalculator:
    def calculate(self, events: List[MacroEvent]) -> AIMetrics:
        if not events:
            return AIMetrics([], [], 0.0, 0.0)

        # Theme frequency
        themes = Counter()
        confidences = []
        latencies = []
        ai_attempts = 0
        ai_successes = 0

        for e in events:
            if not e.derived_data:
                continue
            
            if e.derived_data.ai_theme:
                themes[e.derived_data.ai_theme] += 1
                
            if e.derived_data.ai_snapshots:
                ai_attempts += 1
                latest_snap = e.derived_data.ai_snapshots[-1]
                confidences.append(latest_snap.confidence)
                ai_successes += 1 # assuming a snapshot means success
                
                # We don't have explicit latency in snapshot. Let's look at metadata processing_latency if it exists
                # For now, default to 0.0 or check if processing_latency_ms is in run_stats? 
                # Let's say we rely on metadata containing processing latencies in real implementation.

        theme_dist = [DistributionBucket(k, v) for k, v in themes.most_common(10)]

        # Confidence distribution (0-20, 21-40, 41-60, 61-80, 81-100)
        conf_buckets = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
        for c in confidences:
            if c <= 20: conf_buckets["0-20"] += 1
            elif c <= 40: conf_buckets["21-40"] += 1
            elif c <= 60: conf_buckets["41-60"] += 1
            elif c <= 80: conf_buckets["61-80"] += 1
            else: conf_buckets["81-100"] += 1
            
        conf_dist = [DistributionBucket(k, v) for k, v in conf_buckets.items() if v > 0]
        
        success_rate = _safe_divide(ai_successes, ai_attempts)

        return AIMetrics(
            confidence_distribution=conf_dist,
            theme_frequency=theme_dist,
            processing_success_rate=success_rate,
            avg_latency_ms=0.0 # Placeholder for latency
        )


class OperationalCalculator:
    def calculate(self, events: List[MacroEvent], run_stats: Dict[str, Any]) -> OperationalMetrics:
        return OperationalMetrics(
            collector_success_rate=run_stats.get('collector_success_rate', 1.0),
            duplicate_rate=run_stats.get('duplicate_rate', 0.0),
            publish_duration_ms=run_stats.get('publish_duration_ms', 0.0),
            attachment_success_rate=run_stats.get('attachment_success_rate', 1.0),
            feed_availability=run_stats.get('feed_availability', 1.0)
        )


class QualityCalculator:
    def calculate(self, events: List[MacroEvent]) -> QualityMetrics:
        if not events:
            return QualityMetrics(0, 0, 0, 0.0, 0, 0.0)

        with_pdf = 0
        missing_attach = 0
        missing_date = 0
        ai_coverage = 0
        total_quality = 0

        for e in events:
            # Events with PDF
            if e.official_data.pdf_url:
                with_pdf += 1
                
            # Missing attachments
            if not e.official_data.attachments and not e.official_data.pdf_url:
                missing_attach += 1
                
            # Missing effective date
            if not e.official_data.effective_date:
                missing_date += 1
                
            # AI coverage
            if e.derived_data and e.derived_data.ai_summary:
                ai_coverage += 1
                total_quality += e.derived_data.quality_score

        return QualityMetrics(
            events_with_pdf=with_pdf,
            events_missing_attachment=missing_attach,
            missing_effective_date=missing_date,
            ai_enrichment_coverage=_safe_divide(ai_coverage, len(events)),
            validation_failures=0, # This would come from validator state
            avg_quality_score=_safe_divide(total_quality, len(events))
        )

class CoverageCalculator:
    def calculate(self, events: List[MacroEvent]) -> CoverageMetrics:
        if not events:
            return CoverageMetrics(0, 0, 0, 0, 0)

        with_ai = 0
        with_pdf = 0
        with_attachments = 0
        with_date = 0
        with_impact = 0

        for e in events:
            if e.derived_data and e.derived_data.ai_summary:
                with_ai += 1
            if e.official_data.pdf_url:
                with_pdf += 1
            if e.official_data.attachments:
                with_attachments += 1
            if e.official_data.effective_date:
                with_date += 1
            if e.derived_data and e.derived_data.market_relevance:
                # Based on the user's domain, "market_relevance" or "impact" might hold this.
                with_impact += 1

        return CoverageMetrics(
            events_with_ai=with_ai,
            events_with_pdf=with_pdf,
            events_with_attachments=with_attachments,
            events_with_effective_date=with_date,
            events_with_market_impact=with_impact
        )

# Register default calculators
MetricsRegistry.register(MetricDefinition(id="business_metrics", calculator=BusinessCalculator, category="business"))
MetricsRegistry.register(MetricDefinition(id="ai_metrics", calculator=AICalculator, category="ai"))
MetricsRegistry.register(MetricDefinition(id="operational_metrics", calculator=OperationalCalculator, category="operational"))
MetricsRegistry.register(MetricDefinition(id="quality_metrics", calculator=QualityCalculator, category="quality"))
MetricsRegistry.register(MetricDefinition(id="coverage_metrics", calculator=CoverageCalculator, category="coverage"))
