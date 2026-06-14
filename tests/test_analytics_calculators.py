import pytest
from datetime import datetime

from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData, EventMetadata, ImpactAnalysis
from src.services.macro_intelligence.analytics_calculators import (
    BusinessCalculator, AICalculator, QualityCalculator, OperationalCalculator
)

def build_mock_event(
    event_id="1", 
    category="Policy", 
    severity="Low", 
    effective_date=None, 
    pub_date="2023-10-01T10:00:00Z",
    has_ai=False,
    confidence=0,
    quality_score=0,
    has_pdf=False,
    attachments=None
):
    attachments = attachments or []
    impact = ImpactAnalysis.from_dict({"severity": severity}) if severity else None
    
    official_data = OfficialData.from_dict({
        "title": "Test",
        "category": category,
        "publication_date": pub_date,
        "effective_date": effective_date,
        "pdf_url": "http://test.pdf" if has_pdf else None,
        "attachments": attachments
    })
    
    derived_data = DerivedData.from_dict({
        "ai_summary": "Test Summary" if has_ai else None,
        "quality_score": quality_score
    })
    derived_data.impact = impact
    
    if has_ai:
        from src.services.macro_intelligence.models import AISummarySnapshot
        derived_data.ai_snapshots = [AISummarySnapshot.from_dict({"confidence": confidence})]
        derived_data.ai_theme = "Inflation"

    metadata = EventMetadata.from_dict({})
    
    return MacroEvent(event_id=event_id, official_data=official_data, derived_data=derived_data, metadata=metadata)

def test_business_calculator_empty():
    calc = BusinessCalculator()
    res = calc.calculate([])
    assert res.high_priority_circulars == 0
    assert len(res.category_distribution) == 0

def test_business_calculator_logic():
    calc = BusinessCalculator()
    events = [
        build_mock_event("1", "Policy", "High", "2099-01-01", "2023-10-01"),
        build_mock_event("2", "Policy", "Low", "2020-01-01", "2023-10-01"),
        build_mock_event("3", "Press Release", "Critical", None, "2023-10-02"),
    ]
    
    res = calc.calculate(events)
    assert res.high_priority_circulars == 2
    assert res.upcoming_effective_dates == 1
    
    # Category dist
    cats = {b.label: b.count for b in res.category_distribution}
    assert cats["Policy"] == 2
    assert cats["Press Release"] == 1
    
    # Time series
    assert len(res.events_per_day.points) == 2
    assert res.events_per_day.points[0].date == "2023-10-01"
    assert res.events_per_day.points[0].value == 2.0

def test_ai_calculator_confidence_distribution():
    calc = AICalculator()
    events = [
        build_mock_event("1", has_ai=True, confidence=15),
        build_mock_event("2", has_ai=True, confidence=50),
        build_mock_event("3", has_ai=True, confidence=90),
        build_mock_event("4", has_ai=False),
    ]
    
    res = calc.calculate(events)
    # 3 attempts, 3 success = 1.0 success rate
    assert res.processing_success_rate == 1.0
    
    conf_buckets = {b.label: b.count for b in res.confidence_distribution}
    assert conf_buckets.get("0-20") == 1
    assert conf_buckets.get("41-60") == 1
    assert conf_buckets.get("81-100") == 1

def test_quality_calculator_averages():
    calc = QualityCalculator()
    events = [
        build_mock_event("1", has_pdf=True, has_ai=True, quality_score=100),
        build_mock_event("2", has_pdf=False, attachments=["link1"], has_ai=False, quality_score=0),
        build_mock_event("3", has_pdf=False, attachments=[], has_ai=True, quality_score=50),
    ]
    
    res = calc.calculate(events)
    assert res.events_with_pdf == 1
    assert res.events_missing_attachment == 1 # Event 3
    assert res.missing_effective_date == 3
    assert res.ai_enrichment_coverage == (2 / 3)
    assert res.avg_quality_score == 50.0 # (100 + 0 + 50) / 3

def test_operational_calculator():
    calc = OperationalCalculator()
    run_stats = {"publish_duration_ms": 150.5, "duplicate_rate": 0.05}
    res = calc.calculate([], run_stats)
    assert res.publish_duration_ms == 150.5
    assert res.duplicate_rate == 0.05
    assert res.collector_success_rate == 1.0 # default fallback
