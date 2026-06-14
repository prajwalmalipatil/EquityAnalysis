import pytest
from pathlib import Path
import json
import time

from src.services.macro_intelligence.models import MacroEvent, OfficialData, DerivedData
from src.services.macro_intelligence.analytics_provider import AnalyticsProvider
from src.services.macro_intelligence.builders import AnalyticsBuilder
from src.services.macro_intelligence.publishers import AnalyticsPublisher
from src.services.macro_intelligence.publish_pipeline import run_publish_pipeline
from src.services.macro_intelligence.read_models import AnalyticsViewModel

@pytest.fixture
def empty_output_dir(tmp_path):
    return tmp_path

def test_empty_repository(empty_output_dir):
    provider = AnalyticsProvider()
    read_model = provider.compute([])
    assert read_model.total_events == 0
    view_model = AnalyticsBuilder.build(read_model)
    assert view_model.event_count == 0

    publisher = AnalyticsPublisher(empty_output_dir)
    publisher.publish(view_model)
    
    assert (empty_output_dir / "analytics.json").exists()
    assert (empty_output_dir / "analytics-history.json").exists()

def test_corrupted_events():
    # Events missing attachments, dates, and AI logic
    corrupted_event_1 = MacroEvent(
        event_id="corrupted_1",
        official_data=OfficialData(
            title="Empty title",
            publication_date="",
            category="",
            source="RBI",
            official_url="http://rbi",
            content="",
            attachments=[],
            pdf_url=""
        ),
        derived_data=None,
        metadata=None
    )
    provider = AnalyticsProvider()
    read_model = provider.compute([corrupted_event_1])
    # Should safely handle empty values without division by zero
    assert read_model.total_events == 1
    assert read_model.quality.avg_quality_score == 0.0

def test_determinism(empty_output_dir):
    event = MacroEvent(
        event_id="det_1",
        official_data=OfficialData(
            title="Title",
            publication_date="2026-06-14T00:00:00",
            category="Press",
            source="RBI",
            official_url="http://rbi",
            content="Content",
            attachments=[],
            pdf_url=""
        ),
        derived_data=None,
        metadata=None
    )
    provider = AnalyticsProvider()
    
    read_model_1 = provider.compute([event])
    # Force identical timestamps for testing byte-for-byte determinism
    import dataclasses
    read_model_1 = dataclasses.replace(read_model_1, generated_at="2026-06-14T00:00:00Z")
    view_model_1 = AnalyticsBuilder.build(read_model_1)
    
    read_model_2 = provider.compute([event])
    read_model_2 = dataclasses.replace(read_model_2, generated_at="2026-06-14T00:00:00Z")
    view_model_2 = AnalyticsBuilder.build(read_model_2)

    dir_1 = empty_output_dir / "run1"
    dir_2 = empty_output_dir / "run2"
    AnalyticsPublisher(dir_1).publish(view_model_1)
    AnalyticsPublisher(dir_2).publish(view_model_2)
    
    with open(dir_1 / "analytics.json", "r") as f1, open(dir_2 / "analytics.json", "r") as f2:
        assert f1.read() == f2.read()

def test_performance():
    events = [
        MacroEvent(
            event_id=f"perf_{i}",
            official_data=OfficialData(
                title=f"Title {i}",
                publication_date="2026-06-14T00:00:00",
                category="Press",
                source="RBI",
                official_url="http://rbi",
                content="Content",
                attachments=[],
                pdf_url=""
            ),
            derived_data=None,
            metadata=None
        )
        for i in range(10000)
    ]
    
    start_time = time.time()
    provider = AnalyticsProvider()
    read_model = provider.compute(events)
    view_model = AnalyticsBuilder.build(read_model)
    end_time = time.time()
    
    # Computation for 10k events should take less than 1 second (pure math operations)
    assert (end_time - start_time) < 1.0
