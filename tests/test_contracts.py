import pytest
from typing import List
from src.services.macro_intelligence.interfaces import EventReadRepository, EventWriteRepository
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.read_models import DashboardBundle, ManifestViewModel, AnalyticsReadModel

# --- Contract Tests for Repositories ---

def contract_test_event_read_repository(repo: EventReadRepository):
    """
    Contract: Any EventReadRepository must return lists of MacroEvent objects
    and correctly filter for active events.
    """
    all_events = repo.get_all_events()
    assert isinstance(all_events, list)
    if all_events:
        assert isinstance(all_events[0], MacroEvent)
        
    active_events = repo.get_active_events()
    assert isinstance(active_events, list)
    for e in active_events:
        assert e.metadata.lifecycle_status == "ACTIVE"


def contract_test_event_write_repository(repo: EventWriteRepository, sample_event: MacroEvent):
    """
    Contract: Any EventWriteRepository must handle exact duplicates safely,
    returning True on initial save and False on subsequent identical saves.
    """
    # First save should ideally be True if empty, but we just verify it doesn't crash
    result1 = repo.save_event(sample_event)
    assert isinstance(result1, bool)
    
    # Second save of the exact same event must be False (deduplicated)
    result2 = repo.save_event(sample_event)
    assert result2 is False


# --- Contract Tests for Builders ---

class DashboardBuilderContract:
    def build(self) -> DashboardBundle:
        raise NotImplementedError

def contract_test_dashboard_builder(builder: DashboardBuilderContract):
    """
    Contract: Any DashboardBuilder must output a well-formed DashboardBundle
    containing manifest, analytics, and events.
    """
    bundle = builder.build()
    assert isinstance(bundle, DashboardBundle)
    assert isinstance(bundle.manifest, ManifestViewModel)
    assert isinstance(bundle.analytics, AnalyticsReadModel)
    assert isinstance(bundle.events, list)


# --- Contract Tests for Publishers ---

class PublisherContract:
    def publish(self, bundle: DashboardBundle) -> None:
        raise NotImplementedError

def contract_test_publisher(publisher: PublisherContract, sample_bundle: DashboardBundle):
    """
    Contract: Publishers must only serialize the bundle without throwing errors.
    They should not contain any business logic to alter the bundle.
    """
    try:
        publisher.publish(sample_bundle)
    except Exception as e:
        pytest.fail(f"Publisher raised an exception during serialization: {e}")
