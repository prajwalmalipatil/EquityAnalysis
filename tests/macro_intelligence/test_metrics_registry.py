import pytest
from src.services.macro_intelligence.metrics_registry import MetricsRegistry, MetricDefinition

def test_metrics_registry_idempotency_and_reset():
    # Clear registry first
    MetricsRegistry.reset()
    assert len(MetricsRegistry.get_all()) == 0
    
    # Define metric definitions
    m1 = MetricDefinition(id="m1", calculator=object, category="test_cat")
    m2 = MetricDefinition(id="m2", calculator=object, category="test_cat")
    
    # Register m1
    MetricsRegistry.register(m1)
    assert len(MetricsRegistry.get_all()) == 1
    
    # Register m1 again - should not duplicate
    MetricsRegistry.register(m1)
    assert len(MetricsRegistry.get_all()) == 1
    
    # Register m2
    MetricsRegistry.register(m2)
    assert len(MetricsRegistry.get_all()) == 2
    
    # Get by category
    cats = MetricsRegistry.get_by_category("test_cat")
    assert len(cats) == 2
    
    # Reset and verify
    MetricsRegistry.reset()
    assert len(MetricsRegistry.get_all()) == 0
