from dataclasses import dataclass
from typing import Type, List, ClassVar, Set

@dataclass(frozen=True)
class MetricDefinition:
    id: str
    calculator: Type
    category: str

class MetricsRegistry:
    _metrics: ClassVar[List[MetricDefinition]] = []
    _registered_ids: ClassVar[Set[str]] = set()

    @classmethod
    def register(cls, metric: MetricDefinition):
        if metric.id not in cls._registered_ids:
            cls._metrics.append(metric)
            cls._registered_ids.add(metric.id)

    @classmethod
    def get_all(cls) -> List[MetricDefinition]:
        return cls._metrics

    @classmethod
    def get_by_category(cls, category: str) -> List[MetricDefinition]:
        return [m for m in cls._metrics if m.category == category]

    @classmethod
    def reset(cls):
        """For testing: clears all registered metrics."""
        cls._metrics.clear()
        cls._registered_ids.clear()
