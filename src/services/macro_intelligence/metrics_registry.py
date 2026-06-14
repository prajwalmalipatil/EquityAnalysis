from dataclasses import dataclass
from typing import Type, List

@dataclass(frozen=True)
class MetricDefinition:
    id: str
    calculator: Type
    category: str

class MetricsRegistry:
    _metrics: List[MetricDefinition] = []

    @classmethod
    def register(cls, metric: MetricDefinition):
        cls._metrics.append(metric)

    @classmethod
    def get_all(cls) -> List[MetricDefinition]:
        return cls._metrics

    @classmethod
    def get_by_category(cls, category: str) -> List[MetricDefinition]:
        return [m for m in cls._metrics if m.category == category]
