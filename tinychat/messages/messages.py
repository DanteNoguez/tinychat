import time
from dataclasses import dataclass, field

from typing import Literal
from tinychat.utils.utils import random_id


@dataclass(frozen=True)
class Message:
    content: str
    id: str = field(init=False)
    name: str = field(init=False)
    timestamp: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "id", random_id())
        object.__setattr__(self, "name", f"{self.__class__.__name__}#{self.id}")
        object.__setattr__(self, "timestamp", time.monotonic_ns())


@dataclass(frozen=True)
class IngressMessage(Message): ...


@dataclass(frozen=True)
class EgressMessage(Message): ...


@dataclass(frozen=True)
class ControlMessage(Message): ...


@dataclass(frozen=True)
class MetricMessage(Message):
    metric_name: str
    metric_value: float
    metric_unit: Literal["ns", "μs", "ms", "s", "min", "hour", "day"]
    metric_timestamp: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "metric_timestamp", time.monotonic_ns())
