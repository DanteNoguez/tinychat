import time
from typing import Optional
from dataclasses import dataclass, field

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
class IngressMessage(Message):
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


@dataclass(frozen=True)
class EgressMessage(Message):
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


@dataclass(frozen=True)
class ControlMessage(Message): ...
