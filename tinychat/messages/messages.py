import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from tinychat.utils.utils import random_id
from tinychat.messages.models import LLMServiceType


@dataclass
class Message:
    content: str
    id: str = field(init=False)
    name: str = field(init=False)
    timestamp: int = field(init=False)

    def __post_init__(self):
        self.id = random_id()
        self.name = f"{self.__class__.__name__}#{self.id}"
        self.timestamp = time.monotonic_ns()


@dataclass
class IngressMessage(Message):
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


@dataclass
class EgressMessage(Message):
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class LLMMessage(Message):
    service: LLMServiceType
    conversation_id: str


@dataclass
class UserMessage(LLMMessage):
    user_id: Optional[str] = None


@dataclass
class AIMessage(LLMMessage):
    agent_id: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning: Optional[str] = None


@dataclass
class SystemMessage(LLMMessage):
    pass


@dataclass
class ControlMessage(Message):
    pass
