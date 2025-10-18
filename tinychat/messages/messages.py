import time
from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass, field

from tinychat.utils.utils import random_id
from tinychat.messages.models import LLMServiceType

@dataclass
class Message:
    id: str = field(init=False)
    name: str = field(init=False)
    timestamp: int = field(init=False)
    metadata: Optional[Dict[str, Any]] = field(init=False)
    parent_id: Optional[str] = field(init=False)
    processor_path: List[str] = field(init=False)

    def __post_init__(self):
        self.id = random_id()
        self.name = f"{self.__class__.__name__}#{self.id}"
        self.timestamp = time.time_ns()
        self.metadata = None
        self.parent_id = None
        self.processor_path = []
    
    def derive(self, msg_type: type, **kwargs) -> 'Message':
        """
        Create a new message of any type, preserving causal lineage.
        
        Args:
            msg_type: The message class to create (e.g., AIMessage, UserMessage)
            **kwargs: Arguments to pass to the message constructor
        
        Example:
            ai_msg = user_msg.derive(AIMessage, 
                content="response",
                agent_id="agent-1",
                service=LLMServiceType.OPENAI,
                conversation_id=user_msg.conversation_id
            )
        """
        # Create new message of specified type
        new_msg = msg_type(**kwargs)
        new_msg.parent_id = self.id
        new_msg.processor_path = self.processor_path.copy()
        
        return new_msg


@dataclass
class IngressMessage(Message):
    content: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


@dataclass
class EgressMessage(Message):
    content: str
    conversation_id: str
    user_id: Optional[str] = None


@dataclass
class LLMMessage(Message):
    content: str
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
class ErrorMessage(Message):
    source: str
    content: str
    fatal: bool = False
    conversation_id: Optional[str] = None


@dataclass
class ControlMessage(Message):
    pass


@dataclass
class EventMessage(Message):
    """
    Custom event message for triggering handlers in the event router.

    Used when processors need to emit domain-specific events beyond
    the standard message types.

    Example:
        # Processor emits event when it needs data refresh
        event = EventMessage(
            conversation_id=self.conversation.conversation_id,
            event_name="needs_data_refresh",
            content={"reason": "user_data_stale"}
        )
        await self.emit_event(event)
    """

    event_name: str
    content: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
