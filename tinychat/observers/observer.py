from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from tinychat.messages.messages import Message

if TYPE_CHECKING:
    from tinychat.processors.message_processor import MessageProcessor


@dataclass
class MessageReceived:
    processor: "MessageProcessor"
    message: Message
    timestamp: int


@dataclass
class MessageProcessed:
    processor: "MessageProcessor"
    message: Message
    result: Optional[Message]
    timestamp: int
    duration_ns: int


@dataclass
class ProcessorCalled:
    source: "MessageProcessor"
    target: "MessageProcessor"
    message: Message
    timestamp: int


@dataclass
class StateChanged:
    conversation_id: str
    previous_state: str
    new_state: str
    timestamp: int
    metadata: dict


class BaseObserver(ABC):
    @abstractmethod
    async def on_message_received(self, data: MessageReceived):
        pass

    @abstractmethod
    async def on_message_processed(self, data: MessageProcessed):
        pass

    @abstractmethod
    async def on_processor_called(self, data: ProcessorCalled):
        pass

    @abstractmethod
    async def on_state_changed(self, data: StateChanged):
        pass


class Observer(BaseObserver):
    async def on_message_received(self, data: MessageReceived):
        pass

    async def on_message_processed(self, data: MessageProcessed):
        pass

    async def on_processor_called(self, data: ProcessorCalled):
        pass

    async def on_state_changed(self, data: StateChanged):
        pass
