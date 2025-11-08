from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tinychat.messages.messages import Message

if TYPE_CHECKING:
    from tinychat.processors.message_processor import MessageProcessor


@dataclass
class MessageReceived(Message):
    source_processor: "MessageProcessor"
    source_message: Message


@dataclass
class MessageProcessed(Message):
    source_processor: "MessageProcessor"
    source_message: Message


class BaseObserver(ABC):
    @abstractmethod
    async def on_message_received(self, message: MessageReceived) -> None: ...

    @abstractmethod
    async def on_message_processed(self, message: MessageProcessed) -> None: ...

    @abstractmethod
    async def on_exception(
        self, source_message: Message, exception: Exception
    ) -> None: ...
