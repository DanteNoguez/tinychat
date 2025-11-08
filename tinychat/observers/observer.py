from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tinychat.messages.messages import ErrorMessage, Message

if TYPE_CHECKING:
    from tinychat.processors.message_processor import MessageProcessor


@dataclass
class MessageReceived(Message):
    processor: "MessageProcessor"
    message: Message


@dataclass
class MessageProcessed(Message):
    processor: "MessageProcessor"
    message: Message


class BaseObserver(ABC):
    @abstractmethod
    async def on_message_received(self, message: MessageReceived) -> None: ...

    @abstractmethod
    async def on_message_processed(self, message: MessageProcessed) -> None: ...

    @abstractmethod
    async def on_error_message(
        self, source_message: Message, error_msg: ErrorMessage
    ) -> None: ...
