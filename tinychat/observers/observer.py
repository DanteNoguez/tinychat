from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tinychat.messages import Message, MetricMessage

if TYPE_CHECKING:
    from tinychat.processors import MessageProcessor
    from tinychat.services.llm.models import LLMMessage, ToolCall, ToolCallOutput


@dataclass(frozen=True)
class MessageReceived(Message):
    source_processor: "MessageProcessor"
    source_message: Message


@dataclass(frozen=True)
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

    async def on_metric_recorded(self, metric: MetricMessage) -> None:
        """Optional hook to handle metrics."""


class LLMObserver(BaseObserver):
    @abstractmethod
    async def on_tool_call(self, tool_call: "ToolCall") -> None: ...

    @abstractmethod
    async def on_tool_result(self, tool_result: "ToolCallOutput") -> None: ...

    @abstractmethod
    async def on_llm_generation(self, llm_message: "LLMMessage") -> None: ...

    @abstractmethod
    async def on_context_update(self, messages: list["LLMMessage"]) -> None: ...

    async def on_llm_routing(self, message: Message) -> None:
        """Optional hook to handle routing messages."""
