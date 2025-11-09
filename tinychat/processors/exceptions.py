from tinychat.messages.messages import Message
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinychat.processors.message_processor import MessageProcessor


class ProcessorException(Exception):
    def __init__(
        self,
        source_message: Message,
        source_processor: "MessageProcessor",
        details: str,
    ):
        self.source_message = source_message
        self.source_processor = source_processor
        super().__init__(
            f"Processor exception {source_message.name} at {source_processor.name} - {details}"
        )


class MaxHopsExceededError(ProcessorException):
    def __init__(
        self,
        source_message: Message,
        source_processor: "MessageProcessor",
        max_hops: int,
        current_hops: int,
    ):
        self.max_hops = max_hops
        self.current_hops = current_hops
        super().__init__(
            source_message=source_message,
            source_processor=source_processor,
            details=f"Message exceeded maximum hops: {current_hops}/{max_hops}",
        )
