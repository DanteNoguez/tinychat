import asyncio
from typing import Optional

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    Message,
)
from loguru import logger
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.observers.observer import BaseObserver, MessageReceived, MessageProcessed

from tinychat.utils.logging import configure_pretty_logging

configure_pretty_logging(debug_level=10)


class EchoMessage(Message):
    pass


class LoggingObserver(BaseObserver):
    async def on_message_received(self, message: MessageReceived) -> None:
        logger.info(
            f"📨 [{message.source_processor.name}] Received: {message.source_message.name} with content {message.content} at {message.source_message.timestamp}"
        )

    async def on_message_processed(self, message: MessageProcessed) -> None:
        logger.info(
            f"✅ [{message.source_processor.name}] Processed: {message.source_message.name} with result {message.content} at {message.source_message.timestamp}"
        )

    async def on_exception(self, source_message: Message, exception: Exception) -> None:
        logger.error(
            f"❌ [{source_message.name}] Exception: {exception} at {source_message.timestamp}"
        )


class EchoProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, IngressMessage):
            return EchoMessage(
                content=message.content,
            )
        else:
            raise ValueError(f"Unexpected message: {message}")


class TransformerProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, EchoMessage):
            return EgressMessage(
                content="Transformed!",
            )
        else:
            raise ValueError(f"Unexpected message: {message}")


async def main():
    # Get the running event loop and setup task manager
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
        observers=[LoggingObserver()],
    )
    # Create processors
    echo = EchoProcessor(name="echo")
    transformer = TransformerProcessor(name="transformer")

    # Setup message bus with type-based routing
    bus = CompositeProcessor(
        handlers={
            IngressMessage: echo,
            EchoMessage: transformer,
        },
    )
    await bus.setup(config)

    # Create and process ingress message
    message = IngressMessage(
        content="Hello, tinychat!",
        conversation_id="demo",
    )

    await bus.process(message)


if __name__ == "__main__":
    asyncio.run(main())
