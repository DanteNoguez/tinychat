import asyncio
from typing import Optional
from dataclasses import dataclass
from loguru import logger

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    Message,
)
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.observers.observer import (
    BaseObserver,
    MessageReceived,
    MessageProcessed,
)


@dataclass(frozen=True)
class BotMessage(Message):
    content: str
    iteration: int


@dataclass(frozen=True)
class UserMessage(Message):
    content: str
    iteration: int


class LoggingObserver(BaseObserver):
    async def on_message_received(self, message: MessageReceived) -> None:
        logger.info(f"📨 [{message.source_processor.name}] Received: {message.content}")

    async def on_message_processed(self, message: MessageProcessed) -> None:
        logger.info(f"✅ [{message.source_processor.name}] Returned: {message.content}")

    async def on_exception(self, source_message: Message, exception: Exception) -> None:
        logger.error(
            f"❌ [{source_message.name}] Exception: {exception} at {source_message.timestamp}"
        )


class BotProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, IngressMessage):
            return BotMessage(
                content="Ok, ping!",
                iteration=1,
            )
        elif isinstance(message, UserMessage):
            iteration = message.iteration + 1
            return BotMessage(
                content="Ping back!",
                iteration=iteration,
            )
        else:
            raise ValueError(f"Unexpected message: {message}")


class UserProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, BotMessage):
            # After 2 iterations, return egress
            if message.iteration >= 2:
                return EgressMessage(
                    content="Final pong!",
                )
            else:
                return UserMessage(
                    content="Pong!",
                    iteration=message.iteration,
                )
        else:
            raise ValueError(f"Unexpected message: {message}")


async def main():
    # Setup message bus configuration
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
        observers=[LoggingObserver()],
    )

    # Create processors
    bot = BotProcessor(name="bot")
    user = UserProcessor(name="user")

    # Setup message bus with type-based routing
    bus = CompositeProcessor(
        handlers={
            IngressMessage: bot,
            BotMessage: user,
            UserMessage: bot,
        },
    )
    await bus.setup(config)

    # Create and process ingress message
    message = IngressMessage(
        content="Hello, let's play ping pong!",
        conversation_id="pingpong-demo",
    )

    await bus.process(message)


if __name__ == "__main__":
    asyncio.run(main())
