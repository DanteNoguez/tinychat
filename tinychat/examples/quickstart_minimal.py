import asyncio

from typing import Optional
from loguru import logger

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    Message,
)
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor


class EchoMessage(Message): ...


class EchoProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, IngressMessage):
            logger.debug(f"{self} received: {message}")
            return EchoMessage(
                content=message.content,
            )


class TransformerProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, EchoMessage):
            logger.debug(f"{self} received: {message}")
            return EgressMessage(
                content="Transformed!",
            )


async def main():
    # Get the running event loop and setup task manager
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
    )
    # Create processors
    echo = EchoProcessor(name="Echo", output_types={EchoMessage})
    transformer = TransformerProcessor(name="Transformer", output_types={EgressMessage})

    # Setup composite processor with type-based routing
    composite = CompositeProcessor(
        handlers={
            IngressMessage: echo,
            EchoMessage: transformer,
        },
    )
    await composite.setup(config)

    # Create and process ingress message
    message = IngressMessage(
        content="Hello, tinychat!",
    )

    result = await composite.process(message)
    logger.success(f"Final Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
