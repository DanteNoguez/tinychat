import asyncio
from typing import Optional

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    Message,
)
from tinychat.asynchronous.manager import TaskManager, TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor
from tinychat.processors.message_bus import MessageBus


class EchoMessage(Message):
    pass


class EchoProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, IngressMessage):
            print(f"📥 Echo: Received '{message.content}'")
            return EchoMessage(
                content=message.content,
            )
        else:
            raise ValueError(f"Unexpected message: {message}")


class TransformerProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, EchoMessage):
            print(f"🔄 Transformer: Received '{message.content}'")
            return EgressMessage(
                content="Transformed!",
            )
        else:
            raise ValueError(f"Unexpected message: {message}")


async def main():
    # Get the running event loop and setup task manager
    loop = asyncio.get_running_loop()
    task_params = TaskManagerParams(loop=loop)
    task_manager = TaskManager()
    task_manager.setup(task_params)

    # Create processors
    echo = EchoProcessor(name="echo")
    transformer = TransformerProcessor(name="transformer")

    # Setup message bus with type-based routing
    bus = MessageBus(
        handlers={
            IngressMessage: echo,
            EchoMessage: transformer,
        },
        task_manager=task_manager,
    )
    await bus.setup()

    # Create and process ingress message
    message = IngressMessage(
        content="Hello, tinychat!",
        conversation_id="demo",
    )

    print(f"Ingress: {message.content}")
    result = await bus.process(message)

    if isinstance(result, EgressMessage):
        print(f"📤 Bot: {result.content}")
    else:
        print(f"Unexpected result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
