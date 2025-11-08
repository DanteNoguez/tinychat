import asyncio
from typing import Optional
from dataclasses import dataclass

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    ErrorMessage,
    Message,
)
from tinychat.asynchronous.manager import TaskManager, TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor
from tinychat.processors.message_bus import MessageBus
from tinychat.observers.observer import (
    BaseObserver,
    MessageReceived,
    MessageProcessed,
)


@dataclass
class BotMessage(Message):
    content: str
    iteration: int


@dataclass
class UserMessage(Message):
    content: str
    iteration: int


class LoggingObserver(BaseObserver):
    async def on_message_received(self, message: MessageReceived) -> None:
        print(f"📨 [{message.processor.name}] Received: {message.message.content}")

    async def on_message_processed(self, message: MessageProcessed) -> None:
        print(f"✅ [{message.processor.name}] Processed: {message.message.content}")

    async def on_error_message(
        self, source_message: Message, error_msg: ErrorMessage
    ) -> None:
        print(f"❌ Error: {error_msg.content}")


class BotProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, IngressMessage):
            print(f"🤖 Bot: Starting conversation with '{message.content}'")
            return BotMessage(
                content=f"Bot responds to: {message.content}",
                iteration=1,
            )
        elif isinstance(message, UserMessage):
            iteration = message.iteration + 1
            print(f"🤖 Bot: Ping! (iteration {iteration})")
            return BotMessage(
                content=f"Bot pong #{iteration}",
                iteration=iteration,
            )
        else:
            raise ValueError(f"Unexpected message: {message}")


class UserProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, BotMessage):
            # After 2 iterations, return egress
            if message.iteration >= 2:
                print(f"👤 User: Done after {message.iteration} iterations!")
                return EgressMessage(
                    content=f"Conversation complete after {message.iteration} rounds",
                )
            else:
                print(f"👤 User: Pong! (iteration {message.iteration})")
                return UserMessage(
                    content=f"User ping #{message.iteration}",
                    iteration=message.iteration,
                )
        else:
            raise ValueError(f"Unexpected message: {message}")


async def main():
    # Setup task manager
    loop = asyncio.get_running_loop()
    task_params = TaskManagerParams(loop=loop)
    task_manager = TaskManager()
    task_manager.setup(task_params)

    # Create observer
    observer = LoggingObserver()

    # Create processors
    bot = BotProcessor(name="bot")
    user = UserProcessor(name="user")

    # Setup message bus with type-based routing and observer
    bus = MessageBus(
        handlers={
            IngressMessage: bot,
            BotMessage: user,
            UserMessage: bot,
        },
        task_manager=task_manager,
        observers=[observer],
    )
    await bus.setup()

    # Create and process ingress message
    message = IngressMessage(
        content="Hello, let's play ping pong!",
        conversation_id="pingpong-demo",
    )

    print("=" * 60)
    print("🎾 Starting Ping-Pong Demo")
    print(f"Initial message: {message.content}")
    print("=" * 60)
    print()

    result = await bus.process(message)

    print()
    print("=" * 60)
    if isinstance(result, EgressMessage):
        print(f"📤 Final Result: {result.content}")
    else:
        print(f"Unexpected result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
