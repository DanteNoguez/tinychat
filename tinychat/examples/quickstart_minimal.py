"""
tinychat Minimal Example
=========================

The simplest possible echo bot - demonstrates core architecture.
"""

import asyncio
from typing import Optional

from tinychat.conversations.conversation import Conversation
from tinychat.messages.messages import UserMessage, AIMessage, Message, LLMServiceType
from tinychat.processors.message_processor import MessageProcessor


class EchoProcessor(MessageProcessor):
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, UserMessage):
            return AIMessage(
                content=message.content,
                service=message.service,
                conversation_id=message.conversation_id,
                agent_id="echo",
            )
        return message


class EchoBot(Conversation):
    async def _process(self, message: Message) -> Optional[Message]:
        return await self.route_to("echo", message)


async def main():
    bot = EchoBot(conversation_id="demo", processors=[EchoProcessor(name="echo")])

    loop = asyncio.get_event_loop()
    await bot.setup(loop)

    message = UserMessage(
        content="Hello, tinychat!",
        service=LLMServiceType.OPENAI,
        conversation_id=bot.conversation_id,
    )

    print(f"User: {message.content}")
    response = await bot.process(message)
    print(f"Bot: {response.content}")

    await bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
