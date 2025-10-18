"""
tinychat Minimal Example
=========================

The simplest possible tinychat example - just the essentials.
"""

import asyncio
from typing import Optional

from tinychat.conversations.conversation import Conversation
from tinychat.messages.messages import UserMessage, AIMessage, Message
from tinychat.messages.models import LLMServiceType
from tinychat.processors.message_processor import MessageProcessor


class EchoProcessor(MessageProcessor):
    """Simple processor that echoes user messages back."""
    
    def __init__(self):
        super().__init__(name="echo")
    
    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, UserMessage):
            response = AIMessage(
                content=message.content,
                service=LLMServiceType.OPENAI,
                conversation_id=message.conversation_id,
                agent_id="echo-bot"
            )
            print(f"Bot: {response.content}")
            return response
        return message


async def main():
    # Create a conversation with a single processor
    conversation = Conversation(
        conversation_id="minimal-demo",
        processors=[EchoProcessor()]
    )
    
    # Initialize
    loop = asyncio.get_event_loop()
    await conversation.setup(loop)
    
    # Set up routing: when UserMessage arrives, send to echo processor
    @conversation.router.on(UserMessage)
    async def handle_user(msg: Message, conv: Conversation):
        return await conv.route_to("echo", msg)
    
    # Send a message
    message = UserMessage(
        content="Hello, tinychat!",
        service=LLMServiceType.OPENAI,
        conversation_id=conversation.conversation_id
    )
    
    print(f"User: {message.content}")
    await conversation.router.emit(message)
    
    # Cleanup
    await conversation.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

