import asyncio

from tinychat.agents.models import OpenAIAgentConfig
from tinychat.agents.registry import AgentConfigRegistry
from tinychat.manager.in_memory import InMemoryConversationsManager
from tinychat.manager.redis_manager import RedisConversationsManager

DEFAULT_AGENT_CONFIG = OpenAIAgentConfig(
    prompt="You're a knowledgeable AI that teaches in the style of Richard Feynman.",
    max_tokens=100,
    temperature=1.0,
)

agent_registry = AgentConfigRegistry(
    configs={"default": DEFAULT_AGENT_CONFIG}, default_agent_type="default"
)

CONVERSATIONS_MANAGER = RedisConversationsManager(
    agent_registry=agent_registry,
)


async def main():
    print("Welcome to tinychat!")
    conversations = {1: "Leibniz", 2: "Turing", 3: "Schmidhuber"}
    current_conversation = 1

    while True:
        try:
            current_user = conversations[current_conversation]
            user_input = input(f"{current_user}: ")
            print(f"Current conv: {current_conversation}")
            response = await CONVERSATIONS_MANAGER.handle_incoming_message(
                identifier=current_conversation,
                request_type="default",
                message=user_input,
            )
            print(f"AI to {current_user}: {response}")

            # Move to the next conversation
            current_conversation = (current_conversation % len(conversations)) + 1
        except KeyboardInterrupt:
            print("\nExiting all conversations. Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
