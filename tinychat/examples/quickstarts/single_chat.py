import asyncio

from tinychat.agents import OpenAIAgent
from tinychat.agents.models import OpenAIAgentConfig
from tinychat.agents.registry import AgentConfigRegistry
from tinychat.conversations_manager import InMemoryConversationsManager

DEFAULT_AGENT_CONFIG = OpenAIAgentConfig(
    prompt="You're a knowledgeable AI that teaches in the style of Richard Feynman.",
    max_tokens=100,
    temperature=1.0,
)

agent_registry = AgentConfigRegistry(
    configs={"default": DEFAULT_AGENT_CONFIG}, default_agent_type="default"
)

CONVERSATIONS_MANAGER = InMemoryConversationsManager(
    agent_registry=agent_registry,
)


async def main():
    print("Welcome to tinychat!")
    while True:
        try:
            user_input = input("You: ")
            response = await CONVERSATIONS_MANAGER.handle_incoming_message(
                identifier=1, agent_type="default", message=user_input
            )
            print(f"AI: {response}")
        except KeyboardInterrupt:
            print("\nExiting the conversation. Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
