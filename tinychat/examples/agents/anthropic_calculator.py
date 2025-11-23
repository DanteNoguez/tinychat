import asyncio
from loguru import logger

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    Message,
)
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.services.llm.models import (
    AnthropicLLMConfig,
    AnthropicAssistantMessage,
)
from tinychat.services.llm.anthropic_llm import AnthropicLLM

from tinychat.examples.agents.calculator_tool import CalculatorTool


class EgressMessageProcessor(MessageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _process(self, message: Message) -> EgressMessage:
        return EgressMessage(content=message.content)


async def main():
    # Setup configuration
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
    )

    # Create calculator tool
    calculator = CalculatorTool(
        name="calculator",
        precision=2,
    )

    # Create LLM processor with output type declaration
    llm = AnthropicLLM(
        llm_config=AnthropicLLMConfig(
            instructions="You're a helpful math assistant. Use the calculator tool to perform calculations.",
            tools=[calculator],
        )
    )

    egress = EgressMessageProcessor(output_types={EgressMessage})

    # Setup composite processor with simple routing
    # Route Ingress -> LLM -> (Tool Handling Internal) -> AnthropicAssistantMessage -> Egress
    chatbot = CompositeProcessor(
        handlers={
            IngressMessage: llm,
            AnthropicAssistantMessage: egress,
        },
    )
    await chatbot.setup(config)

    # Create and process a math question
    message = IngressMessage(
        content="What is 42 multiplied by 13? Then add 100 to that result.",
    )

    logger.debug(f"{llm} - Tools schema: {llm.tools_schema}")

    result = await chatbot.process(message)

    # In Anthropic, the "system" instruction is used at the request level,
    # so it is not included in the chat history.
    chat_history = [m.to_anthropic_format() for m in llm.chat_history]
    logger.debug(f"{llm} - Chat history: {chat_history}")

    logger.success(f"Final Result: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
