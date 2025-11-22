import asyncio
from loguru import logger

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
)
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.observers.observer import (
    BaseObserver,
    MessageReceived,
    MessageProcessed,
    Message,
)

from tinychat.services.llm.models import (
    OpenAILLMConfig,
    OpenAISystemMessage,
    LLMMessage,
)
from tinychat.services.llm.tools import Tool
from tinychat.services.llm.openai_llm import OpenAILLM
from tinychat.utils.logging import configure_pretty_logging


# Trace level logging allows us to see LLM behavior in the console.
configure_pretty_logging(debug_level=5)


# In tinychat, tools are defined as classes that inherit from the Tool class.
# The run method is called by the agent to execute tool calls.
# The tool subclass can be of any arbitrary complexity.
class CalculatorTool(Tool):
    def __init__(self, *, name: str, precision: int = 2):
        super().__init__(name=name)
        self.precision = precision

    def _validate_operation(self, operation: str) -> bool:
        return operation in ["add", "subtract", "multiply", "divide"]

    def _perform_calculation(self, operation: str, a: float, b: float) -> float | str:
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
        }
        return operations[operation](a, b)

    def _format_result(self, result: float | str) -> str:
        if isinstance(result, str):
            return result
        return f"{result:.{self.precision}f}"

    async def run(self, operation: str, a: float, b: float) -> str:
        """
        Perform basic arithmetic operations.

        :param operation: The operation to perform. Must be one of: add, subtract, multiply, divide.
        :param a: The first number.
        :param b: The second number.
        """
        if not self._validate_operation(operation):
            return "Error: Invalid operation. Valid operations are: add, subtract, multiply, divide."

        result = self._perform_calculation(operation, a, b)
        formatted = self._format_result(result)
        return f"Result: {formatted}"


class LoggingObserver(BaseObserver):
    async def on_message_received(self, message: MessageReceived) -> None:
        logger.debug(
            f"📨 [{message.source_processor.name}] Received: {message.content}"
        )

    async def on_message_processed(self, message: MessageProcessed) -> None:
        logger.debug(
            f"✅ [{message.source_processor.name}] Returned: {message.content}"
        )

    async def on_exception(self, source_message: Message, exception: Exception) -> None:
        logger.error(
            f"❌ [{source_message.name}] Exception: {exception} at {source_message.timestamp}"
        )


class EgressMessageProcessor(MessageProcessor):
    def __init__(self, *, name: str):
        super().__init__(name=name)

    async def _process(self, message: Message) -> Message:
        return EgressMessage(content=message.content)


async def main():
    # Setup configuration
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
        observers=[LoggingObserver()],
    )

    # Create LLM processor with output type declaration
    llm = OpenAILLM(
        llm_config=OpenAILLMConfig(
            prompt=OpenAISystemMessage(
                content="You're a helpful math assistant. Use the calculator tool to perform calculations.",
            ),
            model_name="gpt-4.1",
            temperature=0.0,
            max_tokens=300,
            tools=[
                CalculatorTool(
                    name="calculate",
                    precision=2,
                )
            ],
        )
    )

    # Setup composite processor with simple routing
    chatbot = CompositeProcessor(
        handlers={
            IngressMessage: llm,
            LLMMessage: EgressMessageProcessor(name="egress_message_processor"),
        },
    )
    await chatbot.setup(config)

    # Create and process a math question
    message = IngressMessage(
        content="What is 42 multiplied by 13? Then add 100 to that result.",
        conversation_id="calculator-demo",
    )

    result = await chatbot.process(message)

    logger.success(f"Final Result: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
