import asyncio
from loguru import logger

from typing import Optional, TypedDict, Literal

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    Message,
)
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.services.llm.models import (
    OpenAILLMConfig,
    OpenAISystemMessage,
    OpenAIAssistantMessage,
)
from tinychat.services.llm.tools import Tool
from tinychat.services.llm.openai_llm import OpenAILLM


# In tinychat, tools are defined as classes that inherit from the Tool class.
# The run method is called by the agent to execute tool calls.
# The tool subclass can be of any arbitrary complexity.
# Type annotations and the run metod's docstring are used to generate the tool's definition for the LLM.
# Object arguments in the run method must be either a Pydantic model or a TypedDict.
class CalculatorTool(Tool):
    class ExtraData(TypedDict):
        names: list[str]
        values: list[int]

    def __init__(self, *, name: str, precision: int = 2):
        super().__init__(name=name)
        self.precision = precision

    def _perform_calculation(self, operation: str, a: float, b: float) -> float | str:
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
        }

        if operation not in operations:
            return "Error: Invalid operation. Valid operations are: add, subtract, multiply, divide."

        return operations[operation](a, b)

    def _format_result(self, result: float | str) -> str:
        if isinstance(result, str):
            return result
        return f"{result:.{self.precision}f}"

    async def run(
        self,
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: float,
        b: float,
        extra_data: Optional[ExtraData] = None,
    ) -> str:
        """
        Perform basic arithmetic operations.

        :param operation: The operation to perform.
        :param a: The first operand.
        :param b: The second operand.
        :param extra_data: Extra data to be used in the calculation.
        """
        result = self._perform_calculation(operation, a, b)
        formatted = self._format_result(result)
        return f"Result: {formatted}"


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
    llm = OpenAILLM(
        llm_config=OpenAILLMConfig(
            prompt=OpenAISystemMessage(
                content="You're a helpful math assistant. Use the calculator tool to perform calculations.",
            ),
            model_name="gpt-4.1",
            temperature=0.0,
            max_tokens=300,
            tools=[calculator],
        )
    )

    egress = EgressMessageProcessor(output_types={EgressMessage})

    # Setup composite processor with simple routing
    chatbot = CompositeProcessor(
        handlers={
            IngressMessage: llm,
            OpenAIAssistantMessage: egress,
        },
    )
    await chatbot.setup(config)

    # Create and process a math question
    message = IngressMessage(
        content="What is 42 multiplied by 13? Then add 100 to that result.",
        conversation_id="calculator-demo",
    )

    logger.debug(f"{llm} - Tools schema: {llm.tools_schema}")

    result = await chatbot.process(message)

    logger.debug(f"{llm} - Chat history: {llm.chat_history}")

    logger.success(f"Final Result: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
