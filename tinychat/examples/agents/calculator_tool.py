from typing import Optional, TypedDict, Literal

from tinychat.services.llm.tools import Tool


# In tinychat, tools are defined as classes that inherit from the Tool class.
# The run method is called by the agent to execute tool calls.
# The tool subclass can be of any arbitrary complexity.
# Type annotations and the run method's docstring are used to generate the tool's definition for the LLM.
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
