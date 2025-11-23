from abc import abstractmethod
from typing import Optional

from loguru import logger

from tinychat.processors.message_processor import MessageProcessor
from tinychat.messages.messages import Message, EgressMessage
from tinychat.services.llm.models import LLMConfig, LLMMessage
from tinychat.services.llm.tools import GenerateTypedMessageTool


class LLMService(MessageProcessor):
    """
    Base class for LLM services.

    Manages conversation history as a list of typed LLMMessage objects,
    handles tool execution, and defines the standard processing flow.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm_config = llm_config

        # State: Normalized history and instructions
        self.chat_history: list[LLMMessage] = []
        self.instructions: Optional[str] = llm_config.instructions

        # Tools Setup
        self.tools = llm_config.tools or []

        # Register Routing Tools from output_types
        self.routing_tools = {}
        if self.output_types:
            for msg_type in self.output_types:
                # Skip terminal/special types if necessary
                if msg_type in (type(None), EgressMessage):
                    continue

                # Create a tool definition for this message type
                msg_tool = GenerateTypedMessageTool(msg_type)
                self.tools.append(msg_tool)
                self.routing_tools[msg_tool.name] = msg_type

        self.tools_by_name = {tool.name: tool for tool in self.tools}

    # ==========================================================================
    # Public API & State Management
    # ==========================================================================

    @property
    def context(self) -> list[LLMMessage]:
        """Returns the current conversation history (already in standard format)."""
        return self.chat_history

    def add_message(self, message: LLMMessage) -> None:
        """Manually adds a message to the conversation history."""
        self.chat_history.append(message)

    def add_messages(self, messages: list[LLMMessage]) -> None:
        """Manually adds multiple messages to the history."""
        self.chat_history.extend(messages)

    def clear_history(self) -> None:
        """Clears the conversation history."""
        self.chat_history.clear()

    def set_instructions(self, instructions: str) -> None:
        """Sets the persistent instructions (system prompt) for the LLM."""
        self.instructions = instructions

    # ==========================================================================
    # MessageProcessor Implementation
    # ==========================================================================

    async def _process(self, message: Message) -> Message:
        """
        Standard LLM processing flow:
        1. Ingest: Convert incoming message to Message and append to history.
        2. Generate: Call implementation-specific generation (handles tools/recursion).
        3. Return: Return the final Message.
        """
        # 1. Ingest
        llm_message = self._ensure_llm_message(message)
        self.add_message(llm_message)

        # 2. Generate (implementation handles history updates for response/tools)
        response = await self._generate_completion()

        return response

    async def execute_tool(self, name: str, arguments: dict) -> str:
        """Executes a tool by name and returns the result as a string."""
        if tool := self.tools_by_name.get(name):
            try:
                logger.debug(f"Executing tool {name} with args {arguments}")
                result = await tool.run(**arguments)
                return str(result)
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return f"Error executing tool {name}: {str(e)}"

        return f"Error: Tool {name} not found."

    # ==========================================================================
    # Abstract Methods
    # ==========================================================================

    @abstractmethod
    async def _generate_completion(self) -> LLMMessage:
        """
        Generates a completion from the LLM.
        Must handle:
        - Converting self.chat_history to provider format
        - Calling API
        - Handling tool calls (recursion)
        - Appending results to self.chat_history
        """

    @abstractmethod
    def _ensure_llm_message(self, message: Message) -> LLMMessage:
        """Converts a generic Message to a provider-specific LLMMessage (User role)."""

    def _get_routing_message(self, name: str, arguments: dict) -> Optional[Message]:
        """
        Checks if the tool name corresponds to a routing message.
        If so, instantiates and returns the Message.
        """
        if message_type := self.routing_tools.get(name):
            try:
                return message_type(**arguments)
            except Exception as e:
                logger.warning(f"Failed to instantiate routing message {name}: {e}")
                # Fallback: return None so the caller treats it as a normal tool failure
                return None
        return None
