from abc import abstractmethod

from tinychat.processors.message_processor import MessageProcessor
from tinychat.services.llm.models import LLMConfig, LLMMessage


class LLMService(MessageProcessor):
    """
    Abstract base class for LLM services.

    This defines the contract that all LLM implementations (OpenAI, Anthropic, etc.)
    must follow. It ensures consistent interaction patterns regardless of the
    underlying API's state management or message formats.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm_config = llm_config

    @property
    @abstractmethod
    def context(self) -> list[LLMMessage]:
        """
        Returns the current conversation history as a list of standard LLMMessage objects.

        This provides a unified view of the state for observers and evaluators,
        hiding the internal representation (e.g., dicts for OpenAI, objects for Anthropic).
        """
        ...

    @abstractmethod
    def add_message(self, message: LLMMessage) -> None:
        """
        Manually adds a message to the conversation history.
        """
        ...

    @abstractmethod
    def clear_history(self) -> None:
        """
        Clears the conversation history, preserving system prompts if applicable.
        """
        ...
