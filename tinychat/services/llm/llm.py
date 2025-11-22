from abc import abstractmethod

from tinychat.messages.messages import Message
from tinychat.processors.message_processor import MessageProcessor
from tinychat.services.llm.models import LLMConfig


class LLMService(MessageProcessor):
    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm_config = llm_config
        self._chat_history: list[dict] = []

    def _generate_prompt(self) -> list[dict]:
        return [
            {
                "role": self._llm_config.prompt.role,
                "content": self._llm_config.prompt.content,
            }
        ]

    @abstractmethod
    async def generate_response(self, message: Message) -> Message: ...
