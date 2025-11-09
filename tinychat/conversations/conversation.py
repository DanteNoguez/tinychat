from abc import abstractmethod
from typing import Optional

from loguru import logger

from tinychat.messages.messages import Message
from tinychat.processors.message_bus import MessageBus
from tinychat.state.manager import StateManager, ProcessingState
from tinychat.processors.message_processor import MessageProcessor


class Conversation(MessageBus):
    def __init__(
        self,
        *,
        conversation_id: str,
        handlers: dict[type[Message], MessageProcessor],
        state_manager: Optional[StateManager] = None,
        max_depth: int = 30,
    ):
        super().__init__(
            handlers=handlers,
            max_depth=max_depth,
        )
        self._conversation_id = conversation_id
        self._state_manager = state_manager or StateManager()

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def conversation_state(self) -> ProcessingState:
        return self._state_manager.state

    def __str__(self) -> str:
        return f"Conversation({self._conversation_id})"

    async def _process(self, message: Message) -> Optional[Message]:
        match self.conversation_state:
            case ProcessingState.STOPPED:
                logger.warning(f"Conversation {self} is stopped.")
                await self.wait_if_stopped()
            case ProcessingState.ERROR:
                logger.warning(f"Conversation {self} is in error state.")
                await self.handle_errored_conversation()
                return None
            case ProcessingState.COMPLETED:
                logger.info(f"Conversation {self} is completed.")
                return None

        try:
            return await super()._process(message)
        except Exception:
            self._state_manager.set_error()
            raise

    async def complete_conversation(self):
        await self.interrupt_processing()
        self._state_manager.complete()

    async def resume_conversation(self):
        self._state_manager.resume()

    async def stop_conversation(self):
        await self.interrupt_processing()
        self._state_manager.stop()

    async def wait_if_stopped(self):
        await self._state_manager.wait_if_stopped()

    @abstractmethod
    async def handle_errored_conversation(self): ...
