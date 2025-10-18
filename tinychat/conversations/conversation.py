import asyncio
import time
from typing import List, Optional

from loguru import logger

from tinychat.messages.messages import Message
from tinychat.observers.observer import BaseObserver, MessageRouted
from tinychat.processors.message_processor import CompositeProcessor, ProcessorSetup
from tinychat.asynchronous.manager import (
    BaseTaskManager,
    TaskManager,
    TaskManagerParams,
)


class Conversation(CompositeProcessor):
    """
    A specialized CompositeProcessor for managing conversations.

    Conversation extends CompositeProcessor to add:
    - Conversation-specific state (phase, current processor)
    - Shared agent state for coordination
    - State transition callbacks and notifications

    Like all CompositeProcessors, sub-processors can call each other directly:
        result = await self.history_processor.process(message)

    Or use the route_to convenience method:
        result = await conversation.route_to("history_processor", message)
    """

    def __init__(
        self,
        conversation_id: str,
        processors: Optional[List] = None,
        *,
        task_manager: Optional[BaseTaskManager] = None,
        observers: Optional[List[BaseObserver]] = None,
    ):
        super().__init__(name=conversation_id, processors=processors)
        self._conversation_id = conversation_id
        self._task_manager = task_manager or TaskManager()

        # Override observers from parent
        if observers:
            self._observers = observers

        self._setup_complete = False

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def __str__(self) -> str:
        return f"Conversation({self._conversation_id})"

    def add_observer(self, observer: BaseObserver):
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: BaseObserver):
        if observer in self._observers:
            self._observers.remove(observer)

    async def setup(self, loop: asyncio.AbstractEventLoop):
        """
        Initialize the conversation and all sub-processors.

        This sets up the conversation as a root composite processor and
        initializes all sub-processors with peer injection.
        """
        if self._setup_complete:
            return

        params = TaskManagerParams(loop=loop)
        self._task_manager.setup(params)

        setup = ProcessorSetup(
            task_manager=self._task_manager, observers=self._observers
        )
        await super().setup(setup)

        self._setup_complete = True

    async def cleanup(self):
        await super().cleanup()
        self._setup_complete = False

    async def _process(self, message: Message) -> Optional[Message]:
        raise NotImplementedError(
            f"{self}: Conversation requires overriding _process() to define routing logic"
        )

    async def route_to(
        self, processor_name: str, message: Message
    ) -> Optional[Message]:
        logger.trace(
            f"{self}: routing message {message.id} ({type(message).__name__}) to {processor_name}"
        )

        # Use parent's route_to which tracks composite state
        return await super().route_to(processor_name, message)

    async def notify_message_routed(self, source, target, message: Message):
        if not self._observers:
            return

        data = MessageRouted(
            source=source, target=target, message=message, timestamp=time.time_ns()
        )

        for observer in self._observers:
            try:
                await observer.on_message_routed(data)
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_message_routed: {e}")
