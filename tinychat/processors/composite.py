from typing import Optional, List
from enum import Enum
from loguru import logger
from tinychat.messages.messages import Message
from tinychat.observers.observer import BaseObserver, MessageRouted
from tinychat.asynchronous.manager import BaseTaskManager, TaskManager
from tinychat.processors.message_processor import MessageProcessor, ProcessorSetup
from tinychat.state.manager import StateManager
import time


class CompositeProcessor(MessageProcessor):
    """
    A processor that contains and orchestrates multiple sub-processors.

    Composite processors:
    - Contain a collection of processors registered by enum type
    - Provide type-safe routing via route(ProcessorType.X, message)
    - Share a task manager with all sub-processors
    - Notify observers of routing events

    Sub-processors route through the composite:
        result = await self.composite.route(ProcessorType.CRM, message)
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        processors: dict[Enum, MessageProcessor],
        max_hops: int = 30,
        task_manager: BaseTaskManager = TaskManager(),
        observers: Optional[List[BaseObserver]] = None,
    ):
        super().__init__(name=name, task_manager=task_manager, observers=observers)
        self._processors = processors
        self._max_hops = max_hops  # Maximum routing depth to prevent infinite loops
        self._observers = observers
        self._task_manager = task_manager
        self.state_manager = StateManager()

    @property
    def processors(self) -> dict[Enum, MessageProcessor]:
        return self._processors

    def get_processor(self, processor_type: Enum) -> MessageProcessor:
        if processor_type not in self._processors:
            raise KeyError(f"Processor type '{processor_type}' not found in {self}")
        return self._processors[processor_type]

    async def setup(self, setup: ProcessorSetup):
        """
        Initialize the composite and all sub-processors.

        This:
        1. Sets up the composite itself
        2. Sets up all sub-processors with shared task manager
        3. Injects composite reference so processors can route via self.composite
        """
        await super().setup(setup)

        sub_setup = ProcessorSetup(
            task_manager=self._task_manager,
            observers=self._observers,
        )

        for processor in self._processors.values():
            processor.composite = self
            await processor.setup(sub_setup)
            logger.debug(
                f"CompositeProcessor[{self.name}]: Injected composite into {processor.name}"
            )

    async def cleanup(self):
        for processor in self._processors.values():
            await processor.cleanup()

        await super().cleanup()

    async def _process(self, message: Message) -> Optional[Message]:
        """
        Subclasses should override this to implement routing logic.
        Alternatively, use route() directly without overriding _process.
        """
        raise NotImplementedError(
            f"{self}: CompositeProcessor requires either overriding _process() "
            "or using route() directly"
        )

    async def route(self, processor_type: Enum, message: Message) -> Message:
        """
        Route a message to a specific sub-processor by enum type.

        Respects state:
        - STOPPED: raises RuntimeError
        - PAUSED: waits until resumed
        - RUNNING/ERROR: proceeds normally

        Usage:
            await self.composite.route(ProcessorType.CRM, message)
        """
        self.state_manager.check_stopped()
        await self.state_manager.wait_if_paused()

        processor = self.get_processor(processor_type)
        logger.debug(f"{self}: routing message {message.id} to {processor_type}")
        result = await processor.process(message)

        self.task_manager.create_task(
            self._notify_routed(processor_type, message)
        )  # TODO: Add source name
        return result

    async def _notify_routed(
        self, source: Enum, target_name: Enum, message: Message
    ) -> None:
        if not self._observers:
            return

        data = MessageRouted(
            source=source,
            target=target_name,
            message=message,
            timestamp=time.monotonic_ns(),
        )

        for observer in self._observers:
            try:
                await observer.on_message_routed(data)
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_message_routed: {e}")
