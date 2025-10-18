import asyncio
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from typing import Coroutine, List, Optional

from loguru import logger

from tinychat.messages.messages import ErrorMessage, Message
from tinychat.observers.observer import (
    BaseObserver,
    MessageReceived,
    MessageProcessed,
    MessageRouted,
)
from tinychat.asynchronous.manager import (
    BaseTaskManager,
    TaskManager,
    TaskManagerParams,
)
from tinychat.processors.exceptions import MaxHopsExceededError


@dataclass
class ProcessorSetup:
    task_manager: BaseTaskManager
    observers: Optional[List[BaseObserver]] = None


class MessageProcessor:
    """
    Base class for standalone message processors.

    A processor is a unit that:
    - Processes messages via the process() method
    - Can create and manage async tasks
    - Notifies observers of processing events
    - Handles errors gracefully

    Processors are standalone and don't know about other processors.
    For inter-processor communication, use CompositeProcessor.
    """

    def __init__(self, *, name: Optional[str] = None):
        self._id = uuid.uuid4()
        self._name = name or f"{self.__class__.__name__}_{self._id.hex[:8]}"
        self._task_manager: Optional[BaseTaskManager] = None
        self._observers: List[BaseObserver] = []
        self._started = False
        self._owns_task_manager = False

    @property
    def id(self) -> uuid.UUID:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def task_manager(self) -> BaseTaskManager:
        if not self._task_manager:
            raise Exception(f"{self} task manager is not initialized")
        return self._task_manager

    @property
    def observers(self) -> List[BaseObserver]:
        return self._observers

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name})"

    async def setup(self, setup: ProcessorSetup):
        """
        Initialize the processor with task manager and observers.

        If no task manager is provided, creates its own.
        """
        if setup.task_manager:
            self._task_manager = setup.task_manager
            self._owns_task_manager = False
        else:
            self._task_manager = TaskManager()
            loop = asyncio.get_event_loop()
            self._task_manager.setup(TaskManagerParams(loop=loop))
            self._owns_task_manager = True

        if setup.observers:
            self._observers.extend(setup.observers)

        self._started = True

    async def cleanup(self):
        if self._owns_task_manager and self._task_manager:
            # TODO: Garbage collection
            pass
        self._started = False

    def create_task(
        self, coroutine: Coroutine, name: Optional[str] = None
    ) -> asyncio.Task:
        if name:
            name = f"{self}::{name}"
        else:
            name = f"{self}::{coroutine.cr_code.co_name}"
        return self.task_manager.create_task(coroutine, name)

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = 1.0):
        await self.task_manager.cancel_task(task, timeout)

    async def process(self, message: Message) -> Optional[Message]:
        """
        Process a message and return the result.

        This is the main entry point for message processing. It handles:
        - Observer notifications
        - Error handling
        - Routing from and to other processors
        """

        await self._notify_received(message)

        try:
            result = await self._process(message)

            await self._notify_processed(message, result)
            return result

        except Exception as e:
            error_msg = await self.handle_error(e, message)
            await self._notify_processed(message, error_msg)
            raise

    @abstractmethod
    async def _process(self, message: Message) -> Optional[Message]:
        """
        Process a message and return the result.

        Subclasses must implement this to define their message processing logic.
        This method receives the message and should return:
        - A new/modified message
        - The same message passed through
        - None if no result
        """
        pass

    async def handle_error(self, error: Exception, message: Message) -> ErrorMessage:
        logger.error(f"{self}: error processing message {message.id}: {error}")

        return ErrorMessage(
            content=str(error),
            source=self.name,
            fatal=False,
        )

    async def _notify_received(self, message: Message) -> None:
        if not self._observers:
            return

        data = MessageReceived(processor=self, message=message, timestamp=time.monotonic_ns())

        for observer in self._observers:
            try:
                await observer.on_message_received(data)
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_message_received: {e}")

    async def _notify_processed(
        self,
        message: Message,
        result: Optional[Message],
    ) -> None:
        if not self._observers:
            return

        data = MessageProcessed(
            processor=self,
            message=message,
            result=result,
            timestamp=time.monotonic_ns(),
        )

        for observer in self._observers:
            try:
                await observer.on_message_processed(data)
            except Exception as e:
                logger.exception(
                    f"Observer {observer} failed on_message_processed: {e}"
                )


class CompositeProcessor(MessageProcessor):
    """
    A processor that contains and orchestrates multiple sub-processors.

    Composite processors:
    - Contain a collection of processors as peers
    - Inject peer references so processors can call each other directly
    - Share a task manager with all sub-processors
    - Notify observers of routing events

    Sub-processors can call each other via injected references:
        result = await self.other_processor.process(message)
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        processors: Optional[List[MessageProcessor]] = None,
        max_hops: int = 30,
    ):
        super().__init__(name=name)
        self._processors: dict[str, MessageProcessor] = {}
        self._max_hops = max_hops  # Maximum routing depth to prevent infinite loops

        if processors:
            for proc in processors:
                self._processors[proc.name] = proc

    @property
    def processors(self) -> dict[str, MessageProcessor]:
        return self._processors

    def add_processor(self, processor: MessageProcessor):
        self._processors[processor.name] = processor

    def get_processor(self, name: str) -> MessageProcessor:
        if name not in self._processors:
            raise KeyError(f"Processor '{name}' not found in {self}")
        return self._processors[name]

    async def setup(self, setup: ProcessorSetup):
        """
        Initialize the composite and all sub-processors.

        This:
        1. Sets up the composite itself
        2. Sets up all sub-processors with shared task manager
        3. Injects peer references so processors can call each other
        """
        await super().setup(setup)

        sub_setup = ProcessorSetup(
            task_manager=self._task_manager,
            observers=self._observers,
        )

        for processor in self._processors.values():
            await processor.setup(sub_setup)

        await self._inject_peer_references()

    async def _inject_peer_references(self):
        """
        Inject references to all processors into each processor.

        After this, each processor can call any other via:
            await self.other_processor.process(message)
        """
        for proc_name, processor in self._processors.items():
            for peer_name, peer in self._processors.items():
                if peer_name != proc_name:
                    setattr(processor, peer_name, peer)

                    logger.debug(
                        f"CompositeProcessor[{self.name}]: Injected {peer_name} into {proc_name} as '{peer_name}'"
                    )

    async def cleanup(self):
        for processor in self._processors.values():
            await processor.cleanup()

        await super().cleanup()

    async def _process(self, message: Message) -> Optional[Message]:
        """
        Subclasses should override this to implement routing logic.
        Alternatively, use route_to() directly without overriding _process.
        """
        raise NotImplementedError(
            f"{self}: CompositeProcessor requires either overriding _process() "
            "or using route_to() directly"
        )

    async def route_to(
        self, processor_name: str, message: Message
    ) -> Optional[Message]:
        """
        Route a message to a specific sub-processor by name.

        This is a convenience method for explicit routing without
        needing direct references.
        """
        processor = self.get_processor(processor_name)

        await self._notify_routed(self, processor, message)

        logger.trace(f"{self}: routing message {message.id} to {processor_name}")

        result = await processor.process(message)
        return result

    async def _notify_routed(
        self, source: MessageProcessor, target: MessageProcessor, message: Message
    ):
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
