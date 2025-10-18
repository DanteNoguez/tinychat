import asyncio
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from typing import Coroutine, List, Optional

from loguru import logger

from tinychat.messages.messages import ErrorMessage, Message
from tinychat.observers.observer import BaseObserver, MessageReceived, MessageProcessed
from tinychat.asynchronous.manager import (
    BaseTaskManager,
    TaskManager,
    TaskManagerParams,
)
from tinychat.processors.exceptions import MaxHopsExceededError


@dataclass
class ProcessorSetup:
    """Configuration passed to processors during setup."""

    task_manager: BaseTaskManager
    observers: Optional[List[BaseObserver]] = None


class ProcessorState:
    """
    Tracks message processing stages for a processor.

    Records operational metrics like:
    - Messages received/processed counts
    - Processing times
    - Error counts
    - Current status
    """

    def __init__(self, processor_name: str):
        self._processor_name = processor_name
        self._messages_received = 0
        self._messages_processed = 0
        self._errors = 0
        self._total_processing_time_ns = 0
        self._status = "idle"

    @property
    def processor_name(self) -> str:
        return self._processor_name

    @property
    def messages_received(self) -> int:
        return self._messages_received

    @property
    def messages_processed(self) -> int:
        return self._messages_processed

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def avg_processing_time_ms(self) -> float:
        if self._messages_processed == 0:
            return 0.0
        return (self._total_processing_time_ns / self._messages_processed) / 1_000_000

    @property
    def status(self) -> str:
        return self._status

    def record_received(self):
        self._messages_received += 1

    def record_processed(self, duration_ns: int):
        self._messages_processed += 1
        self._total_processing_time_ns += duration_ns

    def record_error(self):
        self._errors += 1

    def set_status(self, status: str):
        self._status = status


class MessageProcessor:
    """
    Base class for standalone message processors.

    A processor is a unit that:
    - Processes messages via the process() method
    - Manages its own state (processing metrics)
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
        self._state = ProcessorState(self._name)
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
    def state(self) -> ProcessorState:
        """Processor-level state tracking."""
        return self._state

    @property
    def task_manager(self) -> BaseTaskManager:
        if not self._task_manager:
            raise Exception(f"{self} task manager is not initialized")
        return self._task_manager

    @property
    def observers(self) -> List[BaseObserver]:
        """Observers for this processor."""
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
            # Create own task manager for standalone operation
            self._task_manager = TaskManager()
            loop = asyncio.get_event_loop()
            self._task_manager.setup(TaskManagerParams(loop=loop))
            self._owns_task_manager = True

        if setup.observers:
            self._observers.extend(setup.observers)

        self._started = True

    async def cleanup(self):
        """Clean up processor resources."""
        if self._owns_task_manager and self._task_manager:
            # Cleanup would happen here if TaskManager had cleanup
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
        - State tracking
        - Observer notifications
        - Error handling
        - Timing metrics
        """
        start_time = time.time_ns()

        self._state.record_received()
        self._state.set_status("processing")

        await self._notify_received(message, start_time)

        try:
            result = await self._process(message)
            end_time = time.time_ns()
            duration_ns = end_time - start_time

            self._state.record_processed(duration_ns)
            self._state.set_status("idle")

            await self._notify_processed(message, result, start_time, end_time)
            return result

        except Exception as e:
            end_time = time.time_ns()
            duration_ns = end_time - start_time

            self._state.record_error()
            self._state.set_status("error")

            error_msg = await self.handle_error(e, message)
            await self._notify_processed(message, error_msg, start_time, end_time)
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
        """
        Handle an error that occurred during message processing.

        Override this to customize error handling behavior.
        """
        logger.error(f"{self}: error processing message {message.id}: {error}")

        # Create error message with minimal info (no conversation_id)
        return ErrorMessage(
            content=str(error),
            source=self.name,
            fatal=False,
        )

    async def _notify_received(self, message: Message, timestamp: int):
        """Notify observers that this processor received a message."""
        if not self._observers:
            return

        data = MessageReceived(processor=self, message=message, timestamp=timestamp)

        for observer in self._observers:
            try:
                await observer.on_message_received(data)
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_message_received: {e}")

    async def _notify_processed(
        self,
        message: Message,
        result: Optional[Message],
        start_time: int,
        end_time: int,
    ):
        """Notify observers that this processor completed processing a message."""
        if not self._observers:
            return

        data = MessageProcessed(
            processor=self,
            message=message,
            result=result,
            timestamp=end_time,
            duration_ns=end_time - start_time,
        )

        for observer in self._observers:
            try:
                await observer.on_message_processed(data)
            except Exception as e:
                logger.exception(
                    f"Observer {observer} failed on_message_processed: {e}"
                )


class CompositeProcessorState:
    """
    Tracks orchestration state for a composite processor.

    Records higher-level metrics like:
    - Total messages routed
    - Routing decisions
    - Inter-processor calls
    - Current active processors
    """

    def __init__(self, composite_name: str):
        self._composite_name = composite_name
        self._messages_routed = 0
        self._routing_history: List[tuple[str, str]] = []  # (from, to) pairs
        self._active_processors: List[str] = []

    @property
    def composite_name(self) -> str:
        return self._composite_name

    @property
    def messages_routed(self) -> int:
        return self._messages_routed

    @property
    def routing_history(self) -> List[tuple[str, str]]:
        return self._routing_history.copy()

    @property
    def active_processors(self) -> List[str]:
        return self._active_processors.copy()

    def record_route(self, from_proc: str, to_proc: str):
        self._messages_routed += 1
        self._routing_history.append((from_proc, to_proc))

        # Keep history bounded
        if len(self._routing_history) > 1000:
            self._routing_history = self._routing_history[-500:]

    def set_active(self, processor_name: str):
        if processor_name not in self._active_processors:
            self._active_processors.append(processor_name)

    def set_inactive(self, processor_name: str):
        if processor_name in self._active_processors:
            self._active_processors.remove(processor_name)


class CompositeProcessor(MessageProcessor):
    """
    A processor that contains and orchestrates multiple sub-processors.

    Composite processors:
    - Contain a collection of processors as peers
    - Inject peer references so processors can call each other directly
    - Track orchestration-level state (routing, active processors)
    - Share a task manager with all sub-processors
    - Support both processor-level and composite-level observers

    Sub-processors can call each other via injected references:
        result = await self.other_processor.process(message)

    Example:
        class MyComposite(CompositeProcessor):
            async def _process(self, message: Message):
                # Route based on message type
                if isinstance(message, UserMessage):
                    return await self.history.process(message)
                return None
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        processors: Optional[List[MessageProcessor]] = None,
        max_hops: int = 20,
    ):
        super().__init__(name=name)
        self._processors: dict[str, MessageProcessor] = {}
        self._composite_state = CompositeProcessorState(self._name)
        self._max_hops = max_hops  # Maximum routing depth to prevent infinite loops

        if processors:
            for proc in processors:
                self._processors[proc.name] = proc

    @property
    def processors(self) -> dict[str, MessageProcessor]:
        """All sub-processors in this composite."""
        return self._processors

    @property
    def composite_state(self) -> CompositeProcessorState:
        """Composite-level orchestration state."""
        return self._composite_state

    def add_processor(self, processor: MessageProcessor):
        """Add a processor to this composite."""
        self._processors[processor.name] = processor

    def get_processor(self, name: str) -> MessageProcessor:
        """Get a processor by name."""
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
        # Setup self
        await super().setup(setup)

        # Setup all sub-processors with shared task manager
        sub_setup = ProcessorSetup(
            task_manager=self._task_manager,
            observers=self._observers,  # Share observers with subs
        )

        for processor in self._processors.values():
            await processor.setup(sub_setup)

        # Inject peer references - fully connected graph
        await self._inject_peer_references()

    async def _inject_peer_references(self):
        """
        Inject references to all processors into each processor.

        After this, each processor can call any other via:
            await self.other_processor.process(message)
        """
        for proc_name, processor in self._processors.items():
            # Inject references to all OTHER processors
            for peer_name, peer in self._processors.items():
                if peer_name != proc_name:
                    setattr(processor, peer_name, peer)

                    logger.debug(
                        f"CompositeProcessor[{self.name}]: Injected {peer_name} into {proc_name} as '{peer_name}'"
                    )

    async def cleanup(self):
        """Clean up the composite and all sub-processors."""
        # Cleanup all sub-processors
        for processor in self._processors.values():
            await processor.cleanup()

        # Cleanup self
        await super().cleanup()

    async def _process(self, message: Message) -> Optional[Message]:
        """
        Default implementation raises NotImplementedError.

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

        Includes cycle detection and max-hops enforcement.
        """
        # Max hops enforcement: prevent infinite routing chains
        if len(message.processor_path) >= self._max_hops:
            raise MaxHopsExceededError(self._max_hops, len(message.processor_path))

        processor = self.get_processor(processor_name)

        # Track routing in composite state
        self._composite_state.record_route(self.name, processor_name)
        self._composite_state.set_active(processor_name)

        # Record processor visit in message
        message.processor_path.append(processor_name)

        logger.trace(
            f"{self}: routing message {message.id} to {processor_name} "
            f"(path: {' → '.join(message.processor_path)})"
        )

        try:
            result = await processor.process(message)
            return result
        finally:
            self._composite_state.set_inactive(processor_name)
