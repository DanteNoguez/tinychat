from typing import Optional
from loguru import logger

from tinychat.messages.messages import (
    Message,
    IngressMessage,
    EgressMessage,
    ErrorMessage,
)
from tinychat.processors.message_processor import MessageProcessor, ProcessorSetup
from tinychat.asynchronous.manager import BaseTaskManager, TaskManager
from tinychat.observers.observer import BaseObserver


class MessageBus:
    """
    Sequential message bus.

    Each message type has exactly one subscriber.
    Processors publish messages; bus routes to registered handler.
    Automatic chaining: processor results are re-published until terminal.
    """

    def __init__(
        self,
        handlers: dict[type[Message], MessageProcessor],
        task_manager: Optional[BaseTaskManager] = None,
        observers: Optional[list[BaseObserver]] = None,
        max_depth: int = 30,
    ):
        self._handlers = handlers
        self._processors = {p.id: p for p in handlers.values()}
        self._max_depth = max_depth
        self._task_manager = task_manager or TaskManager()
        self._observers = observers
        self._setup_complete = False

        # Reverse map for logging: processor → message types it handles
        self._processor_types: dict[str, list[type[Message]]] = {}
        for msg_type, processor in handlers.items():
            if processor.id not in self._processor_types:
                self._processor_types[processor.id] = []
            self._processor_types[processor.id].append(msg_type)
        logger.debug(f"MessageBus: Initialized with topology: {self.get_topology()}")

    async def setup(self):
        if self._setup_complete:
            return

        setup = ProcessorSetup(
            task_manager=self._task_manager,
            observers=self._observers,
        )

        for processor in self._processors.values():
            await processor.setup(setup)

        self._setup_complete = True

    @property
    def observers(self) -> Optional[list[BaseObserver]]:
        return self._observers

    def get_handler(self, message_type: type[Message]) -> Optional[MessageProcessor]:
        return self._handlers.get(message_type)

    async def process(
        self,
        message: IngressMessage,
        source_id: Optional[str] = None,
    ) -> Optional[Message]:
        """
        Process message and automatically chain routing.

        Finds handler for message type, processes it, and:
        - If result is None: stops, returns None
        - If result is terminal (EgressMessage): stops, returns it
        - Otherwise: re-publishes result (recursive chaining)
        """
        return await self._publish_recursive(message, source_id, depth=0)

    async def _publish_recursive(
        self,
        message: Message,
        source_id: Optional[str],
        depth: int,
    ) -> Optional[Message]:
        if depth >= self._max_depth:
            raise RuntimeError(
                f"Maximum routing depth {self._max_depth} exceeded at message {message.name}"
            )

        # Get handler for message type
        handler = self.get_handler(type(message))
        if not handler:
            raise ValueError(f"No handler for {type(message).__name__}")

        # Process message
        result = await handler.process(message)

        # Terminal conditions
        if result is None:
            return None
        if isinstance(result, EgressMessage):
            return result
        elif isinstance(result, ErrorMessage):
            return result

        # Continue chain: publish result
        return await self._publish_recursive(result, handler.id, depth + 1)

    def get_topology(self) -> dict[str, list[str]]:
        """
        Get topology map for visualization/debugging.

        Returns:
            Dictionary mapping processor names to message types they handle
        """
        topology = {}
        for processor_id, msg_types in self._processor_types.items():
            processor = self._processors[processor_id]
            topology[processor.name] = [mt.__name__ for mt in msg_types]
        return topology
