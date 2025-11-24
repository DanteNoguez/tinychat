from typing import Optional
import asyncio
from loguru import logger

from tinychat.messages import (
    Message,
    EgressMessage,
)
from tinychat.processors.message_processor import MessageProcessor
from tinychat.processors.models import SetupConfig, Topology, ProcessorNode
from tinychat.processors.exceptions import MaxHopsExceededError


class CompositeProcessor(MessageProcessor):
    """
    An event-driven composite that integrates multiple processors together.

    Each message type has exactly one subscriber.
    Processors produce messages; composite routes to registered consumer/handler.
    Automatic chaining: processor results are re-published until terminal.
    An EgressMessage produced by a processor signals processing termination.
    """

    def __init__(
        self,
        *,
        handlers: dict[type[Message], MessageProcessor],
        max_hops: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._handlers = handlers
        self._processors = {p.id: p for p in handlers.values()}
        self._max_hops = max_hops
        self._setup_complete = False
        self._current_task: Optional[asyncio.Task] = None

        # Reverse map for logging: processor → message types it handles
        self._processor_types: dict[str, list[type[Message]]] = {}
        for msg_type, processor in handlers.items():
            if processor.id not in self._processor_types:
                self._processor_types[processor.id] = []
            self._processor_types[processor.id].append(msg_type)
        # Validate topology if processors declare output types
        self._validate_topology()

        logger.debug(f"[{self}] Topology: ")
        for node in self.topology.nodes:
            logger.debug(
                f"{node.name}: handles={node.handles}, produces={node.produces}"
            )

    async def setup(self, config: SetupConfig):
        if self._setup_complete:
            return

        self._task_manager = config.task_manager
        if not self._task_manager.is_setup():
            await self._task_manager.setup(config.task_manager_params)

        if config.observers:
            self._observers.extend(config.observers)

        for processor in self._processors.values():
            await processor.setup(config)

        self._setup_complete = True

    def get_handler(self, message_type: type[Message]) -> Optional[MessageProcessor]:
        return self._handlers.get(message_type)

    async def _process(
        self,
        message: Message,
        hops: int = 0,
    ) -> Optional[Message]:
        """
        Process message and automatically chain routing.

        Finds handler for message type, processes it, and:
        - If result is None: stops, returns None
        - If result is terminal (EgressMessage): stops, returns it
        - Otherwise: re-publishes result (iterative chaining)
        - Stops if interrupted via interrupt_processing()
        """
        while hops < self._max_hops:
            # Get handler for message type
            handler = self.get_handler(type(message))
            if not handler:
                raise ValueError(f"No handler for {type(message).__name__}")

            # Process message with cancellation support
            try:
                task = self.create_task(
                    handler.process(message), name=f"process::{handler.name}"
                )
                self._current_task = task
                result = await task
            except asyncio.CancelledError:
                return None
            finally:
                self._current_task = None

            # Terminal conditions
            if result is None:
                return None
            elif isinstance(result, EgressMessage):
                # Egress messages signal the end of the chain
                return result

            # Continue chain with result
            message = result
            hops += 1

        raise MaxHopsExceededError(
            source_message=message,
            source_processor=self,
            max_hops=self._max_hops,
            current_hops=hops,
        )

    async def interrupt_processing(self):
        """
        Interrupt ongoing message processing.

        Cancels the current handler task if one is running.
        Sets interruption event to prevent new iterations.
        """
        if self._current_task:
            await self._task_manager.cancel_task(self._current_task, timeout=1.0)

    def _validate_topology(self) -> None:
        """
        Validate topology if processors declare output types.

        Checks:
        1. All declared output types (except terminals) have handlers.
        2. At least one terminal type is produced (prevent infinite loops).
        """
        unhandled_types: set[tuple[str, str]] = set()
        has_terminal = False
        all_typed = True

        for processor in self._processors.values():
            if processor.output_types is None:
                all_typed = False
                continue

            for output_type in processor.output_types:
                is_terminal = output_type is type(None) or (
                    isinstance(output_type, type)
                    and issubclass(output_type, EgressMessage)
                )

                if is_terminal:
                    has_terminal = True
                    continue

                if output_type not in self._handlers:
                    unhandled_types.add((processor.name, output_type.__name__))

        if unhandled_types:
            error_lines = [
                f"  - {proc} produces {msg_type} (no handler registered)"
                for proc, msg_type in sorted(unhandled_types)
            ]
            raise ValueError(
                "Topology validation failed. Unhandled message types:\n"
                + "\n".join(error_lines)
            )

        if all_typed and not has_terminal:
            raise ValueError(
                "Topology validation failed. No terminal state detected.\n"
                "The graph forms a closed loop and will always crash with MaxHopsExceededError.\n"
                "Ensure at least one processor produces 'EgressMessage' (or subclass) or 'None'."
            )

    @property
    def topology(self) -> Topology:
        """
        Get topology graph for visualization/debugging.

        Returns:
            Topology object with nodes representing each processor and their
            input/output types. Output types are included when declared via
            output_types parameter. Terminal types (EgressMessage, None) are
            shown in visualization.
        """
        nodes = []
        for processor_id, msg_types in self._processor_types.items():
            processor = self._processors[processor_id]
            handles = [mt.__name__ for mt in msg_types]

            if processor.output_types is not None:
                produces = []
                for ot in processor.output_types:
                    if ot is type(None):
                        produces.append("None")
                    else:
                        produces.append(ot.__name__)
            else:
                produces = ["undefined"]

            nodes.append(
                ProcessorNode(
                    name=processor.name,
                    handles=handles,
                    produces=produces,
                )
            )

        return Topology(nodes=nodes)
