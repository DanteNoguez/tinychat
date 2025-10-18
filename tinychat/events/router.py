import asyncio
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from loguru import logger

from tinychat.messages.messages import Message

if TYPE_CHECKING:
    from tinychat.conversations.conversation import Conversation
    from tinychat.processors.message_processor import MessageProcessor


# Type aliases for clarity
HandlerFunc = Callable[
    [Message, "Conversation"], Coroutine[Any, Any, Optional[Message]]
]
EventName = Union[str, Type[Message]]


@dataclass
class EventHandler:
    """Represents a registered event handler."""

    event_name: EventName
    handler: HandlerFunc
    name: Optional[str] = None
    concurrent: bool = False  # Whether this handler runs concurrently with others


@dataclass
class EventContext:
    """Context passed to event handlers."""

    message: Message
    conversation: "Conversation"
    source_processor: Optional["MessageProcessor"] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EventRouter:
    """
    Event-driven router that orchestrates message flow through processors.

    Instead of hardcoded routing logic, handlers react to messages and events:
    - Handlers are registered for message types or custom event names
    - Processors emit events that trigger handlers
    - Supports concurrent execution of handlers
    - Naturally handles cycles and dynamic routing

    Example:
        router = EventRouter(conversation)

        # Register handler for UserMessage type
        @router.on(UserMessage)
        async def handle_user(msg, conv):
            ctx = await conv.route_to("history", msg)
            ctx = await conv.route_to("crm", ctx)
            return await conv.route_to("orchestrator", ctx)

        # Register handler for custom events
        @router.on("needs_data_refresh")
        async def refresh_data(msg, conv):
            return await conv.route_to("crm", msg)

        # Emit events from processors
        await router.emit("needs_data_refresh", message)
    """

    def __init__(self, conversation: "Conversation"):
        self._conversation = conversation
        self._handlers: Dict[EventName, List[EventHandler]] = {}
        self._running_tasks: List[asyncio.Task] = []

    @property
    def conversation(self) -> "Conversation":
        return self._conversation

    def on(
        self, event: EventName, *, name: Optional[str] = None, concurrent: bool = False
    ) -> Callable[[HandlerFunc], HandlerFunc]:
        """
        Decorator to register an event handler.

        Args:
            event: Message type class or custom event name string
            name: Optional name for the handler (for debugging)
            concurrent: If True, handler runs concurrently with other handlers

        Example:
            @router.on(UserMessage)
            async def handle_user(message, conversation):
                return await conversation.route_to("history", message)

            @router.on("needs_refresh", concurrent=True)
            async def refresh(message, conversation):
                await conversation.route_to("crm", message)
        """

        def decorator(handler: HandlerFunc) -> HandlerFunc:
            self.register(event, handler, name=name, concurrent=concurrent)
            return handler

        return decorator

    def register(
        self,
        event: EventName,
        handler: HandlerFunc,
        *,
        name: Optional[str] = None,
        concurrent: bool = False,
    ):
        """Register an event handler programmatically."""
        if event not in self._handlers:
            self._handlers[event] = []

        handler_name = name or handler.__name__
        event_handler = EventHandler(
            event_name=event, handler=handler, name=handler_name, concurrent=concurrent
        )

        self._handlers[event].append(event_handler)

        event_display = event if isinstance(event, str) else event.__name__
        logger.debug(
            f"EventRouter: registered handler '{handler_name}' for event '{event_display}'"
        )

    def unregister(self, event: EventName, handler_name: Optional[str] = None):
        """Unregister handler(s) for an event."""
        if event not in self._handlers:
            return

        if handler_name:
            self._handlers[event] = [
                h for h in self._handlers[event] if h.name != handler_name
            ]
        else:
            del self._handlers[event]

    async def emit(
        self,
        event: Union[str, Message],
        message: Optional[Message] = None,
        *,
        source_processor: Optional["MessageProcessor"] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Optional[Message]]:
        """
        Emit an event, triggering all registered handlers.

        Args:
            event: Event name (string) or Message instance
            message: Message to pass to handlers (required if event is a string)
            source_processor: Processor that emitted the event
            metadata: Additional context

        Returns:
            List of results from all handlers (sequential handlers only)

        Example:
            # Emit custom event
            await router.emit("needs_data_refresh", message)

            # Emit message (will match type-based handlers)
            result = await router.emit(UserMessage(...))
        """
        # Determine event name and message
        if isinstance(event, Message):
            event_name = type(event)
            msg = event
        else:
            event_name = event
            if message is None:
                raise ValueError("message is required when event is a string")
            msg = message

        # Find matching handlers
        handlers = self._handlers.get(event_name, [])

        if not handlers:
            event_display = (
                event_name if isinstance(event_name, str) else event_name.__name__
            )
            logger.trace(f"EventRouter: no handlers for event '{event_display}'")
            return []

        event_display = (
            event_name if isinstance(event_name, str) else event_name.__name__
        )
        logger.debug(
            f"EventRouter: emitting event '{event_display}' to {len(handlers)} handler(s)"
        )

        # Separate concurrent and sequential handlers
        concurrent_handlers = [h for h in handlers if h.concurrent]
        sequential_handlers = [h for h in handlers if not h.concurrent]

        results = []

        # Execute sequential handlers in order
        for handler in sequential_handlers:
            try:
                logger.trace(
                    f"EventRouter: executing handler '{handler.name}' for '{event_display}'"
                )
                result = await handler.handler(msg, self._conversation)
                results.append(result)

                # If handler returns a message, update msg for next handler
                if result is not None:
                    msg = result

            except Exception as e:
                logger.exception(
                    f"EventRouter: handler '{handler.name}' failed for event '{event_display}': {e}"
                )
                raise

        # Execute concurrent handlers in parallel (don't wait for results)
        if concurrent_handlers:
            for handler in concurrent_handlers:
                task = asyncio.create_task(
                    self._execute_concurrent_handler(handler, msg, event_display)
                )
                self._running_tasks.append(task)

        return results

    async def _execute_concurrent_handler(
        self, handler: EventHandler, message: Message, event_display: str
    ):
        """Execute a concurrent handler and handle errors."""
        try:
            logger.trace(
                f"EventRouter: executing concurrent handler '{handler.name}' for '{event_display}'"
            )
            await handler.handler(message, self._conversation)
        except Exception as e:
            logger.exception(
                f"EventRouter: concurrent handler '{handler.name}' failed for '{event_display}': {e}"
            )
        finally:
            # Clean up task from running list
            if asyncio.current_task() in self._running_tasks:
                self._running_tasks.remove(asyncio.current_task())

    async def wait_for_concurrent_handlers(self, timeout: Optional[float] = None):
        """Wait for all concurrent handlers to complete."""
        if not self._running_tasks:
            return

        logger.debug(
            f"EventRouter: waiting for {len(self._running_tasks)} concurrent handlers"
        )

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._running_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("EventRouter: timeout waiting for concurrent handlers")
            # Cancel remaining tasks
            for task in self._running_tasks:
                if not task.done():
                    task.cancel()

    async def cleanup(self):
        """Cancel all running concurrent handlers."""
        for task in self._running_tasks:
            if not task.done():
                task.cancel()

        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        self._running_tasks.clear()

    def get_handlers(self, event: EventName) -> List[EventHandler]:
        """Get all handlers registered for an event."""
        return self._handlers.get(event, []).copy()

    def list_events(self) -> List[EventName]:
        """List all events that have registered handlers."""
        return list(self._handlers.keys())

    def __repr__(self) -> str:
        event_count = len(self._handlers)
        handler_count = sum(len(handlers) for handlers in self._handlers.values())
        return f"EventRouter(events={event_count}, handlers={handler_count})"
