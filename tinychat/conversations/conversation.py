import asyncio
import time
from typing import List, Optional

from loguru import logger

from tinychat.messages.messages import Message
from tinychat.observers.observer import BaseObserver
from tinychat.processors.message_processor import CompositeProcessor, ProcessorSetup
from tinychat.events.router import EventRouter
from tinychat.state.state import ConversationState, AgentState, StateEntry
from tinychat.asynchronous.manager import (
    BaseTaskManager,
    TaskManager,
    TaskManagerParams,
)
from tinychat.observers.observer import StateChanged, ProcessorCalled


class Conversation(CompositeProcessor):
    """
    A specialized CompositeProcessor for managing conversations.

    Conversation extends CompositeProcessor to add:
    - Conversation-specific state (phase, current processor)
    - Shared agent state for coordination
    - Event router for event-driven routing (optional)
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
        enable_router: bool = False,
    ):
        super().__init__(name=conversation_id, processors=processors)
        self._conversation_id = conversation_id
        self._task_manager = task_manager or TaskManager()
        self._conversation_state = ConversationState(conversation_id)
        self._agent_state = AgentState(conversation_id, agent_id="shared")

        # Override observers from parent
        if observers:
            self._observers = observers

        # Optional event router for backward compatibility
        self._router = EventRouter(self) if enable_router else None
        self._setup_complete = False

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def conversation_state(self) -> ConversationState:
        """Conversation-specific state (phase, current processor)."""
        return self._conversation_state

    @property
    def agent_state(self) -> AgentState:
        """Shared state for agents to coordinate and share data."""
        return self._agent_state

    @property
    def router(self) -> Optional[EventRouter]:
        """Event router (only available if enable_router=True)."""
        return self._router

    def __str__(self) -> str:
        return f"Conversation({self._conversation_id})"

    def add_observer(self, observer: BaseObserver):
        """Add an observer to this conversation."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: BaseObserver):
        """Remove an observer from this conversation."""
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

        # Setup task manager
        params = TaskManagerParams(loop=loop)
        self._task_manager.setup(params)

        # Add state transition callback
        self._conversation_state.add_transition_callback(self._on_state_transition)

        # Setup composite (which sets up all sub-processors and injects peers)
        setup = ProcessorSetup(
            task_manager=self._task_manager, observers=self._observers
        )
        await super().setup(setup)

        self._setup_complete = True

    async def cleanup(self):
        """Clean up conversation and all sub-processors."""
        if self._router:
            await self._router.cleanup()

        await super().cleanup()
        self._setup_complete = False

    async def _process(self, message: Message) -> Optional[Message]:
        """
        Process message through the event router (if enabled).

        If router is enabled, delegates to event handlers.
        Otherwise, raises NotImplementedError (subclasses should override or use route_to).
        """
        if self._router:
            logger.trace(
                f"{self}: processing message {message.id} ({type(message).__name__}) via event router"
            )
            results = await self._router.emit(message)
            return results[-1] if results else None
        else:
            raise NotImplementedError(
                f"{self}: Conversation requires either enable_router=True or "
                "overriding _process() to define routing logic"
            )

    async def route_to(
        self, processor_name: str, message: Message
    ) -> Optional[Message]:
        """
        Route a message directly to a specific sub-processor by name.

        Updates conversation state and invokes the processor.
        """
        logger.trace(
            f"{self}: routing message {message.id} ({type(message).__name__}) to {processor_name}"
        )

        # Update conversation state
        self._conversation_state.update_processor(
            processor_name, {"message_id": message.id}
        )

        # Use parent's route_to which tracks composite state
        return await super().route_to(processor_name, message)

    async def notify_processor_called(self, source, target, message: Message):
        """Notify observers that one processor called another."""
        if not self._observers:
            return

        data = ProcessorCalled(
            source=source, target=target, message=message, timestamp=time.time_ns()
        )

        for observer in self._observers:
            try:
                await observer.on_processor_called(data)
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_processor_called: {e}")

    async def _on_state_transition(self, entry: StateEntry):
        """Handle state transitions and notify observers."""
        if not self._observers:
            return

        # Only notify observers for phase changes (not processor changes)
        if entry.name != "phase":
            return

        # Get previous phase from history
        phase_history = self._conversation_state.get_history_by_name("phase")
        previous_phase = phase_history[-2].value if len(phase_history) >= 2 else "idle"

        data = StateChanged(
            conversation_id=self._conversation_id,
            previous_state=previous_phase,
            new_state=entry.value,
            timestamp=entry.timestamp,
            metadata=entry.metadata,
        )

        for observer in self._observers:
            try:
                await observer.on_state_changed(data)
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_state_changed: {e}")
