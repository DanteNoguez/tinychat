import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


@dataclass
class StateEntry:
    """
    Represents a single state change entry in the history.
    
    Each entry captures what changed, the new value, when it happened,
    and any additional context through metadata.
    """

    name: str  # Name of the state field that changed (e.g., "phase", "processor")
    value: Any  # The new value
    timestamp: int = field(default_factory=time.time_ns) # Nanosecond timestamp of when this change occurred
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def __repr__(self) -> str:
        return f"StateEntry(name={self.name}, value={self.value}, timestamp={self.timestamp})"


class State(ABC):
    """
    Base class for all state management in tinychat.
    
    States are typed objects with defined fields at initialization.
    All state changes are tracked in an ordered history of StateEntry objects.
    
    Subclasses should:
    - Define typed properties for state fields
    - Call _record_change() when state values are updated
    - Optionally set transition callbacks for state changes
    """

    def __init__(self, conversation_id: str):
        self._conversation_id = conversation_id
        self._history: List[StateEntry] = []
        self._transition_callbacks: List[Callable] = []

    @property
    def conversation_id(self) -> str:
        """The conversation this state belongs to."""
        return self._conversation_id

    @property
    def history(self) -> List[StateEntry]:
        """Ordered history of all state changes (read-only copy)."""
        return self._history.copy()

    def add_transition_callback(self, callback: Callable) -> None:
        """
        Register a callback to be invoked on state changes.
        
        Callback signature: async def callback(entry: StateEntry) -> None
        """
        if callback not in self._transition_callbacks:
            self._transition_callbacks.append(callback)

    def remove_transition_callback(self, callback: Callable) -> None:
        """Remove a previously registered callback."""
        if callback in self._transition_callbacks:
            self._transition_callbacks.remove(callback)

    def _record_change(
        self, name: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> StateEntry:
        """
        Record a state change in the history.
        
        This creates a StateEntry and adds it to the history.
        Should be called by subclasses whenever a state field changes.
        """
        entry = StateEntry(
            name=name,
            value=value,
            timestamp=time.time_ns(),
            metadata=metadata or {},
        )
        self._history.append(entry)
        return entry

    async def _notify_transition(self, entry: StateEntry) -> None:
        """Notify all registered callbacks about a state change."""
        for callback in self._transition_callbacks:
            try:
                await callback(entry)
            except Exception as e:
                logger.exception(f"Transition callback failed for {entry}: {e}")

    def get_history_by_name(self, name: str) -> List[StateEntry]:
        """Get all history entries for a specific state field."""
        return [entry for entry in self._history if entry.name == name]

    def get_latest_entry(self, name: str) -> Optional[StateEntry]:
        """Get the most recent entry for a specific state field."""
        entries = self.get_history_by_name(name)
        return entries[-1] if entries else None

    def clear_history(self) -> None:
        """Clear the state history (use with caution)."""
        self._history.clear()
        logger.trace(f"State[{self._conversation_id}]: history cleared")


class ConversationState(State):
    """
    Tracks the current phase and processor of a conversation with transition history.
    
    This state manages the flow of a conversation through different phases
    and tracks which processor is currently handling messages.
    """

    def __init__(self, conversation_id: str, initial_phase: str = "idle"):
        super().__init__(conversation_id)
        self._phase: str = initial_phase
        self._current_processor: Optional[str] = None

    @property
    def phase(self) -> str:
        """Current phase of the conversation."""
        return self._phase

    @property
    def current_processor(self) -> Optional[str]:
        """Name of the processor currently handling the conversation."""
        return self._current_processor

    async def update_phase(
        self, phase: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the conversation phase.
        
        Records the change in history and notifies callbacks of the transition.
        """
        previous = self._phase
        self._phase = phase

        entry = self._record_change("phase", phase, metadata)

        logger.trace(
            f"ConversationState[{self._conversation_id}]: phase {previous} → {phase}"
        )

        await self._notify_transition(entry)

    def update_processor(
        self, processor_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the current processor handling the conversation.
        
        This is typically called when routing messages to different processors.
        Note: This is synchronous as it's usually called during routing.
        """
        previous = self._current_processor
        self._current_processor = processor_name

        self._record_change("processor", processor_name, metadata)

        logger.trace(
            f"ConversationState[{self._conversation_id}]: processor {previous} → {processor_name}"
        )

    def __repr__(self) -> str:
        return f"ConversationState(phase={self._phase}, processor={self._current_processor})"


class AgentState(State):
    """
    Manages state shared between agents in a conversation.
    
    This state allows agents to coordinate by sharing:
    - Tool call results
    - Intermediate data
    - Context accumulated from multiple sources
    - Any domain-specific data needed for collaboration
    
    Unlike ConversationState which tracks flow control, AgentState
    tracks the actual data agents work with.
    """

    def __init__(self, conversation_id: str, agent_id: str):
        super().__init__(conversation_id)
        self._agent_id = agent_id
        self._data: Dict[str, Any] = {}

    @property
    def agent_id(self) -> str:
        """ID of the agent this state belongs to."""
        return self._agent_id

    @property
    def data(self) -> Dict[str, Any]:
        """Current state data (read-only copy)."""
        return self._data.copy()

    async def set(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set a state value and record the change in history.
        
        This is the primary way to update agent state. Each update
        is tracked and can be observed through the history.
        """
        previous = self._data.get(key)
        self._data[key] = value

        # Build metadata with previous value for context
        full_metadata = metadata or {}
        if previous is not None:
            full_metadata["previous_value"] = previous

        entry = self._record_change(key, value, full_metadata)

        logger.trace(
            f"AgentState[{self._conversation_id}][{self._agent_id}]: {key} = {value}"
        )

        await self._notify_transition(entry)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value, returning default if not found."""
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in the state."""
        return key in self._data

    async def delete(
        self, key: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Delete a key from the state and record the deletion."""
        if key in self._data:
            previous = self._data[key]
            del self._data[key]

            # Record deletion with previous value in metadata
            full_metadata = metadata or {}
            full_metadata["deleted_value"] = previous

            entry = self._record_change(key, None, full_metadata)

            logger.trace(
                f"AgentState[{self._conversation_id}][{self._agent_id}]: deleted {key}"
            )

            await self._notify_transition(entry)

    async def update(
        self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update multiple state values at once.
        
        Each key-value pair is recorded as a separate entry in the history,
        but they share the same timestamp and metadata.
        """
        for key, value in data.items():
            await self.set(key, value, metadata)

        logger.trace(
            f"AgentState[{self._conversation_id}][{self._agent_id}]: bulk update of {len(data)} keys"
        )

    def clear(self) -> None:
        """Clear all state data (history is preserved)."""
        self._data.clear()
        logger.trace(
            f"AgentState[{self._conversation_id}][{self._agent_id}]: data cleared"
        )

    def __repr__(self) -> str:
        return f"AgentState(agent_id={self._agent_id}, keys={list(self._data.keys())})"
