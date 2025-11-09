from enum import Enum
import asyncio
from loguru import logger


class ProcessingState(Enum):
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"
    COMPLETED = "completed"


class StateManager:
    """
    Manages processing state for a Conversation.

    Handles:
    - State transitions (running, error, stopped, completed)
    - Stop/resume with async coordination
    - State queries
    """

    def __init__(self, initial_state: ProcessingState = ProcessingState.RUNNING):
        self._state = initial_state
        self._stop_event = asyncio.Event()
        if initial_state == ProcessingState.RUNNING:
            self._stop_event.set()

    @property
    def state(self) -> ProcessingState:
        return self._state

    def set_state(self, state: ProcessingState):
        self._state = state
        if state == ProcessingState.RUNNING:
            self._stop_event.set()
        else:
            self._stop_event.clear()
        logger.debug(f"StateManager: State changed to {state.value}")

    def resume(self):
        self.set_state(ProcessingState.RUNNING)

    def stop(self):
        self.set_state(ProcessingState.STOPPED)

    def set_error(self):
        self.set_state(ProcessingState.ERROR)

    def complete(self):
        self.set_state(ProcessingState.COMPLETED)

    async def wait_if_stopped(self):
        await self._stop_event.wait()

    def is_running(self) -> bool:
        return self._state == ProcessingState.RUNNING

    def is_error(self) -> bool:
        return self._state == ProcessingState.ERROR

    def is_stopped(self) -> bool:
        return self._state == ProcessingState.STOPPED
