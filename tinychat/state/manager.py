from enum import Enum
import asyncio
from loguru import logger


class ProcessingState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class StateManager:
    """
    Manages processing state for a Conversation.

    Handles:
    - State transitions (running, paused, error, stopped)
    - Pause/resume with async coordination
    - State queries
    """

    def __init__(self, initial_state: ProcessingState = ProcessingState.RUNNING):
        self._state = initial_state
        self._pause_event = asyncio.Event()
        if initial_state == ProcessingState.RUNNING:
            self._pause_event.set()

    @property
    def state(self) -> ProcessingState:
        return self._state

    def set_state(self, state: ProcessingState):
        self._state = state
        if state == ProcessingState.RUNNING:
            self._pause_event.set()
        else:
            self._pause_event.clear()
        logger.debug(f"StateManager: State changed to {state.value}")

    def pause(self):
        self.set_state(ProcessingState.PAUSED)

    def resume(self):
        self.set_state(ProcessingState.RUNNING)

    def stop(self):
        self.set_state(ProcessingState.STOPPED)

    def set_error(self):
        self.set_state(ProcessingState.ERROR)

    async def wait_if_paused(self):
        await self._pause_event.wait()

    def is_running(self) -> bool:
        return self._state == ProcessingState.RUNNING

    def is_paused(self) -> bool:
        return self._state == ProcessingState.PAUSED

    def is_error(self) -> bool:
        return self._state == ProcessingState.ERROR

    def is_stopped(self) -> bool:
        return self._state == ProcessingState.STOPPED
