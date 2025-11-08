import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import Coroutine, List, Optional

from loguru import logger

from tinychat.messages.messages import ErrorMessage, Message
from tinychat.observers.observer import (
    BaseObserver,
    MessageReceived,
    MessageProcessed,
)
from tinychat.asynchronous.manager import (
    BaseTaskManager,
    TaskManager,
    TaskManagerParams,
)
from tinychat.utils.utils import random_id


@dataclass
class ProcessorSetup:
    task_manager: BaseTaskManager = TaskManager()
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
        self._id = random_id()
        self._name = name or f"{self.__class__.__name__}_{self._id}"
        self._task_manager: Optional[BaseTaskManager] = None
        self._observers: Optional[List[BaseObserver]] = None
        self._started = False
        self._owns_task_manager = False

    @property
    def id(self) -> str:
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
    def observers(self) -> Optional[List[BaseObserver]]:
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
            if self._observers:
                self._observers.extend(setup.observers)
            else:
                self._observers = setup.observers

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

        self.create_task(
            self._notify_received(message), name=f"{self}::notify_received"
        )

        try:
            result = await self._process(message)
            self.create_task(
                self._notify_processed(message, result),
                name=f"{self}::notify_processed",
            )
            return result

        except Exception as e:
            error_msg = await self.handle_error(e, message)
            self.create_task(
                self._notify_error(message, error_msg), name=f"{self}::notify_error"
            )
            return error_msg

    @abstractmethod
    async def _process(self, message: Message) -> Optional[Message]:
        """
        Process a message and return the result.

        Subclasses must implement this to define their message processing logic.
        This method receives a message and should return:
        - A new/modified message
        - The same message passed through
        - None if no result
        """
        ...

    async def handle_error(self, error: Exception, message: Message) -> ErrorMessage:
        logger.exception(f"{self}: error processing message {message}: {error}")

        return ErrorMessage(
            content=str(error),
            source_message=message,
            fatal=False,
        )

    async def _notify_received(self, message: Message) -> None:
        if not self._observers:
            return

        data = MessageReceived(
            processor=self,
            message=message,
            content=message.content,
        )

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
            content=result.content if result else "No result",
        )

        for observer in self._observers:
            try:
                await observer.on_message_processed(data)
            except Exception as e:
                logger.exception(
                    f"Observer {observer} failed on_message_processed: {e}"
                )

    async def _notify_error(self, message: Message, error_msg: ErrorMessage) -> None:
        if not self._observers:
            return

        for observer in self._observers:
            try:
                await observer.on_error_message(
                    source_message=message, error_msg=error_msg
                )
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_error_message: {e}")
