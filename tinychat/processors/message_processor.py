import asyncio
import time
from abc import abstractmethod
from typing import Coroutine, List, Optional

from loguru import logger

from tinychat.utils.base_object import BaseObject
from tinychat.messages import Message, MetricMessage
from tinychat.observers.observer import (
    BaseObserver,
    MessageReceived,
    MessageProcessed,
)
from tinychat.asynchronous.manager import TaskManager
from tinychat.processors.models import SetupConfig


class MessageProcessor(BaseObject):
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

    def __init__(
        self,
        *,
        task_manager: Optional[TaskManager] = None,
        observers: Optional[List[BaseObserver]] = None,
        output_types: Optional[set[type[Message]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._task_manager = task_manager
        self._observers = observers or []
        self._output_types = output_types
        self._started = False
        self._owns_task_manager = False

    @property
    def task_manager(self) -> TaskManager:
        if not self._task_manager:
            raise Exception(f"{self} task manager is not initialized")
        return self._task_manager

    @property
    def observers(self) -> Optional[List[BaseObserver]]:
        return self._observers

    @property
    def output_types(self) -> Optional[set[type[Message]]]:
        return self._output_types

    async def setup(self, config: SetupConfig):
        """
        Initialize the processor with task manager and observers.

        If no task manager is provided, creates its own.
        """
        if self._started:
            return

        if not self._task_manager:
            self._task_manager = config.task_manager
        if not self._task_manager.is_setup():
            await self._task_manager.setup(config.task_manager_params)

        if config.observers:
            self._observers.extend(config.observers)

        self._started = True

    async def cleanup(self):
        if not self._started:
            return

        await self._task_manager.cleanup()
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

        self.create_task(self._notify_received(message), name="notify_received")

        try:
            result = await self._process(message)
        except Exception as e:
            self.create_task(self._notify_error(message, e), name="notify_error")
            raise

        self.create_task(
            self._notify_processed(message, result),
            name="notify_processed",
        )

        return result

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

    async def _notify_received(self, message: Message) -> None:
        if not self._observers:
            return

        data = MessageReceived(
            source_processor=self,
            source_message=message,
            content=message.content,
        )

        for observer in self._observers:
            try:
                return await observer.on_message_received(data)
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_message_received: {e}")

    async def _notify_processed(
        self,
        message: Message,
        result: Optional[Message],
        timestamp: int = time.monotonic_ns(),
    ) -> None:
        if not self._observers:
            return

        data = MessageProcessed(
            source_processor=self,
            source_message=message,
            content=result.content if result else None,
        )

        metric = MetricMessage(
            content=message.name,
            metric_name="message_processing_time",
            metric_value=(
                (result.timestamp if result else timestamp) - message.timestamp
            )
            / 1_000_000,
            metric_unit="ms",
        )

        for observer in self._observers:
            try:
                await observer.on_metric_recorded(metric)
                await observer.on_message_processed(data)
            except Exception as e:
                logger.exception(
                    f"Observer {observer} failed on_message_processed or on_metric_recorded: {e}"
                )

    async def _notify_error(self, message: Message, exception: Exception) -> None:
        if not self._observers:
            return

        for observer in self._observers:
            try:
                return await observer.on_exception(
                    source_message=message, exception=exception
                )
            except Exception as e:
                logger.exception(f"Observer {observer} failed on_exception: {e}")
