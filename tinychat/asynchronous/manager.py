import asyncio
from dataclasses import dataclass
from typing import Coroutine, Dict, Optional, Sequence

from loguru import logger


@dataclass
class TaskManagerParams:
    loop: asyncio.AbstractEventLoop


@dataclass
class TaskData:
    task: asyncio.Task


class TaskManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, TaskData] = {}
        self._params: Optional[TaskManagerParams] = None

    async def setup(self, params: TaskManagerParams):
        if not self._params:
            self._params = params

    def is_setup(self) -> bool:
        return self._params is not None

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        if not self._params:
            raise Exception("TaskManager is not setup: unable to get event loop")
        return self._params.loop

    def create_task(self, coroutine: Coroutine, name: str) -> asyncio.Task:
        if not self._params:
            raise Exception("TaskManager is not setup: unable to get event loop")

        async def run_coroutine():
            try:
                return await coroutine
            except Exception as e:
                logger.exception(f"{name}: unexpected exception: {e}")
                raise

        task = self._params.loop.create_task(run_coroutine())
        task.set_name(name)
        task.add_done_callback(self._task_done_handler)
        self._add_task(TaskData(task=task))
        logger.trace(f"{name}: task created")
        return task

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        name = task.get_name()
        task.cancel()
        try:
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.TimeoutError:
            logger.warning(f"{name}: timed out waiting for task to cancel")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"{name}: unexpected exception while cancelling task: {e}")
        except BaseException as e:
            logger.critical(f"{name}: fatal base exception while cancelling task: {e}")
            raise

    def current_tasks(self) -> Sequence[asyncio.Task]:
        return [data.task for data in self._tasks.values()]

    def _add_task(self, task_data: TaskData):
        name = task_data.task.get_name()
        self._tasks[name] = task_data

    def _task_done_handler(self, task: asyncio.Task):
        name = task.get_name()
        logger.trace(f"{name}: task done")
        try:
            del self._tasks[name]
        except KeyError as e:
            logger.trace(f"{name}: unable to remove task data (already removed?): {e}")

    async def cleanup(self):
        for task in self._tasks.values():
            await self.cancel_task(task.task)
        self._tasks.clear()
