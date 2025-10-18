"""Tests for task manager classes."""
import asyncio
import pytest

from tinychat.asynchronous.manager import TaskManager, TaskManagerParams


class TestTaskManager:
    """Test TaskManager class."""

    def test_task_manager_instantiation(self):
        """Test that TaskManager can be instantiated."""
        manager = TaskManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_task_manager_setup(self):
        """Test task manager setup."""
        manager = TaskManager()
        loop = asyncio.get_event_loop()
        params = TaskManagerParams(loop=loop)
        
        manager.setup(params)
        assert manager.get_event_loop() == loop

    @pytest.mark.asyncio
    async def test_task_manager_get_event_loop_without_setup(self):
        """Test get_event_loop raises without setup."""
        manager = TaskManager()
        
        with pytest.raises(Exception, match="TaskManager is not setup"):
            manager.get_event_loop()

    @pytest.mark.asyncio
    async def test_task_manager_create_task(self):
        """Test creating a task."""
        manager = TaskManager()
        loop = asyncio.get_event_loop()
        params = TaskManagerParams(loop=loop)
        manager.setup(params)

        executed = []

        async def dummy_task():
            executed.append(True)
            await asyncio.sleep(0.01)

        task = manager.create_task(dummy_task(), "test_task")
        assert task is not None
        assert task.get_name() == "test_task"
        
        await task
        assert executed == [True]

    @pytest.mark.asyncio
    async def test_task_manager_current_tasks(self):
        """Test tracking current tasks."""
        manager = TaskManager()
        loop = asyncio.get_event_loop()
        params = TaskManagerParams(loop=loop)
        manager.setup(params)

        async def dummy_task():
            await asyncio.sleep(0.1)

        task = manager.create_task(dummy_task(), "test_task")
        tasks = manager.current_tasks()
        
        assert len(tasks) == 1
        assert task in tasks
        
        await task

    @pytest.mark.asyncio
    async def test_task_manager_cancel_task(self):
        """Test cancelling a task."""
        manager = TaskManager()
        loop = asyncio.get_event_loop()
        params = TaskManagerParams(loop=loop)
        manager.setup(params)

        async def long_task():
            await asyncio.sleep(10)

        task = manager.create_task(long_task(), "test_task")
        await manager.cancel_task(task, timeout=0.1)
        
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_task_manager_create_task_without_setup(self):
        """Test create_task raises without setup."""
        manager = TaskManager()

        async def dummy_task():
            pass

        with pytest.raises(Exception, match="TaskManager is not setup"):
            manager.create_task(dummy_task(), "test_task")

