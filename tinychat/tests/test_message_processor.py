"""Tests for message processor classes."""

import asyncio
import pytest

from tinychat.messages.messages import Message
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.asynchronous.manager import TaskManager, TaskManagerParams


class SimpleProcessor(MessageProcessor):
    """Simple processor for testing."""

    async def _process(self, message: Message) -> Message:
        """Echo the message back."""
        return message


class TestMessageProcessor:
    """Test MessageProcessor class."""

    def test_message_processor_instantiation(self):
        """Test that MessageProcessor can be instantiated."""
        processor = SimpleProcessor()
        assert processor is not None
        assert processor.name.startswith("SimpleProcessor_")

    def test_message_processor_with_custom_name(self):
        """Test processor with custom name."""
        processor = SimpleProcessor(name="my_processor")
        assert processor.name == "my_processor"

    def test_message_processor_id(self):
        """Test processor has unique id."""
        processor1 = SimpleProcessor()
        processor2 = SimpleProcessor()
        assert processor1.id != processor2.id

    @pytest.mark.asyncio
    async def test_message_processor_setup(self):
        """Test processor setup."""
        processor = SimpleProcessor(name="test_processor")
        loop = asyncio.get_running_loop()

        config = SetupConfig(
            task_manager=TaskManager(),
            task_manager_params=TaskManagerParams(loop=loop),
        )
        await processor.setup(config)

        assert processor._task_manager is not None
        assert processor._started is True

    @pytest.mark.asyncio
    async def test_message_processor_cleanup(self):
        """Test processor cleanup."""
        processor = SimpleProcessor(name="test_processor")
        loop = asyncio.get_running_loop()

        config = SetupConfig(
            task_manager=TaskManager(),
            task_manager_params=TaskManagerParams(loop=loop),
        )
        await processor.setup(config)
        await processor.cleanup()

        assert processor._started is False

    @pytest.mark.asyncio
    async def test_message_processor_process(self):
        """Test processing a message."""
        processor = SimpleProcessor(name="test_processor")
        loop = asyncio.get_running_loop()

        config = SetupConfig(
            task_manager=TaskManager(),
            task_manager_params=TaskManagerParams(loop=loop),
        )
        await processor.setup(config)

        msg = Message(content="test message")
        result = await processor.process(msg)

        assert result is not None
        assert result.id == msg.id

    @pytest.mark.asyncio
    async def test_message_processor_properties_without_setup(self):
        """Test accessing properties without setup raises."""
        processor = SimpleProcessor()

        with pytest.raises(Exception, match="task manager is not initialized"):
            _ = processor.task_manager

    def test_message_processor_str_repr(self):
        """Test string representations."""
        processor = SimpleProcessor(name="my_processor")
        assert str(processor) == "my_processor"
        assert "SimpleProcessor" in repr(processor)
        assert "my_processor" in repr(processor)


class FailingProcessor(MessageProcessor):
    """Processor that raises an exception."""

    async def _process(self, message: Message) -> Message:
        raise ValueError("Processing failed")


class TestMessageProcessorErrorHandling:
    """Test error handling in MessageProcessor."""

    @pytest.mark.asyncio
    async def test_message_processor_handle_error(self):
        """Test error handling in processor."""
        processor = FailingProcessor(name="failing_processor")
        loop = asyncio.get_running_loop()

        config = SetupConfig(
            task_manager=TaskManager(),
            task_manager_params=TaskManagerParams(loop=loop),
        )
        await processor.setup(config)

        msg = Message(content="test message")

        with pytest.raises(ValueError, match="Processing failed"):
            await processor.process(msg)
