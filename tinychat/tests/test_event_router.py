"""Tests for event router."""
import asyncio
import pytest

from tinychat.events.router import EventRouter
from tinychat.messages.messages import Message, LLMMessage, UserMessage
from tinychat.messages.models import LLMServiceType
from tinychat.conversations.conversation import Conversation
from tinychat.processors.message_processor import MessageProcessor
from tinychat.asynchronous.manager import TaskManager, TaskManagerParams


class DummyProcessor(MessageProcessor):
    """Dummy processor for testing."""

    async def _process(self, message: Message) -> Message:
        return message


class TestEventRouter:
    """Test EventRouter class."""

    @pytest.mark.asyncio
    async def test_event_router_instantiation(self):
        """Test that EventRouter can be instantiated."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )

        router = conversation.router
        assert router is not None
        assert router.conversation == conversation

    @pytest.mark.asyncio
    async def test_event_router_register_handler(self):
        """Test registering an event handler."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        async def handler(msg, conv):
            return msg

        router.register("test_event", handler)
        handlers = router.get_handlers("test_event")
        
        assert len(handlers) == 1
        assert handlers[0].handler == handler

    @pytest.mark.asyncio
    async def test_event_router_decorator(self):
        """Test using decorator to register handler."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        @router.on("test_event")
        async def handler(msg, conv):
            return msg

        handlers = router.get_handlers("test_event")
        assert len(handlers) == 1

    @pytest.mark.asyncio
    async def test_event_router_emit_event(self):
        """Test emitting an event."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        handler_called = []

        @router.on("test_event")
        async def handler(msg, conv):
            handler_called.append(msg)
            return msg

        msg = Message()
        results = await router.emit("test_event", msg)

        assert len(handler_called) == 1
        assert handler_called[0] == msg
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_event_router_emit_message_type(self):
        """Test emitting with message type."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        handler_called = []

        @router.on(UserMessage)
        async def handler(msg, conv):
            handler_called.append(msg)
            return msg

        msg = UserMessage(
            conversation_id="conv_123",
            content="Hello",
            user_id="user_1",
            service=LLMServiceType.OPENAI,
        )
        results = await router.emit(msg)

        assert len(handler_called) == 1
        assert handler_called[0] == msg

    @pytest.mark.asyncio
    async def test_event_router_unregister(self):
        """Test unregistering a handler."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        @router.on("test_event", name="my_handler")
        async def handler(msg, conv):
            return msg

        router.unregister("test_event", "my_handler")
        handlers = router.get_handlers("test_event")
        
        assert len(handlers) == 0

    @pytest.mark.asyncio
    async def test_event_router_list_events(self):
        """Test listing registered events."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        @router.on("event1")
        async def handler1(msg, conv):
            return msg

        @router.on("event2")
        async def handler2(msg, conv):
            return msg

        events = router.list_events()
        assert "event1" in events
        assert "event2" in events

    @pytest.mark.asyncio
    async def test_event_router_concurrent_handler(self):
        """Test concurrent handler execution."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        handler_called = []

        @router.on("test_event", concurrent=True)
        async def concurrent_handler(msg, conv):
            await asyncio.sleep(0.01)
            handler_called.append(msg)

        msg = LLMMessage(content="test", service=LLMServiceType.OPENAI, conversation_id="conv_123")
        results = await router.emit("test_event", msg)

        # Concurrent handlers don't return results immediately
        assert len(results) == 0

        # Wait for concurrent handlers
        await router.wait_for_concurrent_handlers(timeout=1.0)
        assert len(handler_called) == 1

    @pytest.mark.asyncio
    async def test_event_router_cleanup(self):
        """Test router cleanup."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        await router.cleanup()
        # Should not raise

    def test_event_router_repr(self):
        """Test router repr."""
        task_manager = TaskManager()
        loop = asyncio.get_event_loop()
        task_manager.setup(TaskManagerParams(loop=loop))

        processor = DummyProcessor(name="test")
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], task_manager=task_manager
        )
        router = conversation.router

        repr_str = repr(router)
        assert "EventRouter" in repr_str

