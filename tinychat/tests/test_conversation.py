"""Tests for Conversation class."""
import asyncio
import pytest

from tinychat.conversations.conversation import Conversation
from tinychat.messages.messages import Message, LLMMessage
from tinychat.messages.models import LLMServiceType
from tinychat.processors.message_processor import MessageProcessor
from tinychat.asynchronous.manager import TaskManager, TaskManagerParams
from tinychat.observers.observer import Observer


class EchoProcessor(MessageProcessor):
    """Simple echo processor for testing."""

    async def _process(self, message: Message) -> Message:
        return message


class TestConversation:
    """Test Conversation class."""

    def test_conversation_instantiation(self):
        """Test that Conversation can be instantiated."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        assert conversation.conversation_id == "conv_123"
        assert len(conversation.processors) == 1
        assert "echo" in conversation.processors

    def test_conversation_with_custom_task_manager(self):
        """Test conversation with custom task manager."""
        processor = EchoProcessor(name="echo")
        task_manager = TaskManager()
        conversation = Conversation(
            conversation_id="conv_123",
            processors=[processor],
            task_manager=task_manager,
        )

        assert conversation.task_manager == task_manager

    def test_conversation_properties(self):
        """Test conversation properties."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        assert conversation.state is not None
        assert conversation.agent_state is not None
        assert conversation.router is not None
        assert isinstance(conversation.observers, list)

    @pytest.mark.asyncio
    async def test_conversation_setup(self):
        """Test conversation setup."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        loop = asyncio.get_event_loop()
        await conversation.setup(loop)

        assert conversation._setup_complete is True

    @pytest.mark.asyncio
    async def test_conversation_cleanup(self):
        """Test conversation cleanup."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        loop = asyncio.get_event_loop()
        await conversation.setup(loop)
        await conversation.cleanup()

        assert conversation._setup_complete is False

    @pytest.mark.asyncio
    async def test_conversation_route_to(self):
        """Test routing message to processor."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        loop = asyncio.get_event_loop()
        await conversation.setup(loop)

        msg = LLMMessage(content="test", service=LLMServiceType.OPENAI, conversation_id="conv_123")
        result = await conversation.route_to("echo", msg)

        assert result is not None
        assert result.id == msg.id

    def test_conversation_get_processor(self):
        """Test getting a processor by name."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        proc = conversation.get_processor("echo")
        assert proc == processor

    def test_conversation_get_nonexistent_processor(self):
        """Test getting nonexistent processor raises KeyError."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        with pytest.raises(KeyError, match="Processor nonexistent not found"):
            conversation.get_processor("nonexistent")

    def test_conversation_add_observer(self):
        """Test adding an observer."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        observer = Observer()
        conversation.add_observer(observer)

        assert observer in conversation.observers

    def test_conversation_remove_observer(self):
        """Test removing an observer."""
        processor = EchoProcessor(name="echo")
        observer = Observer()
        conversation = Conversation(
            conversation_id="conv_123", processors=[processor], observers=[observer]
        )

        conversation.remove_observer(observer)
        assert observer not in conversation.observers

    def test_conversation_str(self):
        """Test string representation."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        assert str(conversation) == "Conversation(conv_123)"

    @pytest.mark.asyncio
    async def test_conversation_with_multiple_processors(self):
        """Test conversation with multiple processors."""
        proc1 = EchoProcessor(name="proc1")
        proc2 = EchoProcessor(name="proc2")
        conversation = Conversation(
            conversation_id="conv_123", processors=[proc1, proc2]
        )

        loop = asyncio.get_event_loop()
        await conversation.setup(loop)

        assert len(conversation.processors) == 2
        assert "proc1" in conversation.processors
        assert "proc2" in conversation.processors

    @pytest.mark.asyncio
    async def test_conversation_state_updates(self):
        """Test that routing updates state."""
        processor = EchoProcessor(name="echo")
        conversation = Conversation(conversation_id="conv_123", processors=[processor])

        loop = asyncio.get_event_loop()
        await conversation.setup(loop)

        msg = LLMMessage(content="test", service=LLMServiceType.OPENAI, conversation_id="conv_123")
        await conversation.route_to("echo", msg)

        assert conversation.state.current_processor == "echo"

