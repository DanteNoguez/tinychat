"""Tests for message classes."""
import pytest

from tinychat.messages.messages import (
    Message,
    UserMessage,
    AIMessage,
    SystemMessage,
    ErrorMessage,
    EventMessage,
)
from tinychat.messages.models import LLMServiceType


class TestMessage:
    """Test base Message class."""

    def test_message_instantiation(self):
        """Test that Message can be instantiated."""
        msg = Message()
        assert msg.id is not None
        assert msg.timestamp is not None
        assert msg.metadata is None

    def test_message_with_metadata(self):
        """Test Message with metadata."""
        metadata = {"key": "value"}
        msg = Message()
        msg.metadata = metadata
        assert msg.metadata == metadata

    def test_message_name_updated(self):
        """Test that message name is updated with id."""
        msg = Message()
        assert msg.name.startswith("Message#")


class TestUserMessage:
    """Test UserMessage class."""

    def test_user_message_instantiation(self):
        """Test that UserMessage can be instantiated."""
        msg = UserMessage(
            conversation_id="conv_123",
            content="Hello",
            user_id="user_1",
            service=LLMServiceType.OPENAI,
        )
        assert msg.content == "Hello"
        assert msg.user_id == "user_1"
        assert msg.service == LLMServiceType.OPENAI
        assert msg.conversation_id == "conv_123"


class TestAIMessage:
    """Test AIMessage class."""

    def test_ai_message_instantiation(self):
        """Test that AIMessage can be instantiated."""
        msg = AIMessage(
            conversation_id="conv_123",
            content="Hi there",
            agent_id="agent_1",
            service=LLMServiceType.OPENAI,
        )
        assert msg.content == "Hi there"
        assert msg.agent_id == "agent_1"
        assert msg.service == LLMServiceType.OPENAI
        assert msg.tool_calls is None
        assert msg.reasoning is None

    def test_ai_message_with_tool_calls(self):
        """Test AIMessage with tool calls."""
        msg = AIMessage(
            conversation_id="conv_123",
            content="",
            agent_id="agent_1",
            service=LLMServiceType.OPENAI,
        )
        msg.tool_calls = [{"name": "get_weather", "args": {"city": "NYC"}}]
        assert len(msg.tool_calls) == 1

    def test_ai_message_with_reasoning(self):
        """Test AIMessage with reasoning."""
        msg = AIMessage(
            conversation_id="conv_123",
            content="",
            agent_id="agent_1",
            service=LLMServiceType.ANTHROPIC,
        )
        msg.reasoning = "I need to check the weather first"
        assert msg.reasoning is not None


class TestSystemMessage:
    """Test SystemMessage class."""

    def test_system_message_instantiation(self):
        """Test that SystemMessage can be instantiated."""
        msg = SystemMessage(
            conversation_id="conv_123",
            content="You are a helpful assistant",
            service=LLMServiceType.OPENAI,
        )
        assert msg.content == "You are a helpful assistant"
        assert msg.service == LLMServiceType.OPENAI


class TestErrorMessage:
    """Test ErrorMessage class."""

    def test_error_message_instantiation(self):
        """Test that ErrorMessage can be instantiated."""
        msg = ErrorMessage(
            content="Something went wrong",
            conversation_id="conv_123",
            source="processor_1",
            fatal=False,
        )
        assert msg.content == "Something went wrong"
        assert msg.source == "processor_1"
        assert msg.fatal is False

    def test_error_message_fatal(self):
        """Test ErrorMessage with fatal flag."""
        msg = ErrorMessage(
            conversation_id="conv_123",
            content="Critical error",
            source="processor_1",
            fatal=True,
        )
        assert msg.fatal is True


class TestEventMessage:
    """Test EventMessage class."""

    def test_event_message_instantiation(self):
        """Test that EventMessage can be instantiated."""
        msg = EventMessage(
            conversation_id="conv_123",
            event_name="needs_refresh",
            content={"reason": "stale_data"},
        )
        assert msg.event_name == "needs_refresh"
        assert msg.content == {"reason": "stale_data"}
        assert msg.source is None
        assert msg.destination is None

    def test_event_message_with_routing(self):
        """Test EventMessage with source and destination."""
        msg = EventMessage(
            conversation_id="conv_123",
            event_name="data_ready",
            source="processor_a",
            destination="processor_b",
        )
        assert msg.source == "processor_a"
        assert msg.destination == "processor_b"

