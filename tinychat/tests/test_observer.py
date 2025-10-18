"""Tests for observer classes."""
import pytest

from tinychat.observers.observer import (
    Observer,
    MessageReceived,
    MessageProcessed,
    ProcessorCalled,
    StateChanged,
)
from tinychat.messages.messages import Message


class TestObserver:
    """Test Observer class."""

    def test_observer_instantiation(self):
        """Test that Observer can be instantiated."""
        observer = Observer()
        assert observer is not None

    @pytest.mark.asyncio
    async def test_observer_on_message_received(self):
        """Test on_message_received method."""
        observer = Observer()
        msg = Message()
        data = MessageReceived(processor=None, message=msg, timestamp=123456)
        
        # Should not raise
        await observer.on_message_received(data)

    @pytest.mark.asyncio
    async def test_observer_on_message_processed(self):
        """Test on_message_processed method."""
        observer = Observer()
        msg = Message()
        data = MessageProcessed(
            processor=None, message=msg, result=None, timestamp=123456, duration_ns=1000
        )
        
        # Should not raise
        await observer.on_message_processed(data)

    @pytest.mark.asyncio
    async def test_observer_on_processor_called(self):
        """Test on_processor_called method."""
        observer = Observer()
        msg = Message()
        data = ProcessorCalled(
            source=None, target=None, message=msg, timestamp=123456
        )
        
        # Should not raise
        await observer.on_processor_called(data)

    @pytest.mark.asyncio
    async def test_observer_on_state_changed(self):
        """Test on_state_changed method."""
        observer = Observer()
        data = StateChanged(
            conversation_id="conv_123",
            previous_state="idle",
            new_state="processing",
            timestamp=123456,
            metadata={},
        )
        
        # Should not raise
        await observer.on_state_changed(data)


class TestCustomObserver:
    """Test custom observer implementation."""

    @pytest.mark.asyncio
    async def test_custom_observer(self):
        """Test that we can extend Observer."""
        events = []

        class CustomObserver(Observer):
            async def on_message_received(self, data: MessageReceived):
                events.append("received")

            async def on_message_processed(self, data: MessageProcessed):
                events.append("processed")

            async def on_processor_called(self, data: ProcessorCalled):
                events.append("called")

            async def on_state_changed(self, data: StateChanged):
                events.append("state_changed")

        observer = CustomObserver()
        msg = Message()

        await observer.on_message_received(
            MessageReceived(processor=None, message=msg, timestamp=123456)
        )
        await observer.on_message_processed(
            MessageProcessed(
                processor=None, message=msg, result=None, timestamp=123456, duration_ns=1000
            )
        )
        await observer.on_processor_called(
            ProcessorCalled(source=None, target=None, message=msg, timestamp=123456)
        )
        await observer.on_state_changed(
            StateChanged(
                conversation_id="conv_123",
                previous_state="idle",
                new_state="processing",
                timestamp=123456,
                metadata={},
            )
        )

        assert events == ["received", "processed", "called", "state_changed"]

