"""Tests for state management classes."""
import pytest

from tinychat.state.state import AgentState, ConversationState, StateEntry


class TestStateEntry:
    """Test StateEntry dataclass."""

    def test_state_entry_creation(self):
        """Test that StateEntry can be created."""
        entry = StateEntry(
            name="test_field", value="test_value", timestamp=1234567890, metadata={"key": "value"}
        )
        assert entry.name == "test_field"
        assert entry.value == "test_value"
        assert entry.timestamp == 1234567890
        assert entry.metadata == {"key": "value"}

    def test_state_entry_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        entry = StateEntry(name="test", value="value", timestamp=123)
        assert entry.metadata == {}

    def test_state_entry_repr(self):
        """Test repr method."""
        entry = StateEntry(name="phase", value="active", timestamp=123456)
        repr_str = repr(entry)
        assert "phase" in repr_str
        assert "active" in repr_str


class TestAgentState:
    """Test AgentState class."""

    def test_agent_state_instantiation(self):
        """Test that AgentState can be instantiated."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        assert state.conversation_id == "conv_123"
        assert state.agent_id == "agent_1"
        assert state.data == {}
        assert state.history == []

    @pytest.mark.asyncio
    async def test_agent_state_set_get(self):
        """Test set and get operations."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        await state.set("key1", "value1")
        assert state.get("key1") == "value1"
        assert state.get("nonexistent", "default") == "default"

    @pytest.mark.asyncio
    async def test_agent_state_set_records_history(self):
        """Test that set records changes in history."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        await state.set("key1", "value1")
        
        history = state.history
        assert len(history) == 1
        assert history[0].name == "key1"
        assert history[0].value == "value1"
        assert isinstance(history[0].timestamp, int)

    @pytest.mark.asyncio
    async def test_agent_state_set_with_metadata(self):
        """Test set with metadata."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        await state.set("key1", "value1", metadata={"source": "processor_1"})
        
        entry = state.history[0]
        assert entry.metadata["source"] == "processor_1"

    @pytest.mark.asyncio
    async def test_agent_state_set_tracks_previous_value(self):
        """Test that updating a value tracks the previous value."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        await state.set("key1", "value1")
        await state.set("key1", "value2")
        
        history = state.history
        assert len(history) == 2
        assert history[1].metadata.get("previous_value") == "value1"

    def test_agent_state_has(self):
        """Test has method."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        # Use asyncio.run for async methods in sync tests
        import asyncio
        asyncio.run(state.set("key1", "value1"))
        assert state.has("key1") is True
        assert state.has("key2") is False

    @pytest.mark.asyncio
    async def test_agent_state_delete(self):
        """Test delete method."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        await state.set("key1", "value1")
        await state.delete("key1")
        
        assert state.has("key1") is False
        # Check deletion is recorded in history
        history = state.history
        assert len(history) == 2  # set + delete
        assert history[1].name == "key1"
        assert history[1].value is None
        assert history[1].metadata.get("deleted_value") == "value1"

    @pytest.mark.asyncio
    async def test_agent_state_update(self):
        """Test bulk update method."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        await state.update({"key1": "value1", "key2": "value2"})
        
        assert state.get("key1") == "value1"
        assert state.get("key2") == "value2"
        # Each key should have a history entry
        assert len(state.history) == 2

    def test_agent_state_clear(self):
        """Test clear method."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        import asyncio
        asyncio.run(state.update({"key1": "value1", "key2": "value2"}))
        state.clear()
        assert state.data == {}
        # History should still be preserved
        assert len(state.history) == 2

    def test_agent_state_data_property(self):
        """Test data property returns a copy."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        import asyncio
        asyncio.run(state.set("key1", "value1"))
        
        data = state.data
        data["key2"] = "value2"  # Modify the returned dict
        
        # Original state should be unchanged
        assert state.has("key2") is False

    def test_agent_state_repr(self):
        """Test repr method."""
        state = AgentState(conversation_id="conv_123", agent_id="agent_1")
        import asyncio
        asyncio.run(state.set("key1", "value1"))
        
        repr_str = repr(state)
        assert "agent_1" in repr_str
        assert "key1" in repr_str


class TestConversationState:
    """Test ConversationState class."""

    def test_conversation_state_instantiation(self):
        """Test that ConversationState can be instantiated."""
        state = ConversationState(conversation_id="conv_123")
        assert state.conversation_id == "conv_123"
        assert state.current_processor is None
        assert state.phase == "idle"
        assert state.history == []

    def test_conversation_state_custom_initial_phase(self):
        """Test instantiation with custom initial phase."""
        state = ConversationState(conversation_id="conv_123", initial_phase="starting")
        assert state.phase == "starting"

    def test_conversation_state_update_processor(self):
        """Test updating processor."""
        state = ConversationState(conversation_id="conv_123")
        state.update_processor("processor_1", metadata={"msg_id": "123"})
        
        assert state.current_processor == "processor_1"
        assert len(state.history) == 1
        
        entry = state.history[0]
        assert entry.name == "processor"
        assert entry.value == "processor_1"
        assert entry.metadata.get("msg_id") == "123"

    @pytest.mark.asyncio
    async def test_conversation_state_update_phase(self):
        """Test updating phase."""
        state = ConversationState(conversation_id="conv_123")
        await state.update_phase("processing", metadata={"step": "1"})
        
        assert state.phase == "processing"
        assert len(state.history) == 1
        
        entry = state.history[0]
        assert entry.name == "phase"
        assert entry.value == "processing"
        assert entry.metadata.get("step") == "1"

    @pytest.mark.asyncio
    async def test_conversation_state_transition_callback(self):
        """Test state transition callback."""
        state = ConversationState(conversation_id="conv_123")
        callback_called = []

        async def callback(entry: StateEntry):
            callback_called.append(entry)

        state.add_transition_callback(callback)
        await state.update_phase("processing")
        
        assert len(callback_called) == 1
        assert callback_called[0].name == "phase"
        assert callback_called[0].value == "processing"

    @pytest.mark.asyncio
    async def test_conversation_state_multiple_callbacks(self):
        """Test multiple transition callbacks."""
        state = ConversationState(conversation_id="conv_123")
        callback1_called = []
        callback2_called = []

        async def callback1(entry: StateEntry):
            callback1_called.append(entry)

        async def callback2(entry: StateEntry):
            callback2_called.append(entry)

        state.add_transition_callback(callback1)
        state.add_transition_callback(callback2)
        await state.update_phase("processing")
        
        assert len(callback1_called) == 1
        assert len(callback2_called) == 1

    @pytest.mark.asyncio
    async def test_conversation_state_remove_callback(self):
        """Test removing a transition callback."""
        state = ConversationState(conversation_id="conv_123")
        callback_called = []

        async def callback(entry: StateEntry):
            callback_called.append(entry)

        state.add_transition_callback(callback)
        state.remove_transition_callback(callback)
        await state.update_phase("processing")
        
        assert len(callback_called) == 0

    def test_conversation_state_history(self):
        """Test history tracking."""
        state = ConversationState(conversation_id="conv_123")
        state.update_processor("processor_1")
        state.update_processor("processor_2")
        
        history = state.history
        assert len(history) == 2
        assert history[0].value == "processor_1"
        assert history[1].value == "processor_2"

    def test_conversation_state_get_history_by_name(self):
        """Test filtering history by field name."""
        state = ConversationState(conversation_id="conv_123")
        import asyncio
        
        state.update_processor("processor_1")
        asyncio.run(state.update_phase("phase_1"))
        state.update_processor("processor_2")
        asyncio.run(state.update_phase("phase_2"))
        
        processor_history = state.get_history_by_name("processor")
        phase_history = state.get_history_by_name("phase")
        
        assert len(processor_history) == 2
        assert len(phase_history) == 2
        assert all(e.name == "processor" for e in processor_history)
        assert all(e.name == "phase" for e in phase_history)

    def test_conversation_state_get_latest_entry(self):
        """Test getting the latest entry for a field."""
        state = ConversationState(conversation_id="conv_123")
        state.update_processor("processor_1")
        state.update_processor("processor_2")
        
        latest = state.get_latest_entry("processor")
        assert latest is not None
        assert latest.value == "processor_2"
        
        # Non-existent field should return None
        assert state.get_latest_entry("nonexistent") is None

    def test_conversation_state_repr(self):
        """Test repr method."""
        state = ConversationState(conversation_id="conv_123")
        state.update_processor("processor_1")
        
        repr_str = repr(state)
        assert "idle" in repr_str
        assert "processor_1" in repr_str
