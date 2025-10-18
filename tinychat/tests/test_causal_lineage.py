"""
Tests for causal lineage tracking and cycle detection.
"""
import pytest
import asyncio
from typing import Optional

from tinychat.messages.messages import Message, UserMessage
from tinychat.messages.models import LLMServiceType
from tinychat.processors import (
    MessageProcessor,
    CompositeProcessor,
    ProcessorSetup,
    CycleDetectedError,
    MaxHopsExceededError,
)
from tinychat.asynchronous.manager import TaskManager, TaskManagerParams


class SimpleProcessor(MessageProcessor):
    """A simple processor that passes messages through."""
    
    async def _process(self, message: Message) -> Optional[Message]:
        # Just return the same message
        return message


class DerivedMessageProcessor(MessageProcessor):
    """A processor that creates derived messages."""
    
    async def _process(self, message: Message) -> Optional[Message]:
        # Create a new message derived from the input
        if isinstance(message, UserMessage):
            return message.derive(
                UserMessage,
                content=f"Processed: {message.content}",
                service=message.service,
                conversation_id=message.conversation_id,
            )
        return message


class CyclicComposite(CompositeProcessor):
    """A composite that routes in a cycle: A → B → A"""
    
    async def _process(self, message: Message) -> Optional[Message]:
        # Start routing at processor A
        return await self.route_to("processor_a", message)


class ProcessorA(MessageProcessor):
    """Routes to B"""
    
    async def _process(self, message: Message) -> Optional[Message]:
        # Route to processor B
        composite = getattr(self, "_composite", None)
        if composite:
            return await composite.route_to("processor_b", message)
        return message


class ProcessorB(MessageProcessor):
    """Routes back to A (creates cycle)"""
    
    async def _process(self, message: Message) -> Optional[Message]:
        # Route back to processor A (cycle!)
        composite = getattr(self, "_composite", None)
        if composite:
            return await composite.route_to("processor_a", message)
        return message


class LongChainComposite(CompositeProcessor):
    """A composite that creates a long chain to test max hops."""
    
    async def _process(self, message: Message) -> Optional[Message]:
        # Keep routing to next processor
        return await self.route_to("proc_0", message)


class ChainProcessor(MessageProcessor):
    """Routes to the next processor in a chain."""
    
    def __init__(self, name: str, next_proc: Optional[str] = None):
        super().__init__(name=name)
        self.next_proc = next_proc
    
    async def _process(self, message: Message) -> Optional[Message]:
        if self.next_proc:
            composite = getattr(self, "_composite", None)
            if composite:
                return await composite.route_to(self.next_proc, message)
        return message


@pytest.mark.asyncio
async def test_causal_lineage_basic():
    """Test basic causal lineage tracking."""
    # Create a message
    msg = UserMessage(
        content="Hello",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Initially has no parent
    assert msg.parent_id is None
    assert msg.processor_path == []
    
    # Derive a new message of same type
    derived = msg.derive(
        UserMessage,
        content="Derived",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Check causal link
    assert derived.parent_id == msg.id
    assert derived.id != msg.id  # Different message
    assert derived.processor_path == []  # Copied empty path


@pytest.mark.asyncio
async def test_processor_path_tracking():
    """Test that processor_path is updated during routing."""
    # Setup processors
    proc = SimpleProcessor(name="test_processor")
    
    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    
    setup = ProcessorSetup(task_manager=task_manager)
    await proc.setup(setup)
    
    # Create composite with the processor
    composite = CompositeProcessor(
        name="test_composite",
        processors=[proc]
    )
    await composite.setup(setup)
    
    # Create a message
    msg = UserMessage(
        content="Test",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Route through the processor
    result = await composite.route_to("test_processor", msg)
    
    # Check that processor was recorded in path
    assert "test_processor" in msg.processor_path
    assert len(msg.processor_path) == 1


@pytest.mark.asyncio
async def test_cycle_detection():
    """Test that routing cycles are detected and raise an error."""
    # Create processors that form a cycle
    proc_a = ProcessorA(name="processor_a")
    proc_b = ProcessorB(name="processor_b")
    
    # Setup
    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    
    composite = CyclicComposite(
        name="cyclic_composite",
        processors=[proc_a, proc_b]
    )
    
    setup = ProcessorSetup(task_manager=task_manager)
    await composite.setup(setup)
    
    # Inject composite reference so processors can route
    proc_a._composite = composite
    proc_b._composite = composite
    
    # Create a message
    msg = UserMessage(
        content="Test",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Attempt to process - should detect cycle
    with pytest.raises(CycleDetectedError) as exc_info:
        await composite.process(msg)
    
    # Verify the cycle was detected correctly
    assert "processor_a" in str(exc_info.value)
    assert exc_info.value.processor_name == "processor_a"
    assert "processor_a" in exc_info.value.processor_path


@pytest.mark.asyncio
async def test_max_hops_exceeded():
    """Test that max_hops limit is enforced."""
    # Create a chain of processors longer than max_hops
    max_hops = 5
    chain_length = 10
    
    # Create chain processors
    processors = []
    for i in range(chain_length):
        next_name = f"proc_{i+1}" if i < chain_length - 1 else None
        proc = ChainProcessor(name=f"proc_{i}", next_proc=next_name)
        processors.append(proc)
    
    # Setup
    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    
    composite = LongChainComposite(
        name="chain_composite",
        processors=processors,
        max_hops=max_hops
    )
    
    setup = ProcessorSetup(task_manager=task_manager)
    await composite.setup(setup)
    
    # Inject composite reference
    for proc in processors:
        proc._composite = composite
    
    # Create a message
    msg = UserMessage(
        content="Test",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Attempt to process - should exceed max hops
    with pytest.raises(MaxHopsExceededError) as exc_info:
        await composite.process(msg)
    
    # Verify the error
    assert exc_info.value.max_hops == max_hops
    assert exc_info.value.current_hops == max_hops


@pytest.mark.asyncio
async def test_derived_message_preserves_path():
    """Test that deriving a message preserves processor_path."""
    # Create a message
    msg = UserMessage(
        content="Hello",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Simulate some routing by adding to path
    msg.processor_path.append("proc_1")
    msg.processor_path.append("proc_2")
    
    # Derive a new message
    derived = msg.derive(
        UserMessage,
        content="Derived",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Check that path is preserved
    assert derived.processor_path == ["proc_1", "proc_2"]
    assert derived.parent_id == msg.id


@pytest.mark.asyncio
async def test_cross_type_derivation():
    """Test deriving a message of a different type (UserMessage → AIMessage)."""
    from tinychat.messages.messages import AIMessage
    
    # Create a user message
    user_msg = UserMessage(
        content="What is the capital of France?",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1",
        user_id="user_123"
    )
    
    # Add processor path
    user_msg.processor_path.append("ingress_processor")
    user_msg.processor_path.append("history_processor")
    
    # Derive an AI response from it
    ai_msg = user_msg.derive(
        AIMessage,
        content="The capital of France is Paris.",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1",
        agent_id="agent_1"
    )
    
    # Check type transformation
    assert isinstance(ai_msg, AIMessage)
    assert not isinstance(ai_msg, UserMessage)
    
    # Check causal link
    assert ai_msg.parent_id == user_msg.id
    assert ai_msg.processor_path == ["ingress_processor", "history_processor"]
    
    # Check content is different
    assert ai_msg.content != user_msg.content
    assert ai_msg.content == "The capital of France is Paris."


@pytest.mark.asyncio  
async def test_no_cycle_with_different_paths():
    """Test that visiting different processors doesn't trigger false positives."""
    # Create processors
    proc_a = SimpleProcessor(name="processor_a")
    proc_b = SimpleProcessor(name="processor_b")
    proc_c = SimpleProcessor(name="processor_c")
    
    # Setup
    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    
    composite = CompositeProcessor(
        name="test_composite",
        processors=[proc_a, proc_b, proc_c]
    )
    
    setup = ProcessorSetup(task_manager=task_manager)
    await composite.setup(setup)
    
    # Create a message
    msg = UserMessage(
        content="Test",
        service=LLMServiceType.OPENAI,
        conversation_id="conv_1"
    )
    
    # Route through different processors - should not raise
    await composite.route_to("processor_a", msg)
    await composite.route_to("processor_b", msg)
    await composite.route_to("processor_c", msg)
    
    # Verify path
    assert msg.processor_path == ["processor_a", "processor_b", "processor_c"]
    
    # Now trying to revisit processor_a should raise
    with pytest.raises(CycleDetectedError):
        await composite.route_to("processor_a", msg)

