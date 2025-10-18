"""
tinychat Quickstart Example
===========================

This example demonstrates the core concepts of tinychat:
- Message processors that handle different types of messages
- Event-driven routing using the EventRouter
- Conversation state management
- Observer pattern for monitoring

We use mock/dummy processors to keep the example simple and illustrative.
"""

import asyncio
from typing import Optional

from tinychat.conversations.conversation import Conversation
from tinychat.events.router import EventRouter
from tinychat.messages.messages import UserMessage, AIMessage, Message
from tinychat.messages.models import LLMServiceType
from tinychat.observers.observer import Observer, MessageReceived, MessageProcessed, StateChanged
from tinychat.processors.message_processor import MessageProcessor


# ============================================================================
# Step 1: Define custom message processors
# ============================================================================

class HistoryProcessor(MessageProcessor):
    """Stores conversation history and enriches messages with context."""
    
    def __init__(self):
        super().__init__(name="history")
        self.messages = []
    
    async def _process(self, message: Message) -> Optional[Message]:
        """Add message to history and pass it along."""
        if isinstance(message, (UserMessage, AIMessage)):
            self.messages.append(message)
            print(f"📚 [History] Stored message #{len(self.messages)}: {message.content[:50]}...")
        return message


class AnalyzerProcessor(MessageProcessor):
    """Analyzes user messages for intent and sentiment."""
    
    def __init__(self):
        super().__init__(name="analyzer")
    
    async def _process(self, message: Message) -> Optional[Message]:
        """Analyze the message (mock implementation)."""
        if isinstance(message, UserMessage):
            # Mock analysis
            content_lower = message.content.lower()
            if "hello" in content_lower or "hi" in content_lower:
                intent = "greeting"
            elif "?" in message.content:
                intent = "question"
            else:
                intent = "statement"
            
            # Store analysis in message metadata
            if message.metadata is None:
                message.metadata = {}
            message.metadata["intent"] = intent
            
            print(f"🔍 [Analyzer] Detected intent: {intent}")
        
        return message


class AIAssistantProcessor(MessageProcessor):
    """Generates AI responses (mocked for this example)."""
    
    def __init__(self):
        super().__init__(name="ai_assistant")
    
    async def _process(self, message: Message) -> Optional[Message]:
        """Generate a response based on user message."""
        if isinstance(message, UserMessage):
            # Mock AI response generation
            intent = message.metadata.get("intent", "unknown") if message.metadata else "unknown"
            
            if intent == "greeting":
                response_text = f"Hello! How can I help you today?"
            elif intent == "question":
                response_text = f"That's a great question about '{message.content[:30]}...'. Let me help!"
            else:
                response_text = f"I understand. You mentioned: '{message.content[:30]}...'"
            
            # Create AI response
            ai_message = AIMessage(
                content=response_text,
                service=LLMServiceType.OPENAI,
                conversation_id=message.conversation_id,
                agent_id="assistant-001"
            )
            
            print(f"🤖 [AI] Generated response: {ai_message.content}")
            
            # Store in history
            await self.call_processor("history", ai_message)
            
            return ai_message
        
        return message


# ============================================================================
# Step 2: Create a custom observer for monitoring
# ============================================================================

class LoggingObserver(Observer):
    """Observes and logs conversation events."""
    
    async def on_message_received(self, data: MessageReceived):
        print(f"📥 [Observer] Processor '{data.processor.name}' received message: {data.message.id}")
    
    async def on_message_processed(self, data: MessageProcessed):
        duration_ms = data.duration_ns / 1_000_000
        print(f"✅ [Observer] Processor '{data.processor.name}' processed in {duration_ms:.2f}ms")
    
    async def on_state_changed(self, data: StateChanged):
        print(f"🔄 [Observer] State changed: {data.previous_state} → {data.new_state}")


# ============================================================================
# Step 3: Set up conversation with event routing
# ============================================================================

async def setup_conversation() -> Conversation:
    """Create and configure a conversation with processors and routing."""
    
    # Create processors
    history = HistoryProcessor()
    analyzer = AnalyzerProcessor()
    assistant = AIAssistantProcessor()
    
    # Create conversation
    conversation = Conversation(
        conversation_id="quickstart-demo",
        processors=[history, analyzer, assistant],
        observers=[LoggingObserver()]
    )
    
    # Set up the conversation (must be done in async context)
    loop = asyncio.get_event_loop()
    await conversation.setup(loop)
    
    # Configure event routing
    router = conversation.router
    
    # When a UserMessage arrives, route it through our pipeline
    @router.on(UserMessage)
    async def handle_user_message(message: Message, conv: Conversation):
        """Route user messages through the processing pipeline."""
        print(f"\n🎯 [Router] Handling user message: {message.content}\n")
        
        # Step 1: Store in history
        msg = await conv.route_to("history", message)
        
        # Step 2: Analyze the message
        msg = await conv.route_to("analyzer", msg)
        
        # Step 3: Generate AI response
        result = await conv.route_to("ai_assistant", msg)
        
        return result
    
    print("✨ Conversation setup complete!\n")
    return conversation


# ============================================================================
# Step 4: Run the example
# ============================================================================

async def main():
    """Main execution function."""
    
    print("=" * 70)
    print("🚀 tinychat Quickstart Example")
    print("=" * 70)
    print()
    
    # Set up conversation
    conversation = await setup_conversation()
    
    # Simulate some user interactions
    test_messages = [
        "Hello there!",
        "What is the meaning of life?",
        "I love using tinychat framework."
    ]
    
    try:
        for user_text in test_messages:
            print("─" * 70)
            print(f"💬 User: {user_text}")
            print("─" * 70)
            
            # Create user message
            message = UserMessage(
                content=user_text,
                service=LLMServiceType.OPENAI,
                conversation_id=conversation.conversation_id,
                user_id="user-123"
            )
            
            # Emit the message to the router (triggers the handler)
            results = await conversation.router.emit(message)
            
            # Wait a bit between messages for readability
            await asyncio.sleep(0.5)
            print()
    
    finally:
        # Clean up
        print("─" * 70)
        print("🧹 Cleaning up conversation...")
        await conversation.cleanup()
        print("✅ Done!")
        print("=" * 70)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

