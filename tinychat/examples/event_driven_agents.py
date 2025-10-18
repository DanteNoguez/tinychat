"""
tinychat Event-Driven Agent Orchestration Example
==================================================

This example demonstrates advanced patterns in tinychat:
- Complex multi-agent orchestration with event-driven routing
- Dynamic flow control based on user intent
- Integration with external systems (DB, CRM, Human API)
- Tool calling and error handling
- Human-in-the-loop workflows
- State management across processors
- Custom event emission for coordination

This replaces hardcoded routing logic with a flexible event-based system
where processors emit events that trigger other processors dynamically.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tinychat.conversations.conversation import Conversation
from tinychat.messages.messages import (
    AIMessage,
    ErrorMessage,
    IngressMessage,
    LLMMessage,
    SystemMessage,
    Message,
    UserMessage,
)
from tinychat.messages.models import LLMServiceType
from tinychat.services.openai.llm.openai_agent import OpenAIAgent
from tinychat.services.openai.llm.models import OpenAIAgentConfig
from tinychat.observers.observer import (
    MessageProcessed,
    Observer,
    StateChanged,
)
from tinychat.processors.message_processor import MessageProcessor


# ============================================================================
# Dummy External Systems (mock implementations)
# ============================================================================


class DummyDB:
    """Simulates a conversation history database."""

    async def get_history(self, conversation_id: str) -> List[LLMMessage]:
        return [
            SystemMessage(conversation_id=conversation_id, content="You are a helpful assistant."),
            UserMessage(conversation_id=conversation_id, content="Hi, I need to schedule an appointment"),
            AIMessage(conversation_id=conversation_id, content="I'd be happy to help! What type of appointment?"),
        ]

    async def save_message(self, conversation_id: str, message: LLMMessage):
        """Save a message to the history."""
        if isinstance(message, UserMessage):
            print(f"[DB] Saved user message for {conversation_id}: {message.content[:50]}...")
        elif isinstance(message, AIMessage):
            print(f"[DB] Saved AI message for {conversation_id}: {message.content[:50]}...")
        elif isinstance(message, SystemMessage):
            print(f"[DB] Saved system message for {conversation_id}: {message.content[:50]}...")

class DummyCRM:
    """Simulates a Customer Relationship Management system."""

    async def get_user_info(self, user_id: str) -> Dict:
        """Retrieve user profile information."""
        return {
            "name": "John Doe",
            "phone": "+1234567890",
            "email": "john@example.com",
        }

    async def get_appointments(self, user_id: str) -> List[Dict]:
        """Retrieve user's scheduled appointments."""
        return [{"date": "2025-01-15", "time": "10:00", "type": "Consultation"}]

    async def schedule_appointment(
        self, user_id: str, date: str, time: str, appointment_type: str
    ) -> Dict:
        """Schedule a new appointment."""
        print(
            f"[CRM] Scheduling {appointment_type} for {user_id} on {date} at {time}"
        )
        return {"id": "apt_123", "status": "scheduled"}


class DummyHumanAPI:
    """Simulates an API for human oversight/intervention."""

    async def create_task(self, error: ErrorMessage) -> str:
        """Create a task for human review."""
        task_id = f"task_{error.id[:8]}"
        print(f"[HUMAN API] Created task {task_id} for error: {error.content}")
        return task_id

    async def notify_resolution(self, conversation_id: str, resolution: str):
        """Notify that a human has resolved an issue."""
        print(f"[HUMAN API] Notifying conversation {conversation_id}: {resolution}")


class DummyLLM(OpenAIAgent):
    """Simulates an LLM service."""

    def __init__(self):
        super().__init__(
            OpenAIAgentConfig(
                prompt="You are a helpful assistant",
                api_key="test-key",
            )
        )

    async def chat(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> Dict:
        """Generate a chat completion."""
        await asyncio.sleep(0.1)  # Simulate API latency

        if tools:
            # If tools are available, simulate a tool call
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "schedule_appointment",
                        "args": {
                            "date": "2025-01-20",
                            "time": "14:00",
                            "type": "Follow-up",
                        },
                    }
                ],
            }

        # Otherwise return a regular response
        return {
            "role": "assistant",
            "content": "I've scheduled your follow-up appointment for January 20th at 2 PM.",
        }


# ============================================================================
# Message Processors (the core business logic)
# ============================================================================


class ConversationHistoryProcessor(MessageProcessor):
    """
    Manages conversation history storage and retrieval.
    
    Responsibilities:
    - Load previous conversation history for context
    - Save new messages (both user and AI)
    - Transform UserMessage → ContextMessage with history
    """

    def __init__(self, db: DummyDB):
        super().__init__(name="history")
        self._db = db

    async def _process(self, message: Message) -> Optional[List[LLMMessage]]:
        """Process incoming messages to manage history."""
        if isinstance(message, UserMessage):
            conversation_id = message.conversation_id
            await self._db.save_message(conversation_id, message)
            history = await self._db.get_history(conversation_id)
            return history

        elif isinstance(message, AIMessage):
            conversation_id = message.conversation_id
            await self._db.save_message(conversation_id, message)
            history = await self._db.get_history(conversation_id)
            return history

        elif isinstance(message, IngressMessage):
            conversation_id = message.conversation_id
            await self._db.save_message(conversation_id, message)
            history = await self._db.get_history(conversation_id)
            return history

        return None


class CRMProcessor(MessageProcessor):
    """
    Integrates with CRM system to enrich messages with user data.
    
    Responsibilities:
    - Fetch user profile and appointments
    - Provide tools for CRM operations (scheduling, etc.)
    - Emit events when data is refreshed
    """

    def __init__(self, crm: DummyCRM):
        super().__init__(name="crm")
        self._crm = crm

    async def _process(self, message: IngressMessage) -> Optional[Message]:
        """Enrich ContextMessage with CRM data."""
        user_id = message.user_id
        user_info = await self._crm.get_user_info(user_id)
        appointments = await self._crm.get_appointments(user_id)

        # Emit event when data is refreshed
        await self.emit_event("crm_data_refreshed", message, payload={"user_id": user_id})

        return SystemMessage(
            content=f"User {user_id} has the following appointments: {appointments}",
            conversation_id=message.conversation_id,
            service=LLMServiceType.OPENAI,
            agent_id=self.name,
            metadata={"user_info": user_info},
        )

    def get_tools(self) -> List[Dict]:
        """Define available CRM tools."""
        return [
            {
                "name": "schedule_appointment",
                "description": "Schedule a new appointment for the user",
                "parameters": {
                    "date": "string (YYYY-MM-DD)",
                    "time": "string (HH:MM)",
                    "type": "string (appointment type)",
                },
            },
            {
                "name": "get_appointments",
                "description": "Get user's current appointments",
                "parameters": {},
            },
        ]

    async def execute_tool(self, tool_name: str, args: Dict, user_id: str) -> Dict:
        """Execute a CRM tool."""
        if tool_name == "schedule_appointment":
            return await self._crm.schedule_appointment(
                user_id, args["date"], args["time"], args["type"]
            )
        elif tool_name == "get_appointments":
            appointments = await self._crm.get_appointments(user_id)
            return {"appointments": appointments}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


class SchedulingSubAgent(MessageProcessor):
    """
    Specialized sub-agent that handles appointment scheduling.
    
    Responsibilities:
    - Process scheduling-related requests
    - Execute tool calls via CRM
    - Request data refresh after making changes
    """

    def __init__(self, llm: DummyLLM, crm_processor: CRMProcessor):
        super().__init__(name="scheduling_subagent")
        self._llm = llm
        self._crm = crm_processor

    async def _process(self, message: Message) -> Optional[Message]:
        """Handle scheduling requests."""
        if not isinstance(message, ContextMessage):
            return None

        print(f"[{self.name}] Processing scheduling request...")

        # Update conversation agent state
        await self.conversation.agent_state.set("current_agent", self.name)
        await self.conversation.agent_state.set("agent_task", "scheduling")

        # Prepare LLM messages with tools
        tools = self._crm.get_tools()
        llm_messages = [
            {"role": "system", "content": "You are a scheduling assistant."},
            *message.history,
        ]

        # Get LLM response (might include tool calls)
        response = await self._llm.chat(llm_messages, tools=tools)

        if response.get("tool_calls"):
            tool_call = response["tool_calls"][0]
            user_id = message.metadata.get("user_id") if message.metadata else None

            if not user_id:
                raise ValueError("User ID required for tool execution")

            await self.conversation.agent_state.set("tool_called", tool_call["name"])

            try:
                # Execute the tool
                result = await self._crm.execute_tool(
                    tool_call["name"], tool_call["args"], user_id
                )

                await self.conversation.agent_state.set("tool_result", result)

                # Emit event to trigger CRM data refresh
                await self.emit_event(
                    "needs_data_refresh", message, payload={"user_id": user_id}
                )

                # Get final response from LLM
                final_response = await self._llm.chat(
                    llm_messages
                    + [
                        {"role": "assistant", "content": f"Tool result: {result}"},
                    ]
                )

                # Create AI message with the response
                ai_message = AIMessage(
                    conversation_id=message.conversation_id,
                    content=final_response["content"],
                    service=LLMServiceType.OPENAI,
                    agent_id=self.name,
                    tool_calls=[tool_call],
                )
                if message.metadata:
                    ai_message.metadata = message.metadata.copy()
                    ai_message.metadata["from_processor"] = self.name

                return ai_message

            except Exception as e:
                raise Exception(f"Tool execution failed: {e}")

        return None


class OrchestratorAgent(MessageProcessor):
    """
    Main orchestrator that analyzes intent and delegates to sub-agents.
    
    Responsibilities:
    - Analyze user intent from enriched context
    - Emit events to trigger appropriate sub-agents
    - Handle general queries directly
    """

    def __init__(self, llm: DummyLLM):
        super().__init__(name="orchestrator")
        self._llm = llm

    async def _process(self, message: Message) -> Optional[Message]:
        """Orchestrate the conversation flow."""
        if not isinstance(message, ContextMessage):
            return None

        print(f"[{self.name}] Analyzing user intent...")

        await self.conversation.state.update_phase("agent_processing")
        await self.conversation.agent_state.set("orchestrator_active", True)

        user_message = message.history[-1]["content"] if message.history else ""
        await self.conversation.agent_state.set("user_intent_raw", user_message)

        # Intent detection (simple keyword matching for demo)
        if "appointment" in user_message.lower() or "schedule" in user_message.lower():
            print(f"[{self.name}] Detected scheduling intent, delegating...")
            await self.conversation.agent_state.set("detected_intent", "scheduling")

            # Emit event for scheduling - handler will route to sub-agent
            results = await self.emit_event("scheduling_intent_detected", message)
            return results[0] if results else None

        # Handle general queries
        await self.conversation.agent_state.set("detected_intent", "general_query")

        llm_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            *message.history,
        ]
        response = await self._llm.chat(llm_messages)

        ai_message = AIMessage(
            conversation_id=message.conversation_id,
            content=response["content"],
            service=LLMServiceType.OPENAI,
            agent_id=self.name,
        )
        if message.metadata:
            ai_message.metadata = message.metadata.copy()
            ai_message.metadata["from_processor"] = self.name

        return ai_message


class HumanInTheLoopProcessor(MessageProcessor):
    """
    Handles errors that require human intervention.
    
    Responsibilities:
    - Catch unrecoverable errors
    - Create tasks for human review
    - Pause conversation until resolution
    """

    def __init__(self, human_api: DummyHumanAPI):
        super().__init__(name="human_in_the_loop")
        self._human_api = human_api
        self._pending_tasks: Dict[str, str] = {}

    async def _process(self, message: Message) -> Optional[Message]:
        """Handle errors that need human attention."""
        if isinstance(message, ErrorMessage) and message.fatal:
            task_id = await self._human_api.create_task(message)
            self._pending_tasks[message.conversation_id] = task_id

            # Update state to indicate waiting for human
            await self.conversation.state.update_phase(
                "awaiting_human",
                metadata={"task_id": task_id, "error": message.content},
            )
            await self.conversation.agent_state.set("blocked_on_human", True)
            await self.conversation.agent_state.set("human_task_id", task_id)

            print(f"[{self.name}] Conversation paused, awaiting human resolution...")
            return None

        return None

    async def resolve_and_continue(self, conversation_id: str, resolution: str):
        """Called when a human resolves the issue."""
        if conversation_id in self._pending_tasks:
            await self._human_api.notify_resolution(conversation_id, resolution)
            del self._pending_tasks[conversation_id]

            await self.conversation.state.update_phase(
                "active", metadata={"resolution": resolution}
            )
            await self.conversation.agent_state.set("blocked_on_human", False)
            await self.conversation.agent_state.set("human_resolution", resolution)

            print(f"[{self.name}] Human resolved issue, conversation resumed")


# ============================================================================
# Observer Implementation (for monitoring/debugging)
# ============================================================================


class ConversationMonitor(Observer):
    """Observes and logs conversation events for debugging."""

    async def on_message_received(self, data):
        """Log when a processor receives a message."""
        msg_type = type(data.message).__name__
        print(f"  📥 {data.processor.name} received {msg_type}")

    async def on_message_processed(self, data: MessageProcessed):
        """Log when a processor completes processing."""
        duration_ms = data.duration_ns / 1_000_000
        result_type = type(data.result).__name__ if data.result else "None"
        print(
            f"  ✅ {data.processor.name} processed in {duration_ms:.2f}ms → {result_type}"
        )

    async def on_processor_called(self, data):
        """Log when one processor calls another."""
        print(f"  🔗 {data.source.name} → {data.target.name}")

    async def on_state_changed(self, data: StateChanged):
        """Log state transitions."""
        print(f"  🔄 Phase: {data.previous_state} → {data.new_state}")


# ============================================================================
# Main Example Setup and Execution
# ============================================================================


async def setup_conversation(
    db: DummyDB,
    crm: DummyCRM,
    llm: DummyLLM,
    human_api: DummyHumanAPI,
) -> Conversation:
    """
    Create and configure the conversation with all processors and event handlers.
    """
    # Create processors
    history_processor = ConversationHistoryProcessor(db)
    crm_processor = CRMProcessor(crm)
    scheduling_subagent = SchedulingSubAgent(llm, crm_processor)
    orchestrator = OrchestratorAgent(llm)
    human_processor = HumanInTheLoopProcessor(human_api)

    processors = [
        history_processor,
        crm_processor,
        orchestrator,
        scheduling_subagent,
        human_processor,
    ]

    # Create conversation
    conversation = Conversation(
        conversation_id="agent_demo",
        processors=processors,
        observers=[ConversationMonitor()],
    )

    # Initialize conversation
    loop = asyncio.get_event_loop()
    await conversation.setup(loop)

    # ========================================================================
    # Register Event Handlers - This is where the magic happens!
    # ========================================================================
    # Instead of hardcoded routing, we register handlers that react to events.

    router = conversation.router

    # Main entry point: Handle UserMessage
    @router.on(UserMessage, name="handle_user_message")
    async def handle_user_message(message: UserMessage, conv: Conversation):
        """
        Main flow: user message → history → crm → orchestrator → save response
        """
        print(f"\n[ROUTER] New message: {message.content}")

        await conv.state.update_phase("gathering_context")

        # Ensure user_id is available
        user_id = message.user_id or (
            message.metadata.get("user_id") if message.metadata else None
        )
        if user_id:
            await conv.agent_state.set("user_id", user_id)

        # Sequential flow through enrichment processors
        ctx = await conv.route_to("history", message)
        if ctx:
            ctx = await conv.route_to("crm", ctx)

        # Route to orchestrator for intent analysis and processing
        if ctx:
            response = await conv.route_to("orchestrator", ctx)

        # Save response to history
        if response:
            await conv.state.update_phase("finalizing")
            if not response.metadata:
                response.metadata = {}
            response.metadata["user_id"] = user_id
            await conv.route_to("history", response)

            print(f"[ROUTER] Final response: {response.content}\n")
            return response

    # Event: Scheduling intent detected
    @router.on("scheduling_intent_detected", name="route_to_scheduling")
    async def handle_scheduling_intent(message: ContextMessage, conv: Conversation):
        """Route scheduling requests to the specialized sub-agent."""
        print("[ROUTER] Routing to scheduling subagent")
        return await conv.route_to("scheduling_subagent", message)

    # Event: Data needs refresh (e.g., after scheduling an appointment)
    @router.on("needs_data_refresh", name="refresh_crm_data")
    async def handle_data_refresh(message: Message, conv: Conversation):
        """
        Refresh CRM data when needed.
        Demonstrates cycles - we go back to CRM processor.
        """
        print("[ROUTER] Refreshing CRM data...")
        user_id = message.metadata.get("user_id") if message.metadata else None
        if user_id:
            # Create a new context message to pass through CRM
            ctx = ContextMessage(
                conversation_id=message.conversation_id,
                user_info={},
                crm_data={},
                history=[],
            )
            ctx.metadata = {"user_id": user_id}
            return await conv.route_to("crm", ctx)

    # Event: CRM data refreshed (runs concurrently - just for logging)
    @router.on("crm_data_refreshed", name="log_crm_refresh", concurrent=True)
    async def log_crm_refresh(message: Message, conv: Conversation):
        """Log when CRM data is refreshed (non-blocking)."""
        user_id = message.metadata.get("user_id") if message.metadata else "unknown"
        print(f"[ROUTER] CRM data refreshed for user {user_id}")

    # Error handling: Route fatal errors to human oversight
    @router.on(ErrorMessage, name="handle_errors")
    async def handle_error(message: ErrorMessage, conv: Conversation):
        """Route fatal errors to human-in-the-loop processor."""
        if message.fatal:
            print("[ROUTER] Routing fatal error to human oversight")
            return await conv.route_to("human_in_the_loop", message)

    return conversation


async def main():
    """Main execution function."""
    print("=" * 80)
    print("🤖 Event-Driven Agent Orchestration Example")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  • Multi-agent orchestration with dynamic routing")
    print("  • Intent detection and delegation to sub-agents")
    print("  • Tool calling (appointment scheduling)")
    print("  • Event-driven data refresh cycles")
    print("  • Integration with external systems (DB, CRM)")
    print("  • State management and context sharing")
    print()

    # Setup external dependencies
    db = DummyDB()
    crm = DummyCRM()
    llm = DummyLLM()
    human_api = DummyHumanAPI()

    # Create and configure conversation
    conversation = await setup_conversation(db, crm, llm, human_api)

    # Display registered event handlers
    print("=" * 80)
    print("📋 Registered Event Handlers")
    print("=" * 80)
    events = conversation.router.list_events()
    print(f"Registered {len(events)} event types:")
    for event in events:
        event_name = event if isinstance(event, str) else event.__name__
        handlers = conversation.router.get_handlers(event)
        print(f"  • {event_name}: {len(handlers)} handler(s)")
        for handler in handlers:
            concurrent_marker = " [concurrent]" if handler.concurrent else ""
            print(f"    - {handler.name}{concurrent_marker}")

    # ========================================================================
    # Execute Example Conversation
    # ========================================================================

    print("\n" + "=" * 80)
    print("💬 Processing User Message")
    print("=" * 80)
    print()

    user_message = UserMessage(
        conversation_id="agent_demo",
        content="I need to schedule a follow-up appointment",
        service=LLMServiceType.OPENAI,
        user_id="user_123",
    )
    user_message.metadata = {"user_id": "user_123"}

    # Emit the message - the router handles all the orchestration!
    await conversation.router.emit(user_message)

    # Wait for any concurrent handlers to complete
    await conversation.router.wait_for_concurrent_handlers(timeout=5.0)

    # ========================================================================
    # Display Final State
    # ========================================================================

    print("\n" + "=" * 80)
    print("📊 Final Conversation State")
    print("=" * 80)
    print(f"Phase: {conversation.state.phase}")
    print(f"Last Processor: {conversation.state.current_processor}")

    print("\n🗂️  Shared Agent State:")
    for key, value in conversation.agent_state.data.items():
        if isinstance(value, dict) and len(str(value)) > 100:
            print(f"  • {key}: <dict with {len(value)} keys>")
        else:
            print(f"  • {key}: {value}")

    print("\n📜 Phase History (last 5 events):")
    for event_type, value, timestamp, metadata in conversation.state.history[-5:]:
        if event_type == "phase_change":
            print(f"  • Phase → {value}")
        elif event_type == "processor_change":
            print(f"  • Processor → {value}")

    # Cleanup
    print("\n" + "=" * 80)
    print("🧹 Cleaning up...")
    await conversation.cleanup()
    print("✅ Done!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

