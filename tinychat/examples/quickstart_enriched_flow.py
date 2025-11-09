import asyncio
from typing import Optional
from dataclasses import dataclass
from loguru import logger

from tinychat.messages.messages import (
    IngressMessage,
    EgressMessage,
    Message,
)
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.observers.observer import (
    BaseObserver,
    MessageReceived,
    MessageProcessed,
)


@dataclass(frozen=True)
class CRMMessage(Message):
    content: dict
    user_id: str
    user_info: dict
    user_message: Message


@dataclass(frozen=True)
class EnrichedMessage(Message):
    content: dict
    user_id: str
    user_info: dict
    history: list[str]
    user_message: Message


@dataclass(frozen=True)
class ReplyRequestMessage(Message):
    content: str
    user_id: str
    user_info: dict
    history: list[str]
    user_message: Message


@dataclass(frozen=True)
class ReplyMessage(Message):
    content: str
    user_message: Message


class LoggingObserver(BaseObserver):
    def __init__(self):
        self._received_messages: dict[str, MessageReceived] = {}

    async def on_message_received(self, message: MessageReceived) -> None:
        logger.info(f"📨 [{message.source_processor.name}] Received: {message.content}")
        # Track received message for latency calculation
        self._received_messages[message.source_message.id] = message

    async def on_message_processed(self, message: MessageProcessed) -> None:
        logger.info(
            f"✅ [{message.source_processor.name}] Processed: {message.content}"
        )

        # Calculate and log latency
        received_msg = self._received_messages.get(message.source_message.id)
        if received_msg:
            latency_ns = message.timestamp - received_msg.timestamp
            latency_us = latency_ns / 1_000
            logger.info(
                f"⏱️  [{message.source_processor.name}] Latency: {latency_us:.2f}μs"
            )
            del self._received_messages[message.source_message.id]

    async def on_exception(self, source_message: Message, exception: Exception) -> None:
        logger.error(
            f"❌ [{source_message.name}] Exception: {exception} at {source_message.timestamp}"
        )


class CRMProcessor(MessageProcessor):
    """Processor that retrieves user information from a CRM."""

    def __init__(
        self, name: str = "CRM", output_types: set[type[Message]] | None = None
    ):
        super().__init__(name=name, output_types=output_types)
        # Mock user database
        self._users = {
            "user_123": {
                "name": "Alice",
                "email": "alice@example.com",
                "tier": "premium",
            },
            "user_456": {
                "name": "Bob",
                "email": "bob@example.com",
                "tier": "standard",
            },
        }

    async def _process(self, message: IngressMessage) -> Optional[Message]:
        user_id = message.user_id
        user_info = self._users.get(
            user_id, {"name": "Unknown", "email": "unknown@example.com", "tier": "free"}
        )

        logger.info(f"🔍 CRM: Retrieved user info for {user_id}: {user_info}")

        return CRMMessage(
            content={"user_id": user_id, "user_info": user_info},
            user_id=user_id,
            user_message=message,
            user_info=user_info,
        )


class DBProcessor(MessageProcessor):
    """Processor that retrieves chat history from a database."""

    def __init__(
        self, name: str = "DB", output_types: set[type[Message]] | None = None
    ):
        super().__init__(name=name, output_types=output_types)
        # Mock chat history database
        self._history = {
            "user_123": [
                "User: Hello!",
                "Bot: Hi Alice! How can I help?",
                "User: What's my account status?",
                "Bot: You're a premium member!",
            ],
            "user_456": [
                "User: Hi there",
                "Bot: Hello Bob!",
            ],
        }

    async def _process(self, message: CRMMessage) -> Optional[Message]:
        user_id = message.user_id
        history = self._history.get(user_id, [])

        logger.info(
            f"💾 DB: Retrieved {len(history)} messages from history for {user_id}"
        )

        return EnrichedMessage(
            content={
                "user_id": user_id,
                "user_info": message.user_info,
                "history": history,
            },
            user_id=user_id,
            user_info=message.user_info,
            history=history,
            user_message=message.user_message,
        )


class OrchestratorAgent(MessageProcessor):
    """Orchestrator that routes to reply agent and approves responses."""

    async def _process(self, message: Message) -> Optional[Message]:
        if isinstance(message, EnrichedMessage):
            # Forward to reply agent
            logger.info("🎯 Orchestrator: Forwarding enriched message to reply agent")
            return ReplyRequestMessage(
                content=f"New message from {message.user_id}: {message.user_message.content}",
                user_id=message.user_id,
                user_info=message.user_info,
                history=message.history,
                user_message=message.user_message,
            )
        elif isinstance(message, ReplyMessage):
            # Approve and return as egress
            logger.info(f"✓ Orchestrator: Approving reply: '{message.content}'")
            return EgressMessage(
                content=message.content,
            )
        else:
            raise ValueError(f"Unexpected message type: {type(message)}")


class ReplyAgent(MessageProcessor):
    """Agent that generates responses based on enriched context."""

    async def _process(self, message: ReplyRequestMessage) -> Optional[Message]:
        # Generate response using enriched context
        user_name = message.user_info.get("name", "there")
        tier = message.user_info.get("tier", "free")
        history_count = len(message.history)

        reply_text = (
            f"Hello {user_name}! Thanks for your message: '{message.user_message.content}'. "
            f"I can see you're a {tier} member and we've chatted {history_count} times before. "
            f"How can I assist you today?"
        )

        logger.info("🤖 Reply Agent: Generated response")

        return ReplyMessage(
            content=reply_text,
            user_message=message.user_message,
        )


async def main():
    # Setup message bus configuration
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
        observers=[LoggingObserver()],
    )

    # Create enrichment processors (CRM → DB) with output type declarations
    crm = CRMProcessor(output_types={CRMMessage})
    db = DBProcessor(output_types={EnrichedMessage})

    # Create agent processors with output type declarations
    orchestrator = OrchestratorAgent(
        name="orchestrator", output_types={ReplyRequestMessage, EgressMessage}
    )
    reply_agent = ReplyAgent(name="reply_agent", output_types={ReplyMessage})

    # Create inner message bus for agents with its own max_hops limit
    # Flow inside agent bus: EnrichedMessage → Orchestrator → ReplyRequest → ReplyAgent → Reply → Orchestrator → Egress
    agent_bus = CompositeProcessor(
        handlers={
            EnrichedMessage: orchestrator,
            ReplyRequestMessage: reply_agent,
            ReplyMessage: orchestrator,
        },
        max_hops=10,  # Agents have their own hop limit
        output_types={EgressMessage},
    )

    # Setup outer message bus with type-based routing
    # Flow: Ingress → CRM → DB → EnrichedMessage → [Agent Bus] → Egress
    bus = CompositeProcessor(
        handlers={
            IngressMessage: crm,
            CRMMessage: db,
            EnrichedMessage: agent_bus,
        },
        max_hops=5,  # Outer bus hop limit
        output_types={EgressMessage},
    )
    await bus.setup(config)

    # Create and process ingress message
    logger.info("=" * 80)
    logger.info("Starting enriched message flow example")
    logger.info("=" * 80)

    message = IngressMessage(
        content="I need help with my account",
        conversation_id="enriched-demo",
        user_id="user_123",
    )

    result = await bus.process(message)

    logger.info("=" * 80)
    logger.info(f"Final result: {result.content if result else 'None'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
