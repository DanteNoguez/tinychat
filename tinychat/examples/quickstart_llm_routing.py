import asyncio
from dataclasses import dataclass
from typing import Literal

from loguru import logger
from dotenv import load_dotenv

from tinychat.messages.messages import Message, EgressMessage
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.services.llm.openai_llm import OpenAILLM, OpenAILLMConfig
from tinychat.services.llm.anthropic_llm import AnthropicLLM, AnthropicLLMConfig
from tinychat.asynchronous.manager import TaskManagerParams
from tinychat.observers.observer import BaseObserver, MessageProcessed, MessageReceived
from tinychat.utils.logging import configure_pretty_logging

load_dotenv()

configure_pretty_logging(debug_level=5)

# ==============================================================================
# 1. Define Message Types (The "Protocol")
# ==============================================================================


@dataclass(frozen=True)
class UserQuery(Message): ...


@dataclass(frozen=True)
class TechnicalIssue(Message):
    """Route to technical support for hardware/software issues."""

    severity: Literal["low", "medium", "high"]
    device: str


@dataclass(frozen=True)
class BillingInquiry(Message):
    """Route to billing department for payment/invoice issues."""

    customer_id: str


@dataclass(frozen=True)
class ResolvedTicket(Message):
    """Emit a final resolution for the user."""

    agent_name: str


# ==============================================================================
# 2. Define Custom Processors
# ==============================================================================


class BillingProcessor(MessageProcessor):
    """A simple deterministic processor (simulating a legacy system)."""

    async def _process(self, message: BillingInquiry) -> ResolvedTicket:
        return ResolvedTicket(
            content=f"Refund processed for {message.customer_id}. Refund ID: #REF-{message.id[:4]}",
            agent_name="BillingBot_v1",
        )


class ResponseFormatter(MessageProcessor):
    """Formats the final response for egress."""

    async def _process(self, message: ResolvedTicket) -> EgressMessage:
        formatted = f"\n[{message.agent_name}] says: {message.content}\n"
        logger.success(f"✅ Final Response: {formatted}")
        return EgressMessage(content=formatted)


class SimpleLogger(BaseObserver):
    """Logs the flow of message types through the system."""

    async def on_message_processed(self, event: MessageProcessed):
        logger.debug(
            f"🔄 {event.source_processor.name} -> {type(event.source_message).__name__}"
        )

    async def on_exception(self, source_message: Message, exception: Exception) -> None:
        logger.error(
            f"❌ {source_message.name} Exception: {exception} at {source_message.timestamp}"
        )

    async def on_message_received(self, message: MessageReceived) -> None:
        logger.debug(f"📨 {message.source_processor.name} -> {message.content}")


# ==============================================================================
# 3. Main Flow
# ==============================================================================


async def main():
    # --- Configuration ---
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
        observers=[SimpleLogger()],
    )

    # --- 1. Router Agent (OpenAI) ---
    # This agent decides WHERE to go next by generating a specific Message type.
    # The framework detects `output_types` and auto-generates the routing tools.
    router_llm = OpenAILLM(
        name="TriageAgent",
        llm_config=OpenAILLMConfig(
            instructions=(
                "You are a triage agent. Analyze the user's query. "
                "If it's a technical problem, issue a TechnicalIssue. "
                "If it's about payments or accounts, issue a BillingInquiry. "
                "Extract relevant details into the fields."
                "Be concise."
            )
        ),
        # CRITICAL: Declaring output_types triggers the tool generation!
        output_types={TechnicalIssue, BillingInquiry},
    )

    # --- 2. Technical Agent (Anthropic) ---
    # This agent handles technical issues and solves them.
    tech_llm = AnthropicLLM(
        name="TechSupport",
        llm_config=AnthropicLLMConfig(
            instructions="You are a helpful tech support engineer. Provide a concise solution."
        ),
        output_types={ResolvedTicket},  # It produces a resolution
    )

    # --- 3. Deterministic Processors ---
    billing = BillingProcessor(name="BillingSystem", output_types={ResolvedTicket})
    formatter = ResponseFormatter(name="Formatter", output_types={EgressMessage})

    # --- 4. Wiring the Graph ---
    processor = CompositeProcessor(
        handlers={
            UserQuery: router_llm,  # Entry point
            TechnicalIssue: tech_llm,  # Router -> Tech
            BillingInquiry: billing,  # Router -> Billing
            ResolvedTicket: formatter,  # All paths lead here
        },
        output_types={EgressMessage},
    )

    await processor.setup(config)

    # --- Test Run 1: Technical Issue ---
    logger.info("--- Test 1: Technical Request ---")
    msg1 = UserQuery(content="My laptop screen is flickering and turning blue.")
    await processor.process(msg1)

    # --- Test Run 2: Billing Issue ---
    logger.info("--- Test 2: Billing Request ---")
    msg2 = UserQuery(content="I was charged twice for my subscription user_id: 888.")
    await processor.process(msg2)

    logger.success("Tests completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
