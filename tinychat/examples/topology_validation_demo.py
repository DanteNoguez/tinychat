import asyncio
from dataclasses import dataclass
from loguru import logger

from tinychat.messages.messages import Message, EgressMessage
from tinychat.processors.message_processor import MessageProcessor, SetupConfig
from tinychat.processors.composite import CompositeProcessor
from tinychat.asynchronous.manager import TaskManagerParams


@dataclass(frozen=True)
class UserInput(Message):
    content: str


@dataclass(frozen=True)
class EnrichedData(Message):
    content: str
    metadata: dict


@dataclass(frozen=True)
class ProcessedResult(Message):
    content: str
    score: float


@dataclass(frozen=True)
class HighScoreResult(Message):
    content: str
    score: float
    priority: str


class EnricherProcessor(MessageProcessor):
    """Enriches user input with metadata."""

    async def _process(self, message: UserInput) -> EnrichedData:
        return EnrichedData(
            content=message.content,
            metadata={"enriched": True, "timestamp": message.timestamp},
        )


class AnalyzerProcessor(MessageProcessor):
    """Analyzes enriched data and produces different results based on score."""

    async def _process(
        self, message: EnrichedData
    ) -> ProcessedResult | HighScoreResult:
        content = f"Analyzed: {message.content}"
        score = len(message.content) / 100.0  # Simple scoring based on length

        # Route to different message types based on score
        if score > 0.5:
            return HighScoreResult(
                content=content,
                score=score,
                priority="high",
            )
        else:
            return ProcessedResult(
                content=content,
                score=score,
            )


class FormatterProcessor(MessageProcessor):
    """Formats the final result as an egress message."""

    async def _process(self, message: ProcessedResult) -> EgressMessage:
        formatted = f"{message.content} (score: {message.score})"
        return EgressMessage(content=formatted)


class HighScoreFormatterProcessor(MessageProcessor):
    """Formats high score results with special treatment."""

    async def _process(self, message: HighScoreResult) -> EgressMessage:
        formatted = f"⭐ {message.content} (score: {message.score}, priority: {message.priority})"
        return EgressMessage(content=formatted)


async def demo_valid_topology():
    """Demonstrate a valid topology with 1:N routing (one processor, two output types)."""
    logger.info("Valid Topology Demo - 1:N Routing")

    enricher = EnricherProcessor(output_types={EnrichedData})
    # AnalyzerProcessor produces TWO different types based on score
    analyzer = AnalyzerProcessor(output_types={ProcessedResult, HighScoreResult})
    formatter = FormatterProcessor(output_types={EgressMessage})
    high_score_formatter = HighScoreFormatterProcessor(output_types={EgressMessage})

    # This will pass validation - all output types have handlers
    composite = CompositeProcessor(
        handlers={
            UserInput: enricher,
            EnrichedData: analyzer,  # One processor, two possible outputs
            ProcessedResult: formatter,  # Handler for low scores
            HighScoreResult: high_score_formatter,  # Handler for high scores
        },
        max_hops=10,
    )

    # Setup
    config = SetupConfig(
        task_manager_params=TaskManagerParams(loop=asyncio.get_running_loop()),
    )
    await composite.setup(config)

    # Test with short input (low score) - routes to FormatterProcessor
    short_input = UserInput(content="Hi!")
    result1 = await composite.process(short_input)
    logger.info(f"Short input: '{short_input.content}' → {result1.content}")

    # Test with long input (high score) - routes to HighScoreFormatterProcessor
    long_input = UserInput(
        content="Hello, world! This is a much longer message that will score higher!"
    )
    result2 = await composite.process(long_input)
    logger.success(f"Long input: '{long_input.content}' → {result2.content}")


async def demo_invalid_topology():
    """Demonstrate topology validation catching an error."""
    logger.info("Invalid Topology Demo")

    enricher = EnricherProcessor(output_types={EnrichedData})
    analyzer = AnalyzerProcessor(output_types={ProcessedResult})
    # Missing formatter - ProcessedResult has no handler!

    try:
        # This will fail validation - ProcessedResult has no handler
        CompositeProcessor(
            handlers={
                UserInput: enricher,
                EnrichedData: analyzer,
                # ProcessedResult: formatter,  # Missing!
            },
            max_hops=10,
        )
        logger.error("Should have raised ValueError!")
    except ValueError as e:
        logger.warning(f"Validation caught the error: {e}")


async def demo_partial_declaration():
    """Demonstrate mixing processors with and without output_types."""
    logger.info("Partial Declaration Demo")

    class SimpleProcessor(MessageProcessor):
        """Processor without declared output_types - skips validation."""

        async def _process(self, message: Message) -> EgressMessage:
            return EgressMessage(content="done")

    # No output_types declared - skips validation
    simple = SimpleProcessor()

    # This works - SimpleProcessor doesn't declare outputs, so no validation
    CompositeProcessor(
        handlers={
            UserInput: simple,
        },
        max_hops=10,
    )


async def demo_terminal_types():
    logger.info("Terminal Types Visualization Demo")

    class FilterProcessor(MessageProcessor):
        """Processor that may return None (filtering logic)."""

        async def _process(self, message: UserInput) -> EnrichedData | None:
            if len(message.content) < 3:
                return None  # Filter out short messages
            return EnrichedData(content=message.content, metadata={})

    class TerminatorProcessor(MessageProcessor):
        """Processor that terminates with EgressMessage."""

        async def _process(self, message: EnrichedData) -> EgressMessage:
            return EgressMessage(content=f"Processed: {message.content}")

    # Declare both terminal types
    filter_proc = FilterProcessor(output_types={EnrichedData, type(None)})
    terminator = TerminatorProcessor(output_types={EgressMessage})

    CompositeProcessor(
        handlers={
            UserInput: filter_proc,
            EnrichedData: terminator,
        },
        max_hops=10,
    )

    logger.info(
        "Note: Terminal types (None, EgressMessage) are shown but don't require handlers."
    )


async def main():
    await demo_valid_topology()
    await demo_invalid_topology()
    await demo_partial_declaration()
    await demo_terminal_types()
    logger.success("Done")


if __name__ == "__main__":
    asyncio.run(main())
