from typing import Optional
from loguru import logger

from anthropic import AsyncAnthropic
from anthropic.types import (
    Message as AnthropicMessage,
    ToolUseBlock,
    TextBlock,
)

from tinychat.messages.messages import Message
from tinychat.services.llm.models import (
    AnthropicLLMConfig,
    LLMMessage,
    ToolCall,
    ToolCallOutput,
    Tool,
    AnthropicUserMessage,
    AnthropicAssistantMessage,
)
from tinychat.services.llm.llm import LLMService
from tinychat.services.llm.tools import TypeSchema


class AnthropicLLM(LLMService):
    def __init__(
        self,
        *,
        llm_config: AnthropicLLMConfig,
        client: Optional[AsyncAnthropic] = None,
        **kwargs,
    ):
        super().__init__(llm_config=llm_config, **kwargs)
        self.client = client or AsyncAnthropic(
            api_key=llm_config.api_key,
            max_retries=llm_config.max_retries if llm_config.enable_retries else 2,
        )
        self.tools_schema = self.create_tools_schema(self.tools) if self.tools else None

    def _ensure_llm_message(self, message: Message) -> LLMMessage:
        if isinstance(message, LLMMessage):
            return message
        return AnthropicUserMessage(content=message.content)

    async def _generate_completion(self, depth: int = 0) -> LLMMessage:
        if depth > self._llm_config.recursion_limit:
            raise RuntimeError("Recursion limit reached.")

        # Prepare Messages
        # Convert internal history to Anthropic format
        raw_history = [m.to_anthropic_format() for m in self.chat_history]

        # Ensure alternation (Anthropic strict requirement)
        api_messages = self._prepare_messages_for_api(raw_history)

        kwargs = {
            "model": self._llm_config.model_name,
            "max_tokens": self._llm_config.max_tokens,
            "temperature": self._llm_config.temperature,
            "messages": api_messages,
        }

        # Use "instructions" via the system parameter
        if self.instructions:
            kwargs["system"] = self.instructions

        logger.trace(
            f"{self} - Instructions: {self.instructions} - Chat history: {api_messages}"
        )

        if self.tools_schema:
            kwargs["tools"] = self.tools_schema

        response: AnthropicMessage = await self.client.messages.create(**kwargs)

        # Handle Stop Reason: Tool Use
        if response.stop_reason and response.stop_reason == "tool_use":
            return await self._handle_tool_use(response, depth)

        # Handle Standard Text Response
        content_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        assistant_msg = AnthropicAssistantMessage(content=content_text)
        self.add_message(assistant_msg)
        return assistant_msg

    async def _handle_tool_use(
        self, response: AnthropicMessage, depth: int
    ) -> LLMMessage:
        tool_calls: list[ToolUseBlock] = []

        # Note: We add individual ToolCalls to our linear history.
        # _prepare_messages_for_api will merge them back into a single block for the API.
        for block in response.content:
            if isinstance(block, TextBlock):
                self.add_message(AnthropicAssistantMessage(content=block.text))

            elif isinstance(block, ToolUseBlock):
                # Check if the tool name corresponds to a routing message
                if routing_msg := self._get_routing_message(block.name, block.input):
                    self.add_message(
                        AnthropicAssistantMessage(content=str(block.input))
                    )
                    return routing_msg

                tool_calls.append(block)
                self.add_message(
                    ToolCall(
                        tool_id=block.id,
                        tool_call_id=block.id,
                        tool_name=block.name,
                        tool_arguments=block.input,
                        content=block.input,
                    )
                )

        for block in tool_calls:
            result = await self.execute_tool(block.name, block.input)

            self.add_message(
                ToolCallOutput(
                    tool_call_id=block.id, tool_output=result, content=result
                )
            )

        return await self._generate_completion(depth + 1)

    def _prepare_messages_for_api(self, history: list[dict]) -> list[dict]:
        """
        Ensures messages alternate between user and assistant.
        Merges consecutive messages of the same role.
        """
        sanitized = []
        current_msg = history[0].copy()

        # Normalize content to list
        if isinstance(current_msg["content"], str):
            current_msg["content"] = [{"type": "text", "text": current_msg["content"]}]
        elif isinstance(current_msg["content"], list):
            current_msg["content"] = list(current_msg["content"])

        for i in range(1, len(history)):
            next_msg = history[i]

            if next_msg["role"] == current_msg["role"]:
                # Merge
                next_content = next_msg["content"]
                if isinstance(next_content, str):
                    current_msg["content"].append(
                        {"type": "text", "text": next_content}
                    )
                elif isinstance(next_content, list):
                    current_msg["content"].extend(next_content)
            else:
                sanitized.append(current_msg)
                current_msg = next_msg.copy()
                if isinstance(current_msg["content"], str):
                    current_msg["content"] = [
                        {"type": "text", "text": current_msg["content"]}
                    ]
                elif isinstance(current_msg["content"], list):
                    current_msg["content"] = list(current_msg["content"])

        sanitized.append(current_msg)
        return sanitized

    # ==========================================================================
    # Schema Helpers
    # ==========================================================================

    def create_tools_schema(self, tools: list[Tool]) -> list[dict]:
        output = []
        for tool in tools:
            properties = {}
            required = []
            for param in tool.parameters:
                param_schema = self._type_schema_to_dict(param.schema)
                param_schema["description"] = param.description
                properties[param.name] = param_schema
                if param.required:
                    required.append(param.name)

            output.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        logger.trace(f"{self} - Tools schema: {output}")
        return output

    def _type_schema_to_dict(self, schema: TypeSchema) -> dict:
        if schema.json_schema:
            return schema.json_schema

        json_schema = {"type": schema.data_type}
        if schema.description:
            json_schema["description"] = schema.description
        if schema.enum:
            json_schema["enum"] = schema.enum
        if schema.items:
            json_schema["items"] = self._type_schema_to_dict(schema.items)
        if schema.properties:
            json_schema["properties"] = {
                k: self._type_schema_to_dict(v) for k, v in schema.properties.items()
            }
            json_schema["additionalProperties"] = False

        return json_schema
