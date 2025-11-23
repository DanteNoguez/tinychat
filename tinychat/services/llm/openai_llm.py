import json
from typing import Optional
from loguru import logger

from openai import AsyncOpenAI
from openai.types.responses import Response

from tinychat.messages.messages import Message
from tinychat.services.llm.models import (
    OpenAILLMConfig,
    Tool,
    OpenAIAssistantMessage,
    OpenAIUserMessage,
    OpenAISystemMessage,
    LLMMessage,
    ToolCall,
    ToolCallOutput,
)
from tinychat.services.llm.llm import LLMService
from tinychat.services.llm.tools import TypeSchema

try:
    from pydantic import BaseModel

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    class BaseModel:
        pass  # Dummy for type checking


OPENAI_ROLE_TO_MESSAGE_CLASS_MAP = {
    "assistant": OpenAIAssistantMessage,
    "user": OpenAIUserMessage,
    "system": OpenAISystemMessage,
}


# Responses API docs: https://platform.openai.com/docs/api-reference/responses/create
# Function calling docs: https://platform.openai.com/docs/guides/function-calling
class OpenAILLM(LLMService):
    def __init__(
        self,
        *,
        llm_config: OpenAILLMConfig,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ):
        super().__init__(
            llm_config=llm_config, output_types={OpenAIAssistantMessage}, **kwargs
        )
        self.client = client or AsyncOpenAI(
            api_key=llm_config.api_key,
            max_retries=llm_config.max_retries if llm_config.enable_retries else None,
        )
        self.model_name = llm_config.model_name
        self.config = llm_config

        # Tools Setup
        self.tools = llm_config.tools
        self.tools_schema = self.create_tools_schema(self.tools) if self.tools else None
        self.tools_by_name = (
            {tool.name: tool for tool in self.tools} if self.tools else None
        )

        # State: Chat history in OpenAI format (list of dicts)
        self.chat_history: list[dict] = []
        self.system_message: dict | None = None

        # Initialize system prompt if provided
        if self.config.prompt:
            self.set_system_prompt(self.config.prompt)

    # ==========================================================================
    # State Management & Observability
    # ==========================================================================

    @property
    def context(self) -> list[LLMMessage]:
        """
        Returns the current chat history as a list of Tinychat LLMMessage objects.
        Useful for monitoring, evaluations, and debugging within the framework.
        """
        return self.openai_to_tinychat_format(self.chat_history)

    def set_system_prompt(self, prompt: str | LLMMessage) -> None:
        """
        Sets the system prompt and ensures it is always the first message.
        """
        content = prompt.content if isinstance(prompt, Message) else prompt
        self.system_message = {"role": "system", "content": content}

        if self.chat_history and self.chat_history[0]["role"] == "system":
            self.chat_history[0] = self.system_message
        else:
            self.chat_history.insert(0, self.system_message)

    def add_message(self, message: LLMMessage) -> None:
        """Manually add a message to the history."""
        self.chat_history.append(message.to_openai_format())

    def add_messages(self, messages: list[LLMMessage]) -> None:
        """Manually add multiple messages to the history."""
        for message in messages:
            self.add_message(message)

    def clear_history(self) -> None:
        """Resets conversation but preserves system prompt."""
        self.chat_history = [self.system_message] if self.system_message else []

    # ==========================================================================
    # MessageProcessor Implementation (Strict Boundary)
    # ==========================================================================

    async def _process(self, message: Message) -> OpenAIAssistantMessage:
        """
        Processes an incoming message using strict stateful logic.

        1. Ingest: Convert generic Message -> LLMMessage and update history.
        2. Process: Generate response using accumulated history.
        3. Egress: Return the generated LLMMessage.
        """
        # 1. Ingest
        llm_message = self._ensure_llm_message(message)
        self.add_message(llm_message)
        logger.debug(f"{self} - Current chat history: {self.chat_history}")

        # 2. Process & Egress
        # _generate_completion handles adding the response to history internally
        return await self._generate_completion()

    def _ensure_llm_message(self, message: Message) -> LLMMessage:
        """Boundary adapter: Converts generic Messages to LLMUserMessages."""
        if isinstance(message, LLMMessage):
            return message
        return OpenAIUserMessage(content=message.content)

    # ==========================================================================
    # Generation Logic
    # ==========================================================================

    async def _generate_completion(self, depth: int = 0) -> LLMMessage:
        """
        Internal recursive generation loop.
        Handles strict tool execution and history updates.
        """
        if depth > self.config.recursion_limit:
            raise RuntimeError(
                f"Recursion limit of {self.config.recursion_limit} reached."
            )

        response = await self.client.responses.create(
            model=self.model_name,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            tools=self.tools_schema,
            tool_choice="auto" if self.config.tools else None,
            input=self.chat_history,
        )

        # Case A: Tool Calls
        if response.output[0].type == "function_call":
            await self._handle_function_calls(response)
            return await self._generate_completion(depth + 1)

        # Case B: Assistant Text
        return self._handle_text_response(response)

    async def _handle_function_calls(self, response: Response) -> None:
        for item in response.output:
            if item.type != "function_call":
                continue

            # 1. Record the call
            self.chat_history.append(
                {
                    "id": item.id,
                    "call_id": item.call_id,
                    "type": "function_call",
                    "name": item.name,
                    "arguments": item.arguments,
                }
            )

            # 2. Execute
            function_name = item.name
            try:
                function_args = json.loads(item.arguments)
                if tool := self.tools_by_name.get(function_name):
                    logger.debug(
                        f"{self} - Executing tool: {function_name} with arguments: {function_args}"
                    )
                    result = await tool.run(**function_args)
                    result_str = str(result)
                else:
                    result_str = f"Error: Tool {function_name} not found."
            except Exception as e:
                result_str = f"Error executing tool: {e}"

            # 3. Record the output
            self.chat_history.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({"result": result_str}),
                }
            )

    def _handle_text_response(self, response: Response) -> OpenAIAssistantMessage:
        content = ""
        for output_item in response.output:
            if output_item.type == "message":
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        content += content_item.text

        self.chat_history.append({"role": "assistant", "content": content})
        return OpenAIAssistantMessage(content=content)

    # ==========================================================================
    # Helpers & Converters
    # ==========================================================================

    def openai_to_tinychat_format(self, messages: list[dict]) -> list[LLMMessage]:
        """Converts OpenAI list[dict] format to Tinychat list[LLMMessage]."""
        output = []
        for message in messages:
            if msg_type := message.get("type"):
                if msg_type == "function_call":
                    output.append(
                        ToolCall(
                            content="",
                            tool_id=message.get("id"),
                            tool_call_id=message.get("call_id"),
                            tool_name=message.get("name"),
                            tool_arguments=message.get("arguments"),
                        )
                    )
                elif msg_type == "function_call_output":
                    output.append(
                        ToolCallOutput(
                            content="",
                            tool_call_id=message.get("call_id"),
                            tool_output=message.get("output"),
                        )
                    )
            elif role := message.get("role"):
                if message_class := OPENAI_ROLE_TO_MESSAGE_CLASS_MAP.get(role):
                    output.append(
                        message_class(
                            content=message.get("content"),
                        )
                    )
        return output

    def create_tools_schema(self, tools: list[Tool]) -> list[dict]:
        output = []
        for tool in tools:
            properties = {}
            required = []
            for param in tool.parameters:
                param_schema = self._type_schema_to_dict(param.schema)
                param_schema["description"] = param.description
                properties[param.name] = param_schema
                # In Strict Mode, all parameters must be listed as required.
                required.append(param.name)

            output.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            )
        return output

    def _type_schema_to_dict(self, schema: TypeSchema) -> dict:
        # Override for Pydantic Models
        if schema.json_schema:
            # If we have a pre-generated schema (from Pydantic), use it.
            # We must ensure additionalProperties is False recursively for Strict Mode.
            final_schema = self._enforce_strict_mode(schema.json_schema)
            if schema.nullable:
                current_type = final_schema.get("type")
                if isinstance(current_type, str):
                    final_schema["type"] = [current_type, "null"]
            return final_schema

        type_val = [schema.data_type, "null"] if schema.nullable else schema.data_type
        json_schema = {"type": type_val}
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
            json_schema["required"] = schema.required or []
            json_schema["additionalProperties"] = False
        return json_schema

    def _enforce_strict_mode(self, schema: dict) -> dict:
        """
        Recursively ensures a JSON schema is compatible with OpenAI Strict Mode.
        Mainly enforces additionalProperties: False on all objects.
        """
        new_schema = schema.copy()

        # Clean up Pydantic metadata OpenAI doesn't like
        new_schema.pop("title", None)

        schema_type = new_schema.get("type")

        if schema_type == "object":
            new_schema["additionalProperties"] = False

            if "properties" in new_schema:
                new_props = {}
                for k, v in new_schema["properties"].items():
                    new_props[k] = self._enforce_strict_mode(v)
                new_schema["properties"] = new_props

                if "required" not in new_schema:
                    new_schema["required"] = list(new_props.keys())

        if schema_type == "array" and "items" in new_schema:
            new_schema["items"] = self._enforce_strict_mode(new_schema["items"])

        # Handle Defs ($defs) - Pydantic puts shared definitions here
        if defs := new_schema.get("$defs"):
            new_defs = {}
            for k, v in defs.items():
                new_defs[k] = self._enforce_strict_mode(v)
            new_schema["$defs"] = new_defs

        return new_schema
