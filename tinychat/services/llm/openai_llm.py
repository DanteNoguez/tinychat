import json
from typing import Optional

from openai import AsyncOpenAI
from openai.types.responses import Response, ResponseFunctionToolCall

from tinychat.messages.messages import Message
from tinychat.services.llm.models import (
    OpenAILLMConfig,
    Tool,
    OpenAIAssistantMessage,
    OpenAIUserMessage,
    LLMMessage,
    ToolCall,
    ToolCallOutput,
)
from tinychat.services.llm.llm import LLMService
from tinychat.services.llm.tools import TypeSchema


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
        super().__init__(llm_config=llm_config, **kwargs)
        self.client = client or AsyncOpenAI(
            api_key=llm_config.api_key,
            max_retries=llm_config.max_retries if llm_config.enable_retries else None,
        )
        self.tools_schema = self.create_tools_schema(self.tools) if self.tools else None

    def _ensure_llm_message(self, message: Message) -> LLMMessage:
        if isinstance(message, LLMMessage):
            return message
        return OpenAIUserMessage(content=message.content)

    async def _generate_completion(self, depth: int = 0) -> LLMMessage:
        if depth > self._llm_config.recursion_limit:
            raise RuntimeError("Recursion limit reached.")

        # Prepare Messages
        api_messages: list[dict] = []
        if self.instructions:
            api_messages.append({"role": "system", "content": self.instructions})

        # Convert internal history to OpenAI format
        api_messages.extend([m.to_openai_format() for m in self.chat_history])

        response = await self.client.responses.create(
            model=self._llm_config.model_name,
            temperature=self._llm_config.temperature,
            max_output_tokens=self._llm_config.max_tokens,
            tools=self.tools_schema,
            tool_choice="auto" if self._llm_config.tools else None,
            input=api_messages,
        )

        # Handle Response
        output_item = response.output[0]

        # Case A: Tool Call
        if output_item.type == "function_call":
            return await self._handle_tool_call(output_item, depth)

        # Case B: Text Message
        return self._handle_text_response(response)

    async def _handle_tool_call(
        self, item: ResponseFunctionToolCall, depth: int
    ) -> LLMMessage:
        args = json.loads(item.arguments)

        tool_call = ToolCall(
            content=args,
            tool_id=item.id,
            tool_call_id=item.call_id,
            tool_name=item.name,
            tool_arguments=args,
        )
        self.add_message(tool_call)

        result = await self.execute_tool(item.name, args)

        tool_output = ToolCallOutput(
            content=result,
            tool_call_id=item.call_id,
            tool_output=result,
        )
        self.add_message(tool_output)

        return await self._generate_completion(depth + 1)

    def _handle_text_response(self, response: Response) -> OpenAIAssistantMessage:
        content = ""
        for output_item in response.output:
            if output_item.type == "message":
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        content += content_item.text

        message = OpenAIAssistantMessage(content=content)
        self.add_message(message)
        return message

    # ==========================================================================
    # Schema Helpers (OpenAI Strict Mode)
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
                required.append(param.name)  # Strict mode requires all params

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
        if schema.json_schema:
            final_schema = self._enforce_strict_mode(schema.json_schema)
            if schema.nullable:
                if isinstance(final_schema.get("type"), str):
                    final_schema["type"] = [final_schema["type"], "null"]
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
        new_schema = schema.copy()
        new_schema.pop("title", None)

        if new_schema.get("type") == "object":
            new_schema["additionalProperties"] = False
            if "properties" in new_schema:
                new_props = {
                    k: self._enforce_strict_mode(v)
                    for k, v in new_schema["properties"].items()
                }
                new_schema["properties"] = new_props
                if "required" not in new_schema:
                    new_schema["required"] = list(new_props.keys())

        if new_schema.get("type") == "array" and "items" in new_schema:
            new_schema["items"] = self._enforce_strict_mode(new_schema["items"])

        if defs := new_schema.get("$defs"):
            new_schema["$defs"] = {
                k: self._enforce_strict_mode(v) for k, v in defs.items()
            }

        return new_schema
