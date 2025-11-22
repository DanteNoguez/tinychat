import json
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI
from openai.types.responses import Response

from tinychat.messages import LLMMessage
from tinychat.messages.messages import Message
from tinychat.services.llm.models import OpenAILLMConfig, Tool
from tinychat.services.llm.llm import LLMService
from tinychat.services.llm.tools import TypeSchema


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
        self.prompt = self._generate_prompt()
        self.model_name = llm_config.model_name
        self.tools = llm_config.tools
        self.tools_schema = self.create_tools_schema(self.tools) if self.tools else None
        self.tools_by_name = (
            {tool.name: tool for tool in self.tools} if self.tools else None
        )
        self.config = llm_config

    def _type_schema_to_dict(self, schema: TypeSchema) -> dict:
        json_schema = {"type": schema.data_type}

        if schema.description:
            json_schema["description"] = schema.description

        if schema.enum:
            json_schema["enum"] = schema.enum

        if schema.items:
            json_schema["items"] = self._type_schema_to_dict(schema.items)

        if schema.properties:
            json_schema["properties"] = {
                name: self._type_schema_to_dict(prop)
                for name, prop in schema.properties.items()
            }
            json_schema["required"] = schema.required or []
            json_schema["additionalProperties"] = False

        return json_schema

    def create_tools_schema(self, tools: list[Tool]) -> Optional[list[dict]]:
        output = []
        for tool in tools:
            properties = {}
            required_params = []

            for param in tool.parameters:
                param_schema = self._type_schema_to_dict(param.schema)
                param_schema["description"] = param.description
                properties[param.name] = param_schema

                if param.required:
                    required_params.append(param.name)

            output.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_params,
                        "additionalProperties": False,
                    },
                }
            )
        return output

    async def execute_function_calls(
        self, messages: list[dict], response: Response
    ) -> None:
        for item in response.output:
            if item.type == "function_call":
                # Add formatted function call to history
                messages.append(
                    {
                        "id": item.id,
                        "call_id": item.call_id,
                        "type": "function_call",
                        "name": item.name,
                        "arguments": item.arguments,
                    }
                )

                # Execute the function and add output to history
                function_name = item.name
                function_arguments = json.loads(item.arguments)
                if tool := self.tools_by_name.get(function_name):
                    result = await tool.run(**function_arguments)
                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"result": str(result)}),
                        }
                    )
                else:
                    raise ValueError(
                        f"[OpenAIAgent] Tool with name {function_name} not found."
                    )

    async def generate_response(self, messages: list[dict], depth: int = 0) -> None:
        if depth > self.config.recursion_limit:
            raise RuntimeError(
                f"[OpenAIAgent] Recursion limit of {self.config.recursion_limit} reached."
            )

        logger.trace(f"[OpenAIAgent] Input chat history: {messages}")
        response = await self.client.responses.create(
            model=self.model_name,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            tools=self.tools_schema,
            tool_choice="auto" if self.config.tools else None,
            input=messages,
        )
        logger.trace(f"[OpenAIAgent] Generated response: {response}")

        if response.output[0].type == "function_call":
            await self.execute_function_calls(messages, response)
            return await self.generate_response(messages, depth + 1)

        # Format and add assistant message to history
        for output_item in response.output:
            if output_item.type == "message":
                content = ""  # TODO: raise an error if there's no content
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        content += content_item.text
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

    async def reply(self, messages: list[dict]) -> LLMMessage:
        """
        Returns a string with the assistant's response only.
        """
        messages = self.prompt + messages
        await self.generate_response(messages, depth=0)
        return LLMMessage(role="assistant", content=messages[-1]["content"])

    async def reply_with_history(self, messages: list[dict]) -> list[dict]:
        """
        Returns a list of all messages in the conversation history, including input, prompt, tool calls and assistant responses.
        """
        messages = self.prompt + messages
        await self.generate_response(messages, depth=0)
        return messages

    async def _process(self, message: Message) -> LLMMessage:
        return await self.reply([{"role": "user", "content": message.content}])
