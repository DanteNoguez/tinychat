import json
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI
from openai.types.responses import Response

from tinychat.services.openai.llm.models import OpenAIAgentConfig, Tool
from tinychat.messages.messages import SystemMessage


class OpenAIAgent:
    def __init__(self, config: OpenAIAgentConfig, client: Optional[AsyncOpenAI] = None):
        self.client = client or AsyncOpenAI(
            api_key=config.api_key, max_retries=config.max_retries
        )
        self.prompt = self.generate_prompt(config.prompt)
        self.model_name = config.model_name
        self.tools = config.tools
        self.tools_schema = self.create_tools_schema(self.tools) if self.tools else None
        self.config = config
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.recursion_count = 0

    def generate_prompt(self, prompt: SystemMessage) -> list[dict]:
        return [{"role": "system", "content": prompt.content}]

    def create_tools_schema(self, tools: list[Tool]) -> Optional[list[dict]]:
        output = []
        for tool in tools:
            properties = {}
            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.data_type,
                    "description": param.description,
                }
                if tool.enum:
                    properties[param.name]["enum"] = tool.enum
            output.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": [param.name for param in tool.parameters],
                        "additionalProperties": False,
                    },
                }
            )
        return output

    async def generate_response_async(self, messages: list[dict]) -> Response:
        self.recursion_count += 1
        if self.recursion_count > self.config.recursion_limit:
            raise ValueError(
                f"Recursion limit of {self.config.recursion_limit} reached."
            )

        try:
            logger.debug(f"Messages so far: {messages}")
            response = await self.client.responses.create(
                model=self.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=self.prompt + messages,
                tools=self.tools_schema,
                tool_choice="auto" if self.config.tools else None,
            )
            logger.debug(f"Response: {response}")
            return response
        except Exception:
            logger.exception(f"Error in OpenAI's generate_response_async: {messages}")
            raise

    async def handle_function_call(
        self, messages: list[dict], response: Response
    ) -> list[dict]:
        messages.append(response)
        self.recursion_count += 1
        if self.recursion_count > self.config.recursion_limit:
            raise ValueError(
                f"Recursion limit of {self.config.recursion_limit} reached."
            )

        function_call = response.output[0]
        function_name = function_call.name
        function_arguments = json.loads(function_call.arguments)
        if tool := self.tools_by_name.get(function_name):
            result = await tool.run(**function_arguments)
            messages.append(
                {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": function_call.call_id,
                }
            )
            new_response = await self.generate_response_async(messages)
            if new_response.output[0].type == "function_call":
                messages = await self.handle_function_call(messages, new_response)
            else:
                messages.append(
                    {"role": "assistant", "content": new_response.output_text}
                )
            logger.debug(f"Messages after function call: {messages}")
            return messages

        raise ValueError(
            f"Tool with name {function_call.name} not found."
        )  # TODO: handle this with a retry

    async def handle_generate_response(self, messages: list[dict]) -> str:
        try:
            response = await self.generate_response_async(messages)
            logger.debug(f"Generated response: {response}")
            if response.output[0].type == "function_call":
                return await self.handle_function_call(messages, response)
            self.recursion_count = 0
            return response.output_text
        except Exception as e:
            logger.exception(f"Error in handle generate response: {e}", exc_info=True)
            raise e
