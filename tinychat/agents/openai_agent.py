import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from tinychat.agents.models import OpenAIAgentConfig, Tool


class OpenAIAgent:
    def __init__(self, config: OpenAIAgentConfig):
        self.client = AsyncOpenAI(
            api_key=config.api_key, max_retries=config.max_retries
        )
        self.prompt = self.generate_prompt(config.prompt)
        self.model_name = config.model_name
        self.tools = config.tools
        self.tools_schema = self.create_tools_schema(self.tools)
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def generate_prompt(self, prompt: str) -> list[dict]:
        return [{"role": "system", "content": prompt}]

    def create_tools_schema(self, tools: list[Tool]) -> list[dict]:
        if not tools:
            return None
        output = []
        for tool in tools:
            properties = {}
            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.data_type,
                    "description": param.description,
                }
            output.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": [param.name for param in tool.parameters],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return output

    async def generate_completion_async(
        self, messages: list[dict]
    ) -> Optional[ChatCompletionMessage | str]:
        try:
            logger.debug(f"Messages so far: {messages}")
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=self.prompt + messages,
                tools=self.tools_schema,
                tool_choice="auto" if self.tools_schema else None,
            )
            if completion.choices[0].message.tool_calls:
                return completion.choices[0].message
            return completion.choices[0].message.content
        except Exception as e:
            logger.exception(
                f"Error while hitting OpenAI with messages: {messages}",
                exc_info=True,
            )
            raise e

    async def handle_function_call(
        self, messages: list[dict], completion: ChatCompletionMessage
    ) -> list[dict]:
        messages.append(completion)
        logger.debug(f"Function called response: {completion}")
        function_call = completion.tool_calls[0].function
        logger.debug(f"Function call object: {completion}")
        for tool in self.tools:
            if tool.name == function_call.name:
                logger.debug(
                    f"Function arguments: {function_call.arguments} {type(function_call.arguments)}"
                )
                result = await tool.run(**json.loads(function_call.arguments))
                messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": completion.tool_calls[0].id,
                    }
                )
                new_completion = await self.generate_completion_async(messages)
                if isinstance(new_completion, ChatCompletionMessage):
                    messages = await self.handle_function_call(messages, new_completion)
                else:
                    messages.append({"role": "assistant", "content": new_completion})
                logger.debug(f"Messages after function call: {messages}")
                return messages

        raise ValueError(
            f"Tool with name {function_call.name} not found."
        )  # TODO: handle this with a retry

    async def handle_generate_response(self, messages: list[dict]) -> str | list[dict]:
        try:
            response = await self.generate_completion_async(messages)
            logger.debug(f"Generated response: {response}")
            if isinstance(response, ChatCompletionMessage):
                return await self.handle_function_call(messages, response)
            return response
        except Exception as e:
            logger.exception(f"Error in handle generate response: {e}", exc_info=True)
            raise e
