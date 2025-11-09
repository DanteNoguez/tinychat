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
        self.tools_by_name = (
            {tool.name: tool for tool in self.tools} if self.tools else None
        )

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
                if param.enum:
                    properties[param.name]["enum"] = param.enum
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

    async def generate_response_async(
        self, messages: list[dict], depth: int = 0
    ) -> None:
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
            return await self.generate_response_async(messages, depth + 1)

        # Format and add assistant message to history
        for output_item in response.output:
            if output_item.type == "message":
                content = ""  # TODO: raise an error fi there's no content
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        content += content_item.text
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

    async def reply(self, messages: list[dict]) -> str:
        """
        Returns a string with the assistant's response only.
        """
        messages = self.prompt + messages
        await self.generate_response_async(messages, depth=0)
        return messages[-1]["content"]

    async def reply_with_history(self, messages: list[dict]) -> list[dict]:
        """
        Returns a list of all messages in the conversation history, including input, prompt, tool calls and assistant responses.
        """
        messages = self.prompt + messages
        await self.generate_response_async(messages, depth=0)

        # Return complete conversation history
        return messages
