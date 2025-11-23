import os
from dotenv import load_dotenv

from typing import Optional, Any
from dataclasses import dataclass, field

from tinychat.messages.messages import Message
from tinychat.services.llm.tools import Tool

load_dotenv()


@dataclass(frozen=True)
class LLMMessage(Message):
    role: str

    def to_openai_format(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass(frozen=True)
class ToolCall(LLMMessage):
    role: str = field(default="tool", init=False)
    tool_call_type: str = field(default="function_call", init=False)
    tool_id: str
    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, Any]

    def to_openai_format(self) -> dict:
        return {
            "id": self.tool_id,
            "call_id": self.tool_call_id,
            "type": self.tool_call_type,
            "name": self.tool_name,
            "arguments": self.tool_arguments,
        }


@dataclass(frozen=True)
class ToolCallOutput(LLMMessage):
    role: str = field(default="tool", init=False)
    tool_output_type: str = field(default="function_call_output", init=False)
    tool_call_id: str
    tool_output: dict[str, Any]

    def to_openai_format(self) -> dict:
        return {
            "type": self.tool_output_type,
            "call_id": self.tool_call_id,
            "output": self.tool_output,
        }


@dataclass
class LLMConfig:
    model_name: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 1000
    enable_retries: bool = True
    max_retries: int = 3
    prompt: Optional[LLMMessage] = None
    tools: Optional[list[Tool]] = None
    base_url: Optional[str] = None
    include_metrics: bool = False
    reasoning_level: Optional[str] = None
    recursion_limit: int = 10


########################################################
# OpenAI specific LLM models
########################################################


@dataclass(frozen=True)
class OpenAIUserMessage(LLMMessage):
    role: str = "user"


@dataclass(frozen=True)
class OpenAIAssistantMessage(LLMMessage):
    role: str = "assistant"
    reasoning: Optional[str] = None


@dataclass(frozen=True)
class OpenAISystemMessage(LLMMessage):
    role: str = "system"


@dataclass
class OpenAILLMConfig(LLMConfig):
    model_name: str = "gpt-4.1"
    temperature: float = 1.0
    api_key: str = os.getenv("OPENAI_API_KEY")
