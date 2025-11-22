import os
from dotenv import load_dotenv

from typing import Optional
from dataclasses import dataclass

from tinychat.messages.messages import LLMMessage
from tinychat.services.llm.tools import Tool

load_dotenv()


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


@dataclass
class OpenAILLMConfig(LLMConfig):
    model_name: str = "gpt-4.1"
    temperature: float = 1.0
    api_key: str = os.getenv("OPENAI_API_KEY")


@dataclass(frozen=True)
class OpenAIUserMessage(LLMMessage):
    role: str = "user"


@dataclass(frozen=True)
class OpenAIAssistantMessage(LLMMessage):
    role: str = "assistant"


@dataclass(frozen=True)
class OpenAISystemMessage(LLMMessage):
    role: str = "system"
