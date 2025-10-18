import os

from typing import Optional
from dataclasses import dataclass

from tinychat.messages.messages import SystemMessage


@dataclass
class ToolParameter:
    name: str
    description: str
    data_type: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter]
    enum: Optional[list[str]] = None

    async def run(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class AgentConfig:
    prompt: SystemMessage
    model_name: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    max_retries: Optional[int] = None
    tools: Optional[list[Tool]] = None


@dataclass
class OpenAIAgentConfig(AgentConfig):
    model_name: str = "gpt-4.1"
    temperature: float = 1.0  # Default compatibility with GPT-5 models
    max_tokens: int = 500
    api_key: str = os.environ.get("OPENAI_API_KEY")
    max_retries: int = 2
    tools: Optional[list[Tool]] = None
    include_metrics: bool = False
    reasoning_level: Optional[str] = None
    recursion_limit: int = 10
