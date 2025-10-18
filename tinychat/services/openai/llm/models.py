import os

from typing import Optional
from dataclasses import dataclass


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

    async def run(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class AgentConfig:
    prompt: str
    model_name: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    max_retries: Optional[int] = None
    tools: Optional[list[Tool]] = None


@dataclass
class OpenAIAgentConfig(AgentConfig):
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 300
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    max_retries: int = 2
    tools: Optional[list[Tool]] = None
