import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from typing import Optional, Literal, Any
from dataclasses import dataclass

from tinychat.messages.messages import SystemMessage


load_dotenv()


DataType = Literal["string", "number", "integer", "boolean", "array", "object", "null"]


@dataclass
class ToolParameter:
    name: str
    description: str
    data_type: DataType
    enum: Optional[list[str]] = None


@dataclass
class Tool(ABC):
    name: str
    description: str
    parameters: list[ToolParameter]

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """
        This method will be called by the agent to execute tool calls.
        Subclasses must implement it.
        """
        ...


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
    api_key: str = os.getenv("OPENAI_API_KEY")
    max_retries: int = 2
    tools: Optional[list[Tool]] = None
    include_metrics: bool = False
    reasoning_level: Optional[str] = None
    recursion_limit: int = 10
