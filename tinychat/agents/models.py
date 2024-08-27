import os

from dotenv import load_dotenv

load_dotenv()

from typing import Optional

from pydantic import BaseModel


class ToolParameter(BaseModel):
    name: str
    description: str
    data_type: str


class Tool(BaseModel):
    name: str
    description: str
    parameters: list[ToolParameter]

    async def run(self, *args, **kwargs):
        raise NotImplementedError


class AgentConfig(BaseModel):
    prompt: str
    model_name: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    max_retries: Optional[int] = None
    tools: Optional[list[Tool]] = None

    # class Config:
    #     arbitrary_types_allowed=True


class OpenAIAgentConfig(AgentConfig):
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 300
    api_key: str = os.environ["OPENAI_API_KEY"]
    max_retries: int = 2
    tools: Optional[list[Tool]] = None


class PandasAgentConfig(OpenAIAgentConfig):
    df_dict: dict
