from enum import Enum

from pydantic import BaseModel


class MemoryRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    content: str


class AgentConfigType(Enum):
    DEFAULT = "whatsapp"
    PANDAS = "pandas"
