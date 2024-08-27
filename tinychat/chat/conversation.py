from typing import Dict, List, Optional

from pydantic import BaseModel

from tinychat.agents.models import AgentConfig
from tinychat.agents.openai_agent import OpenAIAgent
from tinychat.rag.chroma import ChromaCollection
from tinychat.rag.models import VectorDBConfig


class Conversation(BaseModel):
    conversation_id: int
    agent_config: AgentConfig
    max_memory_size: int = 20
    vectordb_config: Optional[VectorDBConfig] = None
    memory: List[Dict[str, str]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **config):
        super().__init__(**config)
        if self.memory is None:
            self.memory = []

    @property  # TODO: make this more efficient
    def agent(self) -> OpenAIAgent:
        return OpenAIAgent(self.agent_config)

    @property
    def vectordb(self) -> Optional[ChromaCollection]:
        if self.vectordb_config:
            return ChromaCollection(config=self.vectordb_config)
        return None

    def add_user_message(self, message: str) -> None:
        self.memory.append({"role": "user", "content": message})

    def add_bot_message(self, message: str) -> None:
        self.memory.append({"role": "assistant", "content": message})

    def add_system_message(self, message: str) -> None:
        self.memory.append({"role": "system", "content": message})

    def update_memory(self, messages: str | List[Dict[str, str]]):
        if isinstance(messages, str):
            self.add_bot_message(messages)
        else:
            self.memory.extend(messages)

        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size :]

    async def generate_response(self, message: str) -> str:
        self.add_user_message(message)
        if self.vectordb:
            info = self.vectordb.similarity_search(message)
            self.add_system_message(
                f"If it's useful, utilize the following information to reply:\n{info}"
            )
        new_messages = await self.agent.handle_generate_response(self.memory)
        self.update_memory(new_messages)
        return self.memory[-1].get("content")


class RedisConversation:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RedisConversation is not yet implemented")
