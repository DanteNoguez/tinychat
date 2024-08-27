from typing import Optional

from tinychat.agents.registry import AgentConfigRegistry
from tinychat.chat.conversation import Conversation
from tinychat.manager.in_memory import InMemoryConversationsManager
from tinychat.manager.redis_db import RedisDB
from tinychat.rag.registry import VectorDBRegistry


class RedisConversationsManager(RedisDB, InMemoryConversationsManager):
    def __init__(
        self,
        agent_registry: AgentConfigRegistry,
        vectordb_registry: Optional[VectorDBRegistry] = None,
        max_size: int = 100,
    ):
        self.agent_registry = agent_registry
        self.vectordb_registry = vectordb_registry
        self.max_size = max_size
        self.access_order: list[str] = []

    async def get_conversation(self, identifier: str) -> Conversation:
        conversation_id = self.get_conversation_id(identifier)
        if conversation_id in self.conversations:
            self._move_to_front(conversation_id)
            return self.conversations[conversation_id]

        conversation_data = await self.get_redis_conversation(conversation_id)
        if conversation_data:
            self.conversations[conversation_id] = conversation_data
            self._add_to_front(conversation_id, conversation_data)
            return conversation_data

        raise ValueError(f"Conversation with ID {identifier} not found.")

    async def remove_conversation(self, identifier: str) -> None:
        conversation_id = self.get_conversation_id(identifier)
        await self.remove_redis_conversation(conversation_id)
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            del self.access_order[conversation_id]
        raise ValueError(f"Conversation with ID {conversation_id} not found.")

    async def _get_or_create_conversation(
        self, conversation_id: str, request_type: str
    ) -> Conversation:
        try:
            return await self.get_conversation(conversation_id)
        except ValueError:
            if len(self.conversations) >= self.max_size:
                self._remove_oldest()
            agent_config = self.agent_registry.get_config(request_type)
            vectordb_config = (
                self.vectordb_registry.get_collection(request_type)
                if self.vectordb_registry
                else None
            )
            new_conversation = Conversation(
                agent_config=agent_config,
                vectordb_config=vectordb_config,
                conversation_id=conversation_id,
            )
            await self.add_redis_conversation(conversation_id, new_conversation)
            self.conversations[conversation_id] = new_conversation
            self._add_to_front(conversation_id)
            return new_conversation

    async def handle_incoming_message(
        self,
        identifier: str,
        request_type: str,
        message: str,
    ):
        conversation_id = self.get_conversation_id(identifier)
        conversation = await self._get_or_create_conversation(
            conversation_id, request_type
        )
        response = await conversation.generate_response(message)
        await self.add_redis_conversation(conversation_id, conversation)
        return response
