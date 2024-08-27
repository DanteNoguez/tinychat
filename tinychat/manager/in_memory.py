from typing import Optional

from tinychat.agents.registry import AgentConfigRegistry
from tinychat.chat.conversation import Conversation
from tinychat.rag.registry import VectorDBRegistry


class InMemoryConversationsManager:
    def __init__(
        self,
        agent_registry: AgentConfigRegistry,
        vectordb_registry: Optional[VectorDBRegistry] = None,
        max_size: int = 100,
    ):
        self.agent_registry = agent_registry
        self.vectordb_registry = vectordb_registry
        self.max_size = max_size
        self.conversations: dict[int, Conversation] = {}
        self.access_order: dict[int, str] = {}  # key: conversation_id, value: next_id
        self.head_id: Optional[int] = None
        self.tail_id: Optional[int] = None

    def get_conversation_id(self, identifier: str) -> int:
        return hash(identifier)

    def get_conversation(self, identifier: str) -> Conversation:
        conversation_id = self.get_conversation_id(identifier)
        if conversation_id in self.conversations:
            self._move_to_front(conversation_id)
            return self.conversations[conversation_id]
        raise ValueError(f"Conversation with ID {conversation_id} not found.")

    def remove_conversation(self, identifier: str) -> None:
        conversation_id = self.get_conversation_id(identifier)
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            del self.access_order[conversation_id]
        raise ValueError(f"Conversation with ID {conversation_id} not found.")

    def _get_or_create_conversation(
        self, conversation_id: str, request_type: str
    ) -> Conversation:
        if conversation_id in self.conversations:
            self._move_to_front(conversation_id)
            return self.conversations[conversation_id]

        if len(self.conversations) >= self.max_size:
            self._remove_oldest()

        agent_config = self.agent_registry.get_config(request_type)
        vectordb_config = (
            self.vectordb_registry.get_config(request_type)
            if self.vectordb_registry
            else None
        )
        new_conversation = Conversation(
            agent_config=agent_config,
            vectordb_config=vectordb_config,
            conversation_id=conversation_id,
        )
        self.conversations[conversation_id] = new_conversation
        self._add_to_front(conversation_id)

        return new_conversation

    def _move_to_front(self, conversation_id: int):
        if conversation_id == self.head_id:
            return

        prev_id = None
        current_id = self.head_id
        while current_id != conversation_id:
            prev_id = current_id
            current_id = self.access_order[current_id]

        if prev_id:
            self.access_order[prev_id] = self.access_order[conversation_id]
        else:
            self.head_id = self.access_order[conversation_id]

        if conversation_id == self.tail_id:
            self.tail_id = prev_id

        self.access_order[conversation_id] = self.head_id
        self.head_id = conversation_id

    def _add_to_front(self, conversation_id: int):
        if not self.head_id:
            self.head_id = self.tail_id = conversation_id
            self.access_order[conversation_id] = None
        else:
            self.access_order[conversation_id] = self.head_id
            self.head_id = conversation_id

    def _remove_oldest(self):
        if not self.tail_id:
            return

        oldest_id = self.tail_id
        self.tail_id = next(
            (k for k, v in self.access_order.items() if v == oldest_id), None
        )
        if self.tail_id:
            self.access_order[self.tail_id] = None
        else:
            self.head_id = None

        del self.conversations[oldest_id]
        del self.access_order[oldest_id]

    async def handle_incoming_message(
        self,
        identifier: str,
        request_type: str,
        message: str,
    ):
        conversation = self._get_or_create_conversation(
            conversation_id=self.get_conversation_id(identifier),
            request_type=request_type,
        )
        return await conversation.generate_response(message)
