import datetime
import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from loguru import logger
from redis.asyncio import ConnectionPool, Redis

from tinychat.chat.conversation import Conversation

load_dotenv()


class RedisDB:
    def __init__(self):
        self.pool = ConnectionPool(
            host=os.getenv("REDISHOST"),
            port=int(os.getenv("REDISPORT", 6379)),
            username=os.getenv("REDISUSER", None),
            password=os.getenv("REDISPASSWORD", None),
            db=0,
            decode_responses=True,
        )
        self.redis = Redis(connection_pool=self.pool)

    async def _set_with_one_week_expiration(self, *args, **kwargs):
        ONE_WEEK_SECONDS = 60 * 60 * 24 * 7
        return await self.redis.set(*args, **{**kwargs, "ex": ONE_WEEK_SECONDS})

    async def add_redis_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        try:
            key = f"conversation:{conversation_id}"
            await self._set_with_one_week_expiration(
                key, conversation.model_dump_json()
            )
            logger.info(f"{key} added to RedisManager")
        except Exception as e:
            logger.exception(f"Error setting conversation {key} in Redis: {e}")
            raise

    async def get_redis_conversation(
        self, conversation_id: str
    ) -> Optional[Conversation]:
        try:
            key = f"conversation:{conversation_id}"
            conversation = await self.redis.get(key)
            if conversation:
                return Conversation.model_validate_json(conversation)
            logger.warning(f"{key} not found in Redismanager")
            return None
        except Exception as e:
            logger.exception(f"Error getting {key} from RedisManager: {e}")
            raise

    async def remove_redis_conversation(self, conversation_id: str) -> None:
        try:
            key = f"conversation:{conversation_id}"
            await self.redis.delete(key)
            logger.info(f"{key} removed from RedisManager")
        except Exception as e:
            logger.exception(f"Error removing {key} from RedisManager: {e}")
            raise

    async def close(self) -> None:
        await self.pool.disconnect()
