import asyncio
import json
import redis.asyncio as redis
from typing import Any, Optional
from datetime import timedelta

from src.memory import Context, Observation


class RedisClient:
    """
    Async Redis client wrapper to handle Redis operations
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        self.redis_url = f"redis://{host}:{port}/{db}"
        if password:
            self.redis_url = f"redis://:{password}@{host}:{port}/{db}"
        self._client = None
    
    async def connect(self):
        """Initialize Redis connection"""
        if not self._client:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client
    
    async def disconnect(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def save_observation(self, key: str, observation_data: dict, ttl: int = 3600) -> bool:
        """
        Save observation to Redis with TTL (default 1 hour = 3600 seconds)
        
        Args:
            key: Redis key to store the observation
            observation_data: Dictionary containing observation data
            ttl: Time to live in seconds (default 1 hour)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            client = await self.connect()
            # Serialize the observation data to JSON
            serialized_data = json.dumps(observation_data, ensure_ascii=False)
            # Store with expiration
            await client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            print(f"Error saving observation to Redis: {e}")
            return False
    
    async def get_observation(self, key: str) -> Optional[dict]:
        """
        Retrieve observation from Redis
        
        Args:
            key: Redis key to retrieve
            
        Returns:
            Optional[dict]: Observation data if found, None otherwise
        """
        try:
            client = await self.connect()
            data = await client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting observation from Redis: {e}")
            return None
    
    async def delete_observation(self, key: str) -> bool:
        """
        Delete observation from Redis
        
        Args:
            key: Redis key to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            client = await self.connect()
            result = await client.delete(key)
            return result > 0
        except Exception as e:
            print(f"Error deleting observation from Redis: {e}")
            return False

        # НОВЫЕ МЕТОДЫ ДЛЯ КОНТЕКСТА
    async def save_context(self, key: str = "agent:context", context: Context = None,
                           ttl: Optional[int] = None) -> bool:
        """
        Save entire agent context to Redis.

        Args:
            key: Redis key for the context (default: "agent:context")
            context: The Context object to save
            ttl: Time to live in seconds (None for permanent storage)

        Returns:
            bool: True if saved successfully
        """
        try:
            client = await self.connect()
            if context is None:
                return False

            # Сериализуем Context в dict
            context_data = {
                "user_goal": context.user_goal,
                "memory": [obs.__dict__ for obs in context.memory] if context.memory else [],
                # Предполагаем, что memory - list[Observation]
                "last_observation": context.last_observation.__dict__ if context.last_observation else None,
                "plan": context.get_plan() if hasattr(context, 'get_plan') else None,
                # Добавь другие поля Context, если они есть (task, etc.)
            }

            serialized_data = json.dumps(context_data, ensure_ascii=False,
                                         default=str)  # default=str для несериализуемых
            if ttl:
                await client.setex(key, ttl, serialized_data)
            else:
                await client.set(key, serialized_data)
            return True
        except Exception as e:
            print(f"Error saving context to Redis: {e}")
            return False

    async def get_context(self, key: str = "agent:context") -> Optional[Context]:
        """
        Retrieve agent context from Redis and deserialize to Context object.

        Args:
            key: Redis key for the context

        Returns:
            Optional[Context]: Deserialized Context if found, None otherwise
        """
        try:
            client = await self.connect()
            data = await client.get(key)
            if not data:
                return None

            context_data = json.loads(data)

            # Десериализуем в Context
            context = Context(
                user_goal=context_data.get("user_goal"),
                memory=[Observation(**obs) for obs in context_data.get("memory", [])] if context_data.get(
                    "memory") else None,
            )
            if context_data.get("last_observation"):
                context.last_observation = Observation(**context_data["last_observation"])
            if context_data.get("plan"):
                context.set_plan(context_data["plan"])  # Предполагаем метод set_plan
            # Добавь другие поля, если нужно

            return context
        except Exception as e:
            print(f"Error getting context from Redis: {e}")
            return None

    async def delete_context(self, key: str = "agent:context") -> bool:
        """
        Delete agent context from Redis.
        """
        try:
            client = await self.connect()
            result = await client.delete(key)
            return result > 0
        except Exception as e:
            print(f"Error deleting context from Redis: {e}")
            return False

# Global Redis client instance
redis_client = RedisClient(host="redis", port=6379)  # Use "redis" as hostname for Docker networking


async def initialize_redis_client():
    """Initialize the global Redis client"""
    global redis_client
    await redis_client.connect()


async def cleanup_redis_client():
    """Cleanup the global Redis client"""
    global redis_client
    await redis_client.disconnect()