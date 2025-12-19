import asyncio
import json
import redis.asyncio as redis
from typing import Any, Optional
from datetime import timedelta


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