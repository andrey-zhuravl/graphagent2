import httpx
from typing import List, Dict, Any, Optional, Iterable

import numpy as np
from graphiti_core.embedder import EmbedderClient


class CustomEmbeddingClient(EmbedderClient):
    """Адаптер для эмбеддингов на http://192.168.1.12:1234"""
    def __init__(self, base_url: str = "http://192.168.1.12:1234"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def create_batch(self, texts):
        return [np.zeros(1536).tolist() for _ in texts]

    async def create(self,
                     input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
                     ) -> list[float]:
        """
                Формат OpenAI Embeddings API
                """
        payload = {
            "input": input_data,
            "model": "text-embedding-nomic-embed-text-v1.5"
        }

        response = await self.client.post(
            f"{self.base_url}/v1/embeddings",  # или /embeddings
            json=payload
        )
        return response.json()["data"][0]["embedding"]

    async def close(self):
        await self.client.aclose()