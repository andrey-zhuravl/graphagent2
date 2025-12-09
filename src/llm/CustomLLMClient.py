import typing
from abc import ABC
import json
import httpx
from typing import List, Dict, Any, Optional

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize, LLMConfig
from graphiti_core.prompts import Message
from openai.cli._models import BaseModel


class CustomLLMClient(LLMClient):
    """Адаптер для вашей локальной LLM по адресу http://192.168.1.12:8000"""

    def __init__(self,
                 url: str = None,
                 model: str = None):
        config = LLMConfig()
        config.base_url = url
        config.model_name = model
        super().__init__(config)
        self.client = httpx.AsyncClient(
            timeout=60
        )
        self.model = config.model

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Формат, который ожидает Graphiti
        """

        def extract_role(m):
            # возможные имена поля роли
            for a in ("role", "speaker", "author", "sender"):
                if isinstance(m, dict) and a in m:
                    return m[a]
                if hasattr(m, a):
                    return getattr(m, a)
            return "user"

        def extract_content(m):
            # если уже словарь с нужным ключом
            if isinstance(m, dict):
                for a in ("content", "text", "message", "body", "value"):
                    if a in m and m[a] is not None:
                        return m[a]
                # если dict, но нет нужных полей — попробуем stringify
                return str(m)

            # для pydantic / dataclass / объёкта
            for a in ("text", "content", "message", "body", "value", "content_text"):
                if hasattr(m, a):
                    val = getattr(m, a)
                    # если это pydantic Field (BaseModel) — привести к строке
                    try:
                        if isinstance(val, (list, dict)):
                            return val
                    except Exception:
                        pass
                    return val

            # pydantic v2: model_dump()
            if hasattr(m, "model_dump"):
                try:
                    dumped = m.model_dump()
                    # Попробуем найти текст в дампе
                    for a in ("content", "text", "message", "body"):
                        if a in dumped and dumped[a] is not None:
                            return dumped[a]
                    return str(dumped)
                except Exception:
                    pass

            # dataclass / object -> __dict__
            if hasattr(m, "__dict__") and m.__dict__:
                for a in ("content", "text", "message", "body"):
                    if a in m.__dict__ and m.__dict__[a] is not None:
                        return m.__dict__[a]
                return str(m.__dict__)

            # fallback — строковое представление
            return str(m)

        payload_messages = []
        payload_messages.append({
            "role": "user",
            "content": "Извлеки сущности из текста и верни только JSON в формате. Никаких схем, описаний или других данных."
        })
        for m in messages:
            role = extract_role(m)
            content = extract_content(m)

            if content is None:
                raise ValueError(f"Cannot extract content from message object: {m!r}")

            payload_messages.append({
                "role": role,
                "content": content
            })

        # Конвертируем сообщения в формат вашего API
        payload = {
            "messages": payload_messages,
            "model": self.model,
            "temperature": self.config.temperature,
            "max_tokens": 2000
        }

        response = await self.client.post(
            f"{self.config.base_url}/v1/chat/completions",  # или /chat/completions, смотрите ваш API
            json=payload
        )
        #print(response.json())
        print(response.json()["choices"][0]["message"]["content"])
        return json.loads(response.json()["choices"][0]["message"]["content"])
