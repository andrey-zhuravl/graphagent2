import uuid
from builtins import list

import requests
from typing import Any, Dict, List, Optional, Type
import time

from src.utils.config import get_config_dict


class McpError(Exception):
    """Исключение для ошибок MCP клиента."""
    pass

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class McpClient:
    """
    Клиент для взаимодействия с MCP сервером (инструменты: filesystem, git, sql и т.д.)
    Пример использования:
        client = McpClient("http://127.0.0.1:8000", path="/mcp")
        tools = client.list_tools()
        resp = client.invoke("filesystem", {"action": "write", "path": "/tmp/test.py", "content": "..."})
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        path: str = "/mcp",
        timeout: float = 10.0,
        retries: int = 1,
        backoff: float = 0.2,
    ):
        self.config = get_config_dict()
        self.base_url = self.config["mcp"]["base_url"]
        self.path = path if path.startswith("/") else f"/{path}"
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff = backoff
        # простой кэш для списка инструментов (TTL можно расширить)
        self._tools_cache: Optional[Dict[str, Any]] = None
        self._tools_cache_ts: float = 0.0
        self._tools_cache_ttl: float = 5.0  # seconds

    # --- Вспомогательные методы ---

    def _url(self) -> str:
        return f"{self.base_url}{self.path}"

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._url()
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.retries:
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                # Попробуем распарсить json в любом случае
                try:
                    data = r.json()
                except ValueError:
                    data = {"raw_text": r.text}
                if 200 <= r.status_code < 300:
                    return {"ok": True, "status_code": r.status_code, "result": data}
                else:
                    return {"ok": False, "status_code": r.status_code, "error": data}
            except requests.RequestException as e:
                last_exc = e
                attempt += 1
                time.sleep(self.backoff * attempt)
        # если все попытки не удались
        raise McpError(f"POST {url} failed after {self.retries + 1} attempts: {last_exc}")

    # --- Основные публичные методы ---

    def dict_tools(self, categories: list[str], use_cache: bool = True) -> dict[str, list[str]]:
        now = time.time()
        if use_cache and self._tools_cache and (now - self._tools_cache_ts) < self._tools_cache_ttl:
            cached = self._tools_cache
            return cached.get("tools", {}) if isinstance(cached, dict) else {}

        tools = {}
        for category in categories:
            tools[category] = self.category_list_tools(category, use_cache=use_cache)
            # кешируем
        self._tools_cache = {"tools": tools}
        self._tools_cache_ts = now
        return tools

    def category_list_tools(self, category: str, use_cache: bool = True) -> List[str]:
        """
        Запросить список инструментов от MCP.
        Возвращает список имён инструментов (если MCP отдаёт структуру - пытаемся её распарсить).
        Формат ответа MCP может быть разный — метод попытался сделать разумное преобразование.
        """

        payload = {"mode": "category", "category": category} # контракт: MCP должен понимать такой формат (адаптируйте если нужно)
        resp = self._post(payload)
        if not resp["ok"]:
            # если MCP вернул ошибку в структуре, пробуем извлечь сообщение
            raise McpError(f"list_tools failed: {resp.get('error')}")
        data = resp["result"]

        # Попытка извлечь список инструментов из разных возможных форматов
        tools = []
        if "tools" in data and isinstance(data["tools"], list):
            tools = [str(t) for t in data["tools"]]
        return tools

    def get_tool_categories(self) -> List[str]:
        """
        Запросить мета-информацию по инструменту (если MCP поддерживает).
        Возвращает словарь с информацией или бросает исключение.
        """
        payload = {"mode": "categories"}
        resp = self._post(payload)
        categories = [cat["name"] for cat in resp["result"].get("categories", [])]
        return categories

    def invoke(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вызываем инструмент на MCP.
        Возвращаем структуру: {"ok": bool, "status_code": int, "result": Any} или бросаем McpError.
        """

        guuid = uuid.uuid4().hex
        payload = {
	       "tool_calls": [{
                "id": guuid,
                "function": {
				"name": tool,
				"arguments": params}
           }]}
        resp = self._post(payload)
        if not resp["ok"]:
            # даём более подробную информацию в исключении
            raise McpError(f"invoke failed for tool={tool}: {resp.get('error')}")
        return resp.get("tool_results")["result"]

    # --- Утилиты управления кэшем / health check ---

    def invalidate_cache(self):
        self._tools_cache = None
        self._tools_cache_ts = 0.0

    def health_check(self) -> bool:
        """
        Простой health check: попробуем получить список инструментов.
        """
        try:
            _ = self.dict_tools(use_cache=False)
            return True
        except Exception:
            return False

if __name__ == "__main__":
    mcp_client = McpClient()
    mcp_client.invalidate_cache()
    categories = mcp_client.get_tool_categories()
    tools_list = mcp_client.dict_tools(categories)
