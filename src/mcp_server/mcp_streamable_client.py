# mcp_client.py
import asyncio
import pprint
import types
from typing import Any, Dict
from mcp import ClientSession, ListToolsResult, Tool
from mcp.client.streamable_http import streamablehttp_client


class McpStreamClient:
    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        self.server_url = server_url
        self._inner_session = None  # это будет ClientSession
        self._transport = None  # это будет streamablehttp_client(...)
        self._read = None
        self._write = None

    async def __aenter__(self):
        # Входим в транспорт
        self._transport = streamablehttp_client(self.server_url)
        self._read, self._write, _ = await self._transport.__aenter__()

        # Входим в ClientSession
        self._inner_session = ClientSession(self._read, self._write)
        await self._inner_session.__aenter__()

        # Теперь инициализируем
        await self._inner_session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._inner_session:
            await self._inner_session.__aexit__(exc_type, exc_val, exc_tb)
        if self._transport:
            await self._transport.__aexit__(exc_type, exc_val, exc_tb)

    async def list_tools1(self) -> ListToolsResult:
        return await self._inner_session.list_tools()

    async def list_tools(self) -> list[Tool]:
        return await self._inner_session.list_tools().tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        result = await self._inner_session.call_tool(name, args)

        if result.structuredContent is not None:
            return result.structuredContent

        if result.content and hasattr(result.content[0], 'text'):
            return result.content

        return None

    def convert_mcp_tool_to_openai_format(self, tool_dict: Tool) -> Dict:
        """
        Преобразует инструмент в формате FastMCP (MCP-spec) в формат OpenAI function calling.
        """
        return {
            "type": "function",
            "function": {
                "name": tool_dict.name,
                "description": tool_dict.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param_name: {
                            "type": param_meta["type"],
                            "description": param_meta.get("description", "")  # может быть пустым
                        }
                        for param_name, param_meta in tool_dict.inputSchema["properties"].items()
                    },
                    "required": tool_dict.inputSchema.get("required", [])
                }
            }
        }

async def main():
    # async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
    #     async with ClientSession(read, write) as session:
    #         await session.initialize()
    #         tools = await session.list_tools()
    #         print("Tools:", [t.name for t in tools.tools])
    # 1. Создать клиент
    client = McpStreamClient("http://localhost:8000/mcp")

    # 2. Явно подключиться
    async with McpStreamClient("http://localhost:8000/mcp") as client:
        tools = await client.list_tools()

        ss = convert_mcp_tool_to_openai_format(tools.tools[2])
        print("Tools:", tools.tools[2])

        res = await client.call_tool("read_file", {"a": 2, "b": 3})
        print("Result:", res)
        # Вызов инструмента
        result = await client.call_tool("add", {"a": 5, "b": 7})
        print("Результат add(5, 7):", result)

        # Чтение ресурса (если есть)
        try:
            greeting = await client.read_resource("greeting://Alice")
            print("Приветствие:", greeting)
        except Exception as e:
            print("Ресурс недоступен:", e)

def convert_mcp_tool_to_openai_format(tool_dict):
    """
    Преобразует инструмент в формате FastMCP (MCP-spec) в формат OpenAI function calling.
    """
    return {
        "type": "function",
        "function": {
            "name": tool_dict.name,
            "description": tool_dict.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param_name: {
                        "type": param_meta["type"],
                        "description": param_meta.get("description", "")  # может быть пустым
                    }
                    for param_name, param_meta in tool_dict.inputSchema["properties"].items()
                },
                "required": tool_dict.inputSchema.get("required", [])
            }
        }
    }

if __name__ == "__main__":
    asyncio.run(main())