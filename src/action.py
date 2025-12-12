from dataclasses import dataclass, field

from src.mcp_server.mcp_streamable_client import McpStreamClient


@dataclass
class Action:
    def __init__(self, tool_name: str,  # имя инструмента, который нужно вызвать
                 params: dict) -> None:
        self.tool_name = tool_name
        self.params = params

    async def execute(self, client: McpStreamClient):
        return await client.call_tool(self.tool_name, self.params)

