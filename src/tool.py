from typing import Dict, Any

from src.mcp.mcp_client import McpClient


class Tool:

    def __init__(self, mcp_client: McpClient, tool_name: str):
        """
        :param mcp_client: настроенный экземпляр McpClient
        :param tool_name: точное имя инструмента на стороне MCP (например, "filesystem", "code_generator", "test_runner")
        """
        self.mcp_client = mcp_client
        self.tool_name = tool_name

    def execute(self, params: dict) -> Dict[str, Any]:
        """
        Выполняет инструмент через MCP.
        Возвращает «сырой» результат от MCP (обычно dict) или бросает исключение при ошибке.
        """
        try:
            return self.mcp_client.invoke(self.tool_name, params)
        except Exception as e:
            print(e)
            raise e