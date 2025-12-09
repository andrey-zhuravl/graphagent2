import json
from typing import Optional, Dict, Any

from src.action import Action
from src.llm.agent_client import AgentClient
from src.mcp.mcp_client import McpClient
from src.memory import Memory, Context, Observation
from src.policy import Policy
from src.tool import Tool

import asyncio
from fastmcp import Client


class Agent:
    def __init__(
            self,
            context: Optional["Context"] = None,
            memory: Optional["Memory"] = None,
            policy: Optional["Policy"] = None,
            tools: Optional[dict[str, Tool]] = None,
    ):
        # Контекст берём через DI; если не передали — создаём пустой.
        self.memory = Memory( history = [])
        self.context = context if context is not None else Context(
            memory=self.memory,
            user_goal=None,
        )
        self.tools = tools or {}
        self.client = AgentClient()
        self.mcp_client = McpClient()
        self.policy = Policy(client=self.client,
                             mcp_client=self.mcp_client)

    def run(self, task: str) -> str:
        self.context.set_task(task)
        self.tools = self.get_tools()
        return self._react_loop()

    def get_tools(self) -> Dict[str, Tool]:
        tools = {}
        categories = self.mcp_client.get_tool_categories()
        categories = ["filesystem", "git", "testing"]  # TODO убрать переделать на динамическое определение
        dict_tools = self.mcp_client.dict_tools(categories, use_cache=True)
        for category in dict_tools:
            for tool in dict_tools[category]:
                try:
                    s = tool.replace("'","\"")
                    a = json.loads(s)
                except Exception as e:
                    print(f"Ошибка чтения инструментов {e}")
                    continue
                tools[a["function"]['name']] = Tool(self.mcp_client, tool)
        return tools

    def _react_loop(self) -> str:
        max_steps = 10
        for step in range(1, max_steps + 1):
            action = self.policy.select_action(self.context)
            if action.tool_name == "done":
                return "Finished"  # Или thought.final_answer если добавишь

            try:
                output = self._execute_action(action)
                success = True
            except Exception as e:
                print(f"Шаг {step} вызвал ошибку - {e}")
                output = {"exception": str(e)}
                success = False

            observation = Observation(action=action,
                                      output=output,
                                      success=success)
            self.context.update(observation)  # Это store + update last

            # Опционально: check goal achieved from observation
            if isinstance(observation.output, dict) and observation.output.get("goal_achieved"):
                return observation.output.get("result", "Goal achieved")

        return "Max steps reached"

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        tool = self.tools[action.tool_name]
        return tool.execute(action.params)
