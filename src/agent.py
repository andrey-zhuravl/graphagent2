from datetime import datetime
from typing import Optional, Dict, Any

from src.action import Action
from src.llm.agent_client import AgentClient
from src.mcp_server.mcp_streamable_client import McpStreamClient
from src.memory import Context, Observation, Thought
from src.thinking.thought_manager import ThoughtManager
from src.tool import Tool

import asyncio

class Agent:
    def __init__(
            self,
            context: Optional["Context"] = None,
            tools: Optional[dict[str, Tool]] = None,
    ):
        # Контекст берём через DI; если не передали — создаём пустой.
        self.mcp_client = None
        self.tools = None
        self.context = context if context is not None else Context(
            memory=None,
            user_goal=None,
        )
        self.client = AgentClient("llm1")
        self.thought_manager = ThoughtManager(context = self.context)

    async def async_run(self, task: str):
        self.context = Context(
            user_goal=task,
        )
        self.context.set_task(task)
        async with McpStreamClient() as client:
            self.mcp_client = client
            self.tools = await self.mcp_client.list_tools()

            for step in range(1, 999):
                await self.async_step(step)

                if self.is_task_complete():
                    break

    async def async_step(self, step: int):
        situation: str = self.build_situation()
        # ← Думаем асинхронно (RAG и LLM — await)
        thought: Thought = await self.thought_manager.think(self.tools, situation)
        # ← Может вернуть одно действие или список независимых
        actions: list[Action] = self.thought_to_actions(thought)  # не action, а actions!
        date_time = f"[{datetime.now().strftime('%y-%m-%d %H:%M:%S.%f')[:-3]}]"

        print(f"{date_time}:Шаг {step} | Мысль: {thought.reasoning} | Действий: {len(actions) if isinstance(actions, list) else 1}")
        observations: list[Observation] = await self.actions_to_observations(actions)
        self.context.update(observations)

        # === Сохранение в долгосрочную память ===
        await self.thought_manager.rag_thought_manager.save_to_rag(thought)

    async def actions_to_observations(self, actions) -> list[Observation]:
        observations: list[Observation] = []
        if isinstance(actions, list):
            for action in actions:
                if ("submit_task" == action.tool_name
                        or "think_along" == action.tool_name
                        or "empty_action" == action.tool_name
                        or "error_llm" == action.tool_name):
                    result = action.tool_name
                else:
                    result = await action.execute(self.mcp_client)

                observations.append(Observation(
                    action=action,
                    output=result,
                    success=True
                ))
                print(result)
        return observations


    def build_situation(self) -> str:
        parts = []

        # 1. Главная цель — всегда наверху
        if self.context.user_goal:
            parts.append(f"ЦЕЛЬ: {self.context.user_goal}")

        # 2. Последнее действие и его результат
        if self.context.last_observation:
            obs = self.context.last_observation
            status = "УСПЕХ" if obs.success else "ОШИБКА"
            parts.append(f"ПОСЛЕДНЕЕ ДЕЙСТВИЕ ({status}): {obs.action.tool_name}")
            if not obs.success:
                error = obs.output.strip().split('\n')[-1]  # последняя строка ошибки
                parts.append(f"ОШИБКА: {error}")

        # 3. Краткая история (последние 3–5 шагов)
        parts.append("История:")
        parts.append(self.context.format_recent_history())

        # 4. Текущая среда (очень важно!)
        # для специальных задач надо делать специально

        # 5. MCP инструменты
        parts.append(f"MCP инструменты:\n[")
        for tool in self.tools:
            t = self.mcp_client.convert_mcp_tool_to_openai_format(tool)
            parts.append(f"{t}\n")
        submit_task = {'type': 'function',
                      'function': {'name': 'submit_task',
                                   'parameters': {}
                                   }}
        think_along = {'type': 'function',
                      'function': {'name': 'think_along',
                                   'parameters': {}
                                   }}
        parts.append(f"{submit_task}\n")
        parts.append(f"{think_along}\n")
        parts.append("]")

        if self.context.get_plan():
            parts.append(f"ПЛАН: {self.context.get_plan()}")

        return "\n".join(parts)

    def thought_to_actions(self, thought: Thought) -> list[Action]:
        """
        Превращает Thought → Action или список Action.
        Главная точка гибкости: здесь решаем, что делать дальше.
        """
        action_list: list[Action] = []
        if thought.source == "llm" and thought.action_plan:
            for tool in thought.action_plan:
                if "parameters" in tool:
                    action_list.append(Action(
                        tool_name=tool["tool"],
                        params=tool["parameters"]
                    ))
                else:
                    action_list.append(Action(
                        tool_name=tool["tool"]
                    ))
        else:
            action_list.append(Action(
                        tool_name="empty_action",
                    )
                )
        return action_list
        #
        # if thought.source.startswith("template_"):
        #     return self._handle_template_thought(thought)
        #
        # if thought.source == "rag_adapted" and thought.action_plan:
        #     return self._handle_rag_thought(thought)

    def _handle_template_thought(self, thought: Thought) -> Action | list[Action]:
        """Шаблоны — самые точные, им доверяем полностью"""
        if thought.source == "template_decomposition":
            # Пример: первый шаг из декомпозиции
            first_line = thought.action_plan.split('\n')[0].lower()
            if "анализ" in first_line or "статистик" in first_line:
                return Action(
                    action_type="execute_bash",
                    command="find . -name '*.py' | wc -l && git status && python --version",
                    description="Сбор статистики по проекту"
                )

        if thought.source == "template_error_handling":
            # Пример: восстановление после ошибки
            last_obs = self.context.last_observation
            if last_obs and "Permission denied" in last_obs.output:
                return Action(
                    action_type="execute_bash",
                    command="sudo pip install -r requirements.txt || pip install --user -r requirements.txt",
                    description="Обход ошибки прав"
                )

        # По умолчанию — просто берём первый пункт плана
        return self._extract_first_action_from_plan(thought.action_plan)

    def _handle_rag_thought(self, thought: Thought) -> Action | list[Action]:
        """RAG уже дал адаптированный план — просто выполняем по шагам"""
        actions = []
        for line in thought.action_plan.split('\n')[:10]:  # не больше 10
            if line.strip() and not line.startswith('#'):
                action = self._line_to_action(line)
                if action:
                    actions.append(action)
        return actions if actions else self._extract_first_action_from_plan(thought.action_plan)

    def _extract_first_action_from_plan(self, action_plan):
        pass

    def is_task_complete(self) -> bool:
        if self.context.last_observation.action.tool_name == "submit_task":
            return True
        if self.context.last_observation.action.tool_name == "empty_action":
            return True
        if self.context.last_observation.action.tool_name == "think_along":
            return False
        if self.context.last_observation.action.tool_name == "error_llm":
            return False
        return False
