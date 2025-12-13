import json
from datetime import datetime
from typing import Optional, Dict, Any

from src.action import Action
from src.llm.agent_client import AgentClient
from src.mcp_server.mcp_client import McpClient
from src.mcp_server.mcp_streamable_client import McpStreamClient
from src.memory import Memory, Context, Observation, ThoughtManager, Thought
from src.policy import Policy
from src.tool import Tool

import asyncio
from fastmcp import Client


class Agent:
    def __init__(
            self,
            context: Optional["Context"] = None,
            tools: Optional[dict[str, Tool]] = None,
    ):
        # Контекст берём через DI; если не передали — создаём пустой.
        self.tools = None
        self.context = context if context is not None else Context(
            memory=None,
            user_goal=None,
        )
        self.client = AgentClient("llm1")
        self.mcp_client = McpStreamClient()

    async def async_run(self, task: str):
        self.context = Context(
            memory=Memory(

            ),
            user_goal=task,
        )
        self.context.set_task(task)
        async with McpStreamClient("http://localhost:8000/mcp") as client:
            t = await client.list_tools1()
            self.tools = t.tools

            for step in range(1, 999):
                await self.async_step(step, client)

                if self.is_task_complete():
                    break

    async def async_step(self, step: int, client: McpStreamClient):
        situation = self.build_situation()
        # ← Думаем асинхронно (RAG и LLM — await)
        thought = await ThoughtManager(self.context, self.tools).think(situation)
        # ← Может вернуть одно действие или список независимых
        actions: list[Action] = self.thought_to_actions(thought)  # не action, а actions!
        date_time = f"[{datetime.now().strftime('%y-%m-%d %H:%M:%S.%f')[:-3]}]"

        print(
            f"{date_time}:Шаг {step} | Мысль: {thought.reasoning} | Действий: {len(actions) if isinstance(actions, list) else 1}")
        observations: list[Observation] = []
        if len(actions) == 0:
            self.context.update(Observation(
                action=Action(
                    tool_name="empty_action",
                ),
                thought = thought.reasoning,
                success = False
            ))
            return

        if isinstance(actions, list):
            for action in actions:
                if "submit_task" != action.tool_name:
                    result = await action.execute(client)
                else:
                    result = "submit_task"
                if isinstance(result, dict):
                    observations.append(Observation(
                        action = action,
                        output = result,
                        success=True
                    ))
                    print(result)
                if isinstance(result, list):
                    observations.append(Observation(
                        action = action,
                        output = result,
                        success=True
                    ))
                    print(result)
                if isinstance(result, str):
                    observations.append(Observation(
                        action=action,
                        output=result,
                        success=True
                    ))
                    print(result)


            for obs in observations:
                self.context.update(obs)

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
        parts.append(f"{submit_task}\n")
        parts.append("]")

        if self.context.get_plan():
            parts.append(f"ПЛАН: {self.context.get_plan()}")

        return "\n".join(parts)

    def thought_to_actions(self, thought: Thought) -> list[Action]:
        """
        Превращает Thought → Action или список Action.
        Главная точка гибкости: здесь решаем, что делать дальше.
        """
        # ------------------------------------------------------------------
        # 1. Специальные обработчики по source (самые быстрые и надёжные)
        # ------------------------------------------------------------------
        if thought.source == "llm" and thought.action_plan:
            return self._handle_llm_thought(thought)
        #
        # if thought.source.startswith("template_"):
        #     return self._handle_template_thought(thought)
        #
        # if thought.source == "rag_adapted" and thought.action_plan:
        #     return self._handle_rag_thought(thought)


        # ------------------------------------------------------------------
        # 2. Универсальный парсер action_plan (если source не дал точного ответа)
        # ------------------------------------------------------------------
        if thought.action_plan:
            self.context.set_plan(thought.action_plan)
            parsed = self._parse_action_plan(thought.action_plan)
            if parsed:
                return parsed

        # ------------------------------------------------------------------
        # 3. Фолбэк: если ничего не распарсили — одно простое действие
        # ------------------------------------------------------------------
        return None

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

    def _handle_llm_thought(self, thought: Thought) -> list[Action]:
        """LLM может сказать "сделай параллельно" — поддерживаем это"""
        plan = thought.action_plan

        # # Явный параллелизм
        # if any(word in plan for word in ["параллельно", "одновременно", "все сразу", "сначала всё"]):
        #     return self._extract_parallel_actions(thought.action_plan)

        # Обычный последовательный план
        action_list: dict[Action] = []
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
        return action_list

    def _parse_action_plan(self, plan: str) -> Action | list[Action] | None:
        """Универсальный парсер: ищет команды в тексте"""
        lines = [l.strip() for l in plan.split('\n') if l.strip() and not l.startswith('#')]

        # Если в плане явно "параллельно" — ищем несколько команд
        if any("параллельно" in l.lower() for l in lines):
            return self._extract_parallel_actions(plan)

        # Иначе — берём первую осмысленную строку
        for line in lines:
            action = self._line_to_action(line)
            if action:
                return action

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        tool = self.tools[action.tool_name]
        return tool.execute(self.mcp_client, action.params)

    def _extract_first_action_from_plan(self, action_plan):
        pass

    def is_task_complete(self) -> bool:
        if self.context.last_observation.action.tool_name == "submit_task":
            return True
        if self.context.last_observation.action.tool_name == "empty_action":
            return True
        return False
