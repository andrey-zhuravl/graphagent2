from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from mcp import Tool

from src.action import Action
from src.artifacts.policy import ArtifactPolicy
from src.artifacts.store import ArtifactStore
from src.artifacts.tools import get_retrieve_artifact_tool_schema, retrieve_artifact, get_retrieve_artifact_mini_tool_schema
from src.llm.agent_client import AgentClient
from src.mcp_server.mcp_streamable_client import McpStreamClient
from src.memory import Context, Observation, Thought
from src.task import CompactGoal
from src.thinking.thought_manager import ThoughtManager

import asyncio
from src.utils.redis_client import redis_client

class Agent:
    def __init__(
            self,
            context: Optional["Context"] = None,
            tools: Optional[dict[str, Tool]] = None,
            artifact_store: Optional[ArtifactStore] = None,
    ):
        # Контекст берём через DI; если не передали — создаём пустой.
        self.mcp_client = None
        self.tools: dict[str, Tool] = {}
        self.context = context if context is not None else Context(
            memory=None,
            user_goal=None,
        )
        self.thought_manager = ThoughtManager(context = self.context)
        self.artifact_store = artifact_store or ArtifactStore(Path(".artifacts"))
        self.artifact_policy = ArtifactPolicy(self.artifact_store)
        self.custom_tool_schema = get_retrieve_artifact_tool_schema()
        self.custom_tool_mini_schema = get_retrieve_artifact_mini_tool_schema()

    async def async_run(self, task: str):
        self.context.set_task(task)
        async with McpStreamClient() as client:
            self.mcp_client = client
            ts = await self.mcp_client.mpc_list_tools()
            for tool in ts:
                self.tools[tool.name] = tool
            await self.begin_step()

            for step in range(1, 999):
                await self.async_step(step)

                if self.is_task_complete():
                    await self.end_step()
                    break

    async def begin_step(self):
        mini_format_tools = await self.get_mini_format_tools()
        await self.thought_manager.pre_think(mini_format_tools)

    async def get_mini_format_tools(self):
        parts = []
        # 2. MCP инструменты
        parts.append(f"MCP инструменты:\n[")
        for tool_name, tool in self.tools.items():
            t = self.mcp_client.convert_mcp_tool_to_openai_mini_format(tool)
            parts.append(f"{t}\n")
        parts.append(f"{self.custom_tool_mini_schema}\n")
        parts.append("]")
        mini_format_tools = "\n".join(parts)
        return mini_format_tools

    async def end_step(self):
        mini_format_tools = await self.get_mini_format_tools()
        observations = await self.get_observations()
        await self.thought_manager.post_think(mini_format_tools, observations)

    async def async_step(self, step: int):
        situation: str = self.build_situation(step)

        # ← Думаем асинхронно (RAG и LLM — await)
        thought: Thought = await self.thought_manager.think(self.tools,
                                                            situation,
                                                            self.context.compact_goal.rag_queries,
                                                            step=step)
        self.context.state_update = thought.state_update
        # ← Может вернуть одно действие или список независимых
        actions: list[Action] = self.thought_to_actions(thought)  # не action, а actions!
        date_time = f"[{datetime.now().strftime('%y-%m-%d %H:%M:%S.%f')[:-3]}]"

        print(f"{date_time}:Шаг {step} | Мысль: {thought.reasoning} | Действий: {len(actions) if isinstance(actions, list) else 1}")
        observations: list[Observation] = await self.actions_to_observations(actions, step)
        self.context.update(observations)

        # === Сохранение в долгосрочную память ===
        await self.thought_manager.rag_thought_manager.save_to_rag(thought)
        
        # === Сохранение всех наблюдений в Redis на 1 час ===
        # await self.save_observations_to_redis(observations)

    async def actions_to_observations(self, actions: list[Action], step: int) -> list[Observation]:
        observations: list[Observation] = []
        if isinstance(actions, list):
            for action in actions:
                params = action.params or {}
                if ("submit_task" == action.tool_name
                        or "think_along" == action.tool_name
                        or "empty_action" == action.tool_name
                        or "error_llm" == action.tool_name):
                    result = action.tool_name
                elif action.tool_name == "retrieve_artifact":
                    result = retrieve_artifact(
                        self.artifact_store,
                        params.get("ref"),
                        start_line=params.get("start_line"),
                        end_line=params.get("end_line"),
                    )
                else:
                    result = await action.execute(self.mcp_client)

                output, output_short, artifact_refs = self.artifact_policy.maybe_persist(result)

                observations.append(Observation(
                    action=action,
                    output_short=output_short,
                    output=output,
                    success=True,
                    step=step,
                    artifacts=artifact_refs,
                ))
                print(output_short if output_short is not None else result)
        return observations

    async def save_observations_to_redis(self, observations: list[Observation]):
        """
        Save all observations to Redis with a TTL of 1 hour (3600 seconds)
        Each observation gets a unique key based on timestamp and index
        """
        try:
            for i, observation in enumerate(observations):
                # Create a unique key for each observation
                key = f"observation:{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                
                # Convert observation to dictionary for storage
                observation_data = {
                    "action": {
                        "tool_name": observation.action.tool_name,
                        "params": observation.action.params if hasattr(observation.action, 'params') else {}
                    },
                    "output": str(observation.output),
                    "success": observation.success,
                    "artifacts": [str(a) for a in observation.artifacts],
                }
                
                # Save to Redis with 1 hour TTL (3600 seconds)
                await redis_client.save_observation(key, observation_data, ttl=3600)
                
                print(f"Saved observation to Redis with key: {key}")
        except Exception as e:
            print(f"Error saving observations to Redis: {e}")


    def build_situation(self, step: int) -> str:
        parts = []

        # 1. Главная цель — всегда наверху

        parts.append(f"Рассуждение № {step}")
        parts.append(f"ЦЕЛЬ: {self.context.compact_goal.to_json()}")

        # # 2. Последнее действие и его результат
        # if self.context.last_observation:
        #     obs = self.context.last_observation
        #     status = "УСПЕХ" if obs.success else "ОШИБКА"
        #     parts.append(f"ПОСЛЕДНЕЕ ДЕЙСТВИЕ ({status}): {obs.action.tool_name}")
        #     if not obs.success:
        #         error = obs.output.strip().split('\n')[-1]  # последняя строка ошибки
        #         parts.append(f"ОШИБКА: {error}")

        # 3. Краткая история (последние 3–5 шагов)
        parts.append("История:")
        parts.append(self.context.format_recent_history())

        # 4. Текущая среда (очень важно!)
        # для специальных задач надо делать специально

        # 5. MCP инструменты
        parts.append(f"MCP инструменты:\n[")
        for tool_name in self.context.compact_goal.tool_name_list:
            t = self.mcp_client.convert_mcp_tool_to_openai_format(self.tools[tool_name])
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
        parts.append(f"{self.custom_tool_schema}\n")
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
        try:
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
        except Exception as e:
            print(f"Error thought actions: {e}")
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

    async def get_observations(self) -> str:
        task_history = []
        for obs in self.context.memory.history:
            task_history.append("observation:{}".format(obs.to_json()))
        return "\n".join(task_history)

