import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

from mcp import Tool

from src.action import Action
from src.llm.agent_client import AgentClient
from src.rag.agent_embeding import Embedder


@dataclass
class Observation:
    def __init__(self,
                 action: Action,  # к какому действию относится
                 output: any,  # результат инструмента (текст, структура, лог)
                 success: bool,  # прошло ли действие успешно
                 ):
        self.action = action
        self.output = output
        self.success = success

@dataclass
class Memory:
    history: List[Observation] = field(default_factory=list)
    scratchpad: Dict = field(default_factory=dict)

    def store(self, observation: Observation):
        self.history.append(observation)
        self.scratchpad.clear()

@dataclass
class Context:
    memory: Memory                   # полная память
    user_goal: Optional[str]                   # что агент должен сделать сейчас
    last_observation: Observation    # последнее наблюдение (может быть None)

    def __init__(self, memory: Memory = None, user_goal: str = None):
        self.memory = memory
        self.user_goal = user_goal
        self.last_observation = None

    def update(self, observation: Observation):
        self.last_observation = observation
        self.memory.store(observation)

    def set_task(self, task):
        self.user_goal = task
        self.memory.scratchpad.clear()

    def set_plan(self, plan):
        self.memory.scratchpad["plan"] = plan

    def get_plan(self):
        if "plan" in self.memory.scratchpad:
            return self.memory.scratchpad["plan"]
        else:
            return None


@dataclass
class Thought:
    def __init__(self, reasoning: str, confidence: float, source: str = "unknown", action_plan: Optional[str] = None):
        self.reasoning = reasoning
        self.confidence = confidence
        self.source = source
        self.timestamp = datetime.now()
        self.priority = 0  # Default priority
        self.action_plan = action_plan  # Concrete steps to take
        self.reasoning_chain = []  # List of reasoning steps leading to this thought


    def add_reasoning(self, reasoning_step: str):
        self.reasoning_chain.append(reasoning_step)

    def adjust_confidence(self, new_confidence: float):
        self.confidence = new_confidence

    def set_priority(self, priority: int):
        self.priority = priority

    def set_action_plan(self, plan: str):
        self.action_plan = plan


# Ядро мышления агента
class ThoughtManager:
    def __init__(self, context: Context, tools: list[Tool] = None):
        self.context = context
        self.client1 = AgentClient("llm1")
        self.client2 = AgentClient("llm2")
        self.embedder = Embedder("embedding_llm1")
        self.tools = tools

    # Основной метод мышления, вызывающий все уровни
    async def think(self, situation: str) -> Thought:
        # 1. Быстро собираем всё, что можем дать LLM бесплатно
        #template_hints = self._collect_active_template_hints(situation)
        rag_context = await self.rag_thinking(situation)
        # recent_errors = self._get_recent_errors()

        # 2. LLM всегда думает — она у тебя локальная и быстрая!
        llm_thought = await self.llm_thinking(
            situation=situation,
            #template_hints=template_hints,  # ← вот они!
            rag_context=rag_context,  # ← и вот это!
            # recent_errors=recent_errors
        )

        # 3. Но после LLM — страховка: если она несёт чушь — шаблон может перебить
        # emergency_override = self._emergency_template_check(llm_thought, situation)
        # if emergency_override:
        #     emergency_override.source = "template_override"
        #     return emergency_override

        return llm_thought

        # ===================================================================
        # 1. Шаблонные рассуждения — синхронные и молниеносные
        # ===================================================================
    def template_thinking(self, situation: str) -> Optional[Thought]:
        for template in [
            self._error_handling_template,
            self._repeated_task_template,
            self._task_decomposition_template,
            self._fast_install_template,  # например, "установи numpy" → сразу pip
            self._completion_check_template,  # "всё готово" → завершаем
        ]:
            thought = template(situation)
            if thought:
                return thought
        return None

        # (тут вставляешь свои 3–5 шаблонов из предыдущих сообщений — они остаются sync)

        # ===================================================================
        # 2. RAG-мышление — асинхронное, из памяти + векторного хранилища
        # ===================================================================
    async def rag_thinking(self, situation: str) -> Optional[Thought]:
        try:
            # 1. Ищем похожие прошлые эпизоды в памяти
            relevant = await self._get_rag_context(situation, k=3)

            if not relevant:
                return None

            # 2. Формируем контекст из найденных наблюдений
            context_lines = []
            for doc in relevant:
                obs = doc.metadata.get("observation")
                if obs:
                    context_lines.append(f"Раньше при похожей задаче сделал: {obs.action.description}")
                    context_lines.append(f"Результат: {'успех' if obs.success else 'провал'}")

            reasoning = "Нашёл похожие успешные кейсы в истории:\n" + "\n".join(context_lines[:8])

            return Thought(
                reasoning=reasoning,
                confidence=0.75 + 0.15 * len(relevant) / 5,
                source="rag_retrieval",
                action_plan="Повторить стратегию из самого похожего кейса"
            )
        except Exception as e:
            print(f"RAG упал: {e}")
            return None

    # ===================================================================
    # 3. LLM-мышление — финальный резерв (самый медленный и дорогой)
    # ===================================================================
    async def llm_thinking(self, situation: str,
                           template_hints: str=None,  # ← вот они!
                           rag_context: str=None,  # ← и вот это!
                           recent_error: str=None
                           ) -> Thought:
        prompt = f"""
Цель пользователя: {self.context.user_goal or "не указана"}

Текущая ситуация:
{situation}

RAG (если есть):
{rag_context}

Последние 5 действий из истории (если есть):
{self._format_recent_history()}

На основе этого — подумай вслух и предложи следующее действия в пункте action_plan соответствующие списку MCP-инструментов который приложен,
Ответ строго в JSON:
{{
    "reasoning": "твои рассуждения на русском",
    "action_plan": [
        {{
            "tool": "write_file",
            "parameters": {{
                "path": "test.py",
                "content": "содержимое файла которое нужно сохранить...",
                "mkdir": true
            }}
        }},
        {{
            "tool": "list_directory",
            "parameters": {{
                "path": "."
            }}
        }}
    ],
    "confidence": 0.XX
}}
"""

        try:
            response = self.client1.request(
                msgs=[{"role": "system", "content": "Ты думаешь быстро и по делу."},
                          {"role": "user", "content": prompt}],
            )

            json_text = response.choices[0].message.content.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]

            data = json.loads(json_text)

            return Thought(
                reasoning=data.get("reasoning", "LLM сгенерировал мысль"),
                confidence=float(data.get("confidence", 0.8)),
                source="llm",
                action_plan=data.get("action_plan")
            )

        except Exception as e:
            print(f"LLM упал: {e}")
            # Абсолютный fallback — хотя бы не падаем
            return Thought(
                reasoning=f"Не смог связаться с LLM. Пробую базовый шаг. Ошибка: {e}",
                confidence=0.4,
                source="llm_fallback",
                action_plan="1. Вывести текущее состояние: ls -la && pwd && git status"
            )

    # Вспомогательная утилита
    def _format_recent_history(self) -> str:
        lines = []
        for obs in self.context.memory.history[-5:]:
            mark = "[Success]" if obs.success else "[Error]"
            lines.append(f"{mark} {obs.action.type}: {obs.action.description or obs.action.command[:80]}")
        return "\n".join(lines) if lines else "История пуста"

    async def _get_rag_context(self, situation, k = 3) -> str:
        chunk_list = self.embedder.find_chunks(situation, k, max_distance = 0.1)
        result = ",".join(x.content for x in chunk_list)
        return result
