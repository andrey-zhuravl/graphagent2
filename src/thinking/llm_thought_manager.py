import json

from src.action import Action
from src.llm.agent_client import AgentClient
from src.memory import Context, Thought
from src.rag.agent_embeding import Embedder
from src.task import CompactGoal, MemoryRecord


class LlmThoughtManager:
    def __init__(self, context: Context,
                 client1: AgentClient,
                 client2: AgentClient,
                 embedder: Embedder):
        self.context = context
        self.client1 = client1
        self.client2 = client2
        self.embedder = embedder

    async def llm_thinking(self, situation: str,
                           template_hints: str = None,  # ← вот они!
                           rag_context: str = None,  # ← и вот это!
                           recent_error: str = None
                           ) -> Thought:
        prompt = f"""
Наша Цель пользователя: {self.context.user_goal or "не указана"}
IMPORTANT:
The parameter for write_file - "path" is FORBIDDEN.
Use ONLY "file_path".
Текущая ситуация:
{situation}

RAG (если есть):
{rag_context}

Ответь ТОЛЬКО JSON.
Правила:
1. Используй think_along ТОЛЬКО ОДИН РАЗ подряд, если нужно сформулировать новую идею. Если используешь think_along - других действий на этом шаге добавлять нельзя.
2. После любого think_along (или если идея уже готова) — ОБЯЗАТЕЛЬНО примени её через инструменты файловой системы: create_file, edit_file, write_file и т.д.
3. НЕ ПОВТОРЯЙ уже существующие идеи — сначала проверь через read_file или list_directory.
4. Если видишь, что в истории уже было 2–3 think_along подряд — СРАЗУ переходи к действию с файлами.
5. Когда достигнуты критерии указанные в цели — используй submit_task.
Ответ строго в JSON, Начни с '{' и закончи '}':
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
        }}
    ],
    "confidence": 0.XX
}}
"""
        json_text = None
        try:
            response = self.client1.request(
                msgs=[{"role": "system", "content": "Ты думаешь быстро и по делу."},
                      {"role": "user", "content": prompt}],
            )

            json_text = response.choices[0].message.content.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
            try:
                data = json.loads(json_text)

                return Thought(
                    reasoning=data.get("reasoning", "LLM сгенерировал мысль"),
                    confidence=float(data.get("confidence", 0.8)),
                    source="llm",
                    action_plan=data.get("action_plan")
                )
            except Exception as e:
                print(f"JSON упал: {e} \n {json_text}")
                return Thought(
                    reasoning=f"Ошибка JSON: {e} \n {json_text}",
                    confidence=1.0,
                    source="llm",
                    action_plan=[
                        Action(tool_name="json_error_llm")
                    ]
                )

        except Exception as e:
            print(f"LLM упал: {e} \n {json_text}")
            # Абсолютный fallback — хотя бы не падаем
            return Thought(
                reasoning=f"Ошибка: {e}",
                confidence=1.0,
                source="error_llm",
                action_plan=[
                    Action(tool_name="error_llm")
                ]
            )

    async def llm_pre_thinking(self, tools: str) -> CompactGoal:
        prompt = f"""
Ты — модуль "Task Spec". Твоя задача: превратить пользовательскую задачу и текущую ситуацию в СТРОГУЮ спецификацию для ReAct-агента.

Вход:
- Задача пользователя: {self.context.user_goal}
- названия MCP-tools: {tools}

Выход: ТОЛЬКО один JSON-объект. Никакого текста вне JSON.

Требования к JSON:
1) Поле compact_goal — объект со структурой, как в схеме ниже. Пиши кратко, но конкретно.
2) tool_name_list - список только необходимых и достаточных для полного решения всей задачи MCP-tools.
3) success_criteria и subgoals[].done_when должны быть проверяемыми (по наблюдаемым фактам).
4) rag_queries: 0–3 штуки, только если они реально могут изменить план/решение. Каждый запрос должен закрывать конкретный unknown.
5) Не выдумывай фактов. Если данных нет — положи это в unknowns.

СХЕМА (заполни все поля, если нечего — ставь пустой массив/пустую строку):
  "compact_goal": {{
    "title": "",
    "objective": "",
    "deliverables": [],
    "tool_name_list": [],
    "context": {{
      "domain": "",
      "entities": [],
      "inputs": [],
      "constraints": [],
      "non_goals": []
    }},
    "success_criteria": [],
    "subgoals": [
      {{
        "id": "S1",
        "description": "",
        "done_when": "",
        "evidence_needed": [],
        "risks": []
      }}
    ],
    "unknowns": [],
    "assumptions": [],
  "rag_queries": [],
  "confidence": 0.0
}}
"""
        json_text = None
        try:
            response = self.client1.request(
                msgs=[{"role": "system", "content": "Ты думаешь быстро и по делу."},
                      {"role": "user", "content": prompt}],
            )

            json_text = response.choices[0].message.content.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
                data = json.loads(json_text)

            return CompactGoal.from_json(json_text)
        except Exception as e:
            print(f"упал: {e} \n {json_text}")
            return CompactGoal.from_json(json.loads(f"""
  "compact_goal": {{
    "title": "",
    "objective": "",
    "deliverables": [],
    "tool_name_list": [],
    "context": {{
      "domain": "",
      "entities": [],
      "inputs": [],
      "constraints": [],
      "non_goals": []
    }},
    "success_criteria": [],
    "subgoals": [
      {{
        "id": "S1",
        "description": {self.context.user_goal},
        "done_when": "",
        "evidence_needed": [],
        "risks": []
      }}
    ],
    "unknowns": [],
    "assumptions": [],
      "rag_queries": [],
  "confidence": 0.0
  }}
                """))

    async def llm_post_thinking(self, compact_goal: CompactGoal, tools: str, observations: str) -> MemoryRecord:
            prompt = f"""
    Ты — модуль "Task Spec". Твоя задача: превратить контекст выполненной задачи в RAG-запись для ReAct-агента.

    Вход:
    - Задача пользователя: {self.context.user_goal}
    - названия использованных MCP-tools: {tools}
    - наблюдения в процессе задачи: {observations}

    Выход: ТОЛЬКО один JSON-объект. Никакого текста вне JSON.

    Требования к JSON:
    1) Поле memory_record — объект со структурой, как в схеме ниже. Пиши кратко, но конкретно.
    2) tool_name_list - список MCP-tools использованных для решения всей задачи.
    3) Не выдумывай фактов.

    СХЕМА (заполни все поля, если нечего — ставь пустой массив/пустую строку):
    {{
  "memory_record": {{
    "title": "кратко что было сделано",
    "task": "какая была задача",
    "solution_outline": ["шаги решения"],
    "tool_name_list": ["MCP-tools-name"],
    "key_decisions": ["почему выбрали так"],
    "pitfalls_and_fixes": [{{"symptom":"", "cause":"", "fix":""}}],
    "artifacts": ["файлы/коммиты/команды/ссылки"],
    "verification": ["как проверили что готово"],
    "reusable_patterns": ["что можно переиспользовать в будущем"],
    "tags": ["kafka", "spring", "react-agent", "..."]
  }}
    }}
    """
            json_text = None
            try:
                response = self.client1.request(
                    msgs=[{"role": "system", "content": "Ты думаешь быстро и по делу."},
                          {"role": "user", "content": prompt}],
                )

                json_text = response.choices[0].message.content.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:-3]
                    data = json.loads(json_text)

                    return MemoryRecord.from_json(json_text)
            except Exception as e:
                print(f"упал: {e} \n {json_text}")
                return MemoryRecord.from_json(json.loads(f"""
                {{
  "memory_record": {{
    "title": "кратко что было сделано",
    "task": {self.context.user_goal},
    "solution_outline": ["шаги решения"],
    "tool_name_list": [{tools}],
    "key_decisions": [],
    "pitfalls_and_fixes": [],
    "artifacts": [],
    "verification": [],
    "reusable_patterns": [],
    "tags": []
  }}
    }}
"""))
