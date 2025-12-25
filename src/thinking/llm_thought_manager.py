import json
from pathlib import Path

from src.action import Action
from src.llm.agent_client import AgentClient
from src.memory import Context, Thought, StateUpdate
from src.rag.agent_embeding import Embedder
from src.task import CompactGoal, MemoryRecord


async def write_file(file_path: str, content: str) -> str:
    """Записывает файл (создает директории при необходимости)"""
    # print(f"Начали записывать файл {file_path}")
    path = Path(file_path)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            path.write_text(content, encoding="utf-8")
        # print(f"✓ Файл успешно записан: {path}")
        return f"✓ Файл успешно записан: {path}"
    except Exception as e:
        print(f"❌ Ошибка при записи: {str(e)}")
        return f"❌ Ошибка при записи: {str(e)}"

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
                           state_update: str = None,
                           recent_error: str = None,
                           step: int = 0,
                           ) -> Thought:
        prompt = f"""
Задача пользователя: {self.context.user_goal}
Текущая ситуация:
{situation}

Текущий статус:
{state_update}

RAG (если есть):
{rag_context}

Когда достигнуты критерии указанные в цели — используй submit_task.
Обнови state_update в ответе учитывая, то что получила в этом запросе 
в пункте Текущий статус - state_update.
указывай start_line - текущую строку из которой читаем текст
в state_update.notes записывай свои замечания.
избегай повторяющихся действий.
Ответь ТОЛЬКО JSON.
Ответ строго в JSON, Начни с '{' и закончи '}':
{{
    "reasoning": "твои рассуждения на русском кратко и по сути.",
    "action_plan": [
        {{
            "tool": "name_tool",
            "parameters": {{
                "param1": "some value",
            }}
        }}
    ],
    "state_update": {{
        "start_line": null,
        "found_end": false,
        "end_line": null,
        "notes": ["risoning..."]
    }}
    "confidence": 0.XX
}}
"""
        json_text = None
        await write_file(f"D:\\temp\\debug\\prompt_{step}.txt", prompt)
        try:
            response = self.client1.request(
                msgs=[{"role": "system", "content": "Ты думаешь быстро и по делу."},
                      {"role": "user", "content": prompt}],
            )

            json_text = response.choices[0].message.content.strip()
            await write_file(f"D:\\temp\\debug\\llm__{step}.txt", json_text)
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
            try:
                data = json.loads(json_text)

                return Thought(
                    reasoning=data.get("reasoning", "LLM сгенерировал мысль"),
                    confidence=float(data.get("confidence", 0.8)),
                    source="llm",
                    action_plan=data.get("action_plan"),
                    state_update=StateUpdate(
                        start_line = data.get("state_update").get("start_line"),
                        end_line = data.get("state_update").get("end_line"),
                        found_end = data.get("state_update").get("found_end"),
                        notes = data.get("state_update").get("notes"),
                    )
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

    async def debug_logging_to_files(self, json_text, prompt, step):
        await write_file(f"D:\\temp\\debug\\prompt_{step}.txt", prompt)
        await write_file(f"D:\\temp\\debug\\llm__{step}.txt", json_text)

    async def llm_pre_thinking(self, tools: str) -> CompactGoal:
        prompt = f"""
Ты — модуль "Task Spec". Твоя задача: превратить пользовательскую задачу 
в СТРОГУЮ спецификацию для ReAct-агента. Продумай как в целом нужно решать такую задачу, 
какие этапы могут быть, какие этапы должны быть, после того как поймешь общий путь решения, 
проверь себя - правильно ли ты заполняешь  СХЕМА ответа.

Вход:
- Задача пользователя: {self.context.user_goal}
- названия MCP-tools: {tools}

Выход: ТОЛЬКО один JSON-объект. Никакого текста вне JSON.

Требования к JSON:
1) Поле compact_goal — объект со структурой, как в схеме ниже. Пиши кратко, но конкретно.
2) tool_name_list - список только необходимых и достаточных для полного решения всей задачи MCP-tools.
3) success_criteria должны быть проверяемыми (по наблюдаемым фактам).
4) Не выдумывай фактов.

СХЕМА ОТВЕТА (заполни все поля, если нечего — ставь пустой массив/пустую строку):
  "compact_goal": {{
    "title": "",
    "tool_name_list":[],
    "objective": "",
    "success_criteria": [],
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
            await self.debug_logging_to_files(json_text, prompt, 0)
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
                json_text = json_text + "}}"

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
