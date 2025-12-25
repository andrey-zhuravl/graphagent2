import json
from pathlib import Path

from src.action import Action
from src.llm.agent_client import AgentClient
from src.memory import Context, Thought, StateUpdate
from src.rag.agent_embeding import Embedder
from src.task import CompactGoal, MemoryRecord

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


async def write_file(file_path: str, content: str) -> str:
    """Записывает файл (создает директории при необходимости)"""
    # print(f"Начали записывать файл {file_path}")
    path = Path(file_path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            path.write_text(content, encoding="utf-8")
        # print(f"✓ Файл успешно записан: {path}")
        return f"✓ Файл успешно записан: {path}"
    except Exception as e:
        print(f"❌ Ошибка при записи: {str(e)}")
        return f"❌ Ошибка при записи: {str(e)}"


class LlmThoughtManager:
    def __init__(
        self,
        context: Context,
        client1: AgentClient,
        client2: AgentClient,
        embedder: Embedder,
    ):
        self.context = context
        self.client1 = client1
        self.client2 = client2
        self.embedder = embedder

    async def llm_thinking(
        self,
        situation: str,
        template_hints: str = None,  # ← вот они!
        rag_context: str = None,  # ← и вот это!
        state_update: str = None,
        recent_error: str = None,
        step: int = 0,
    ) -> Thought:
        prompt = _load_prompt("llm_thinking.txt").format(
            user_goal=self.context.user_goal,
            situation=situation,
            state_update=state_update,
            rag_context=rag_context,
        )
        json_text = None
        await write_file(f"D:\\temp\\debug\\prompt_{step}.txt", prompt)
        try:
            response = self.client1.request(
                msgs=[
                    {"role": "system", "content": "Ты думаешь быстро и по делу."},
                    {"role": "user", "content": prompt},
                ],
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
                        start_line=data.get("state_update").get("start_line"),
                        end_line=data.get("state_update").get("end_line"),
                        found_end=data.get("state_update").get("found_end"),
                        notes=data.get("state_update").get("notes"),
                    ),
                )
            except Exception as e:
                print(f"JSON упал: {e} \n {json_text}")
                return Thought(
                    reasoning=f"Ошибка JSON: {e} \n {json_text}",
                    confidence=1.0,
                    source="llm",
                    action_plan=[Action(tool_name="json_error_llm")],
                )

        except Exception as e:
            print(f"LLM упал: {e} \n {json_text}")
            # Абсолютный fallback — хотя бы не падаем
            return Thought(
                reasoning=f"Ошибка: {e}",
                confidence=1.0,
                source="error_llm",
                action_plan=[Action(tool_name="error_llm")],
            )

    async def debug_logging_to_files(self, json_text, prompt, step):
        await write_file(f"D:\\temp\\debug\\prompt_{step}.txt", prompt)
        await write_file(f"D:\\temp\\debug\\llm__{step}.txt", json_text)

    async def llm_pre_thinking(self, tools: str) -> CompactGoal:
        prompt = _load_prompt("llm_pre_thinking.txt").format(
            user_goal=self.context.user_goal,
            tools=tools,
        )
        json_text = None
        try:
            response = self.client1.request(
                msgs=[
                    {"role": "system", "content": "Ты думаешь быстро и по делу."},
                    {"role": "user", "content": prompt},
                ],
            )

            json_text = response.choices[0].message.content.strip()
            await self.debug_logging_to_files(json_text, prompt, 0)
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
                json_text = json_text + "}}"

            return CompactGoal.from_json(json_text)
        except Exception as e:
            print(f"упал: {e} \n {json_text}")
            return CompactGoal.from_json(
                json.loads(
                    f"""
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
                """
                )
            )

    async def llm_post_thinking(
        self, compact_goal: CompactGoal, tools: str, observations: str
    ) -> MemoryRecord:
        prompt = _load_prompt("llm_post_thinking.txt").format(
            user_goal=self.context.user_goal,
            tools=tools,
            observations=observations,
        )
        json_text = None
        try:
            response = self.client1.request(
                msgs=[
                    {"role": "system", "content": "Ты думаешь быстро и по делу."},
                    {"role": "user", "content": prompt},
                ],
            )

            json_text = response.choices[0].message.content.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
            return MemoryRecord.from_json(json_text)
        except Exception as e:
            print(f"упал: {e} \n {json_text}")
            return MemoryRecord.from_json(
                json.loads(
                    f"""
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
"""
                )
            )
