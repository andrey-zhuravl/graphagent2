import asyncio
import json
from typing import Optional

from src.action import Action
from src.llm.agent_client import AgentClient
from src.memory import Context, Thought
from src.rag.agent_embeding import Embedder
from src.tool import Tool

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