import asyncio
import json
from typing import Optional

from src.action import Action
from src.llm.agent_client import AgentClient
from src.memory import Context, Thought
from src.rag.agent_embeding import Embedder
from src.tool import Tool

class RagThoughtManager:
    def __init__(self, context: Context,
                 client1: AgentClient,
                 client2: AgentClient,
                 embedder: Embedder):
        self.context = context
        self.client1 = client1
        self.client2 = client2
        self.embedder = embedder

    async def rag_thinking(self, situation: str) -> Optional[str]:
        # 1. По коду
        # code_chunks = await asyncio.to_thread(self.embedder.find_chunks, situation, top_k=3)

        # 2. По памяти агента — это важнее!
        memory_chunks = await asyncio.to_thread(
            self.embedder.find_memory_chunks,
            situation,
            top_k=5,
            max_distance=0.12)

        if not memory_chunks:
            return None

        lines = ["Похожие прошлые опыты агента:"]
        for mem in memory_chunks[:4]:
            status = "Успех" if mem.success else "Провал"
            lines.append(f"{status}: {mem.action_description} → {mem.result_summary}")
            if mem.action_plan:
                lines.append(f"  План был: {mem.action_plan[:150]}")

        reasoning = "\n".join(lines)
        return reasoning

    async def save_to_rag(self, thought):
        if self.embedder:
            last_obs = self.context.last_observation
            if last_obs and last_obs.action.tool_name not in ["think_along", "empty_action"]:
                # Краткая ситуация — можно взять short_text из build_situation, если перепишешь на возврат кортежа
                situation_short = f"Цель: {self.context.user_goal} | Последнее: {last_obs.action.tool_name}"

                action_desc = f"Вызвал {last_obs.action.tool_name} с {str(last_obs.action.params)[:150]}"

                result_summary = (
                    f"Успех: {str(last_obs.output)[:200]}" if last_obs.success
                    else f"Ошибка: {str(last_obs.output)[:200]}"
                )

                # Если у тебя есть доступ к thought.reasoning и thought.action_plan
                reasoning = thought.reasoning if 'thought' in locals() else None
                action_plan = thought.action_plan if 'thought' in locals() and hasattr(thought, 'action_plan') else None

                asyncio.create_task(
                    asyncio.to_thread(
                        self.embedder.save_memory_chunk,
                        situation=situation_short,
                        action_description=action_desc,
                        result_summary=result_summary,
                        reasoning=reasoning,
                        action_plan=str(action_plan) if action_plan else None,
                        success=last_obs.success
                    )
                )

    async def _get_rag_context(self, situation) -> str:
        chunk_list = self.embedder.find_chunks(situation, top_k=3, max_distance = 0.1)
        result = ",".join(x.content for x in chunk_list)
        return result