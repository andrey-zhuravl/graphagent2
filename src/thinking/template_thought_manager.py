import asyncio
import json
from typing import Optional

from src.action import Action
from src.llm.agent_client import AgentClient
from src.memory import Context, Thought
from src.rag.agent_embeding import Embedder
from src.tool import Tool

class TemplateThoughtManager:
    def __init__(self, context: Context,
                 client1: AgentClient,
                 client2: AgentClient,
                 embedder: Embedder):
        self.context = context
        self.client1 = client1
        self.client2 = client2
        self.embedder = embedder

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