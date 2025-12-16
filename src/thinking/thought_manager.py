import asyncio
import json
from typing import Optional

from src.action import Action
from src.llm.agent_client import AgentClient
from src.memory import Context, Thought
from src.rag.agent_embeding import Embedder
from src.thinking.llm_thought_manager import LlmThoughtManager
from src.thinking.rag_thought_manager import RagThoughtManager
from src.thinking.template_thought_manager import TemplateThoughtManager
from src.tool import Tool


# Ядро мышления агента
class ThoughtManager:
    def __init__(self, context: Context = None, tools: list[Tool] = None, embedder: Embedder = None):
        self.context = context
        self.client1 = AgentClient("llm1")
        self.client2 = AgentClient("llm2")
        self.embedder = Embedder("embedding_llm1")
        self.tools = tools
        self.rag_thought_manager = RagThoughtManager(
            context = self.context,
            client1 = self.client1,
            client2 = self.client2,
            embedder = self.embedder
        )
        self.llm_thought_manager = LlmThoughtManager(
            context = self.context,
            client1 = self.client1,
            client2 = self.client2,
            embedder = self.embedder
        )
        self.template_thought_manager = TemplateThoughtManager(
            context = self.context,
            client1 = self.client1,
            client2 = self.client2,
            embedder = self.embedder
        )

    # Основной метод мышления, вызывающий все уровни
    async def think(self,tools: list[Tool], situation: str) -> Thought:
        self.tools = tools

        #template_hints = self.template_thought_manager.template_thinking(situation)

        rag_context = await self.rag_thought_manager.rag_thinking(situation)
        # recent_errors = self._get_recent_errors()

        llm_thought = await self.llm_thought_manager.llm_thinking(
            situation=situation,
            #template_hints=template_hints,  # ← вот они!
            rag_context=rag_context,  # ← и вот это!
            # recent_errors=recent_errors
        )

        return llm_thought
