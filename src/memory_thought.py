from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.memory_state_update import StateUpdate


@dataclass
class Thought:
    def __init__(
        self,
        reasoning: str,
        confidence: float,
        source: str = "unknown",
        action_plan: Optional[str] = None,
        state_update: StateUpdate = None,
    ):
        self.reasoning = reasoning
        self.confidence = confidence
        self.source = source
        self.timestamp = datetime.now()
        self.priority = 0  # Default priority
        self.action_plan = action_plan  # Concrete steps to take
        self.reasoning_chain = []  # List of reasoning steps leading to this thought
        self.state_update: StateUpdate = (
            state_update  # List of reasoning steps leading to this thought
        )

    def add_reasoning(self, reasoning_step: str):
        self.reasoning_chain.append(reasoning_step)

    def adjust_confidence(self, new_confidence: float):
        self.confidence = new_confidence

    def set_priority(self, priority: int):
        self.priority = priority

    def set_action_plan(self, plan: str):
        self.action_plan = plan
