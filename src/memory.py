from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

from src.action import Action


@dataclass
class Observation:
    def __init__(self,
                 action: Action = None,  # к какому действию относится
                 output: any = None,  # результат инструмента (текст, структура, лог)
                 success: bool = False,  # прошло ли действие успешно
                 thought: str = None,
                 ):
        self.action = action
        self.output = output
        self.thought = thought
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
    memory: Memory  # полная память
    user_goal: Optional[str]  # что агент должен сделать сейчас
    last_observation: Observation  # последнее наблюдение (может быть None)

    def __init__(self, memory: Memory = None, user_goal: str = None):
        self.memory = memory or Memory()
        self.user_goal = user_goal
        self.last_observation = None

    def update(self, observations: list[Observation]):
        for observation in observations:
            self.update_observation(observation)

    def update_observation(self, observation: Observation):
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

    # Вспомогательная утилита
    def format_recent_history(self, num_obs=10) -> str:
        lines = []
        for obs in self.memory.history[-num_obs:]:
            mark = "[Success]" if obs.success else "[Error]"
            lines.append(f"{mark} {obs.action.tool_name}: {obs.action.params} - результат: {obs.output}")
        return "\n".join(lines) if lines else "История пуста"


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
