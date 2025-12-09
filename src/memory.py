from dataclasses import dataclass, field
from typing import Optional

from src.action import Action

@dataclass
class Observation:
    def __init__(self,
    action: Action,        # к какому действию относится
    output: any,           # результат инструмента (текст, структура, лог)
    success: bool,         # прошло ли действие успешно
    ):
        self.action = action
        self.output = output
        self.success = success

@dataclass
class Memory:
    history: list[Observation]    # вся последовательность наблюдений
    scratchpad: dict = field(default_factory=dict)          # временные данные между шагами

    def store(self, observation: Observation):
        self.history.append(observation)
        #self.scratchpad.clear()

@dataclass
class Context:
    memory: Memory                   # полная память
    user_goal: Optional[str]                   # что агент должен сделать сейчас
    last_observation: Observation    # последнее наблюдение (может быть None)

    def __init__(self, memory: Memory, user_goal: str):
        self.memory = memory
        self.user_goal = user_goal
        self.last_observation = None

    def update(self, observation: Observation):
        self.last_observation = observation
        self.memory.store(observation)

    def set_task(self, task):
        self.user_goal = task
        #self.memory.scratchpad.clear()

@dataclass
class Thought:
    def __init__(self, text, final_answer=None):
        self.text = text
        self.final_answer = final_answer