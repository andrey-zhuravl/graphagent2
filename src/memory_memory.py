from dataclasses import dataclass, field
from typing import Dict, List

from src.memory_observation import Observation


@dataclass
class Memory:
    history: List[Observation] = field(default_factory=list)
    scratchpad: Dict = field(default_factory=dict)

    def store(self, observation: Observation):
        self.history.append(observation)
        self.scratchpad.clear()
