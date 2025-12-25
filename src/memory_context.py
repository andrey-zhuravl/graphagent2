from dataclasses import dataclass
from typing import Optional

from src.memory_memory import Memory
from src.memory_observation import Observation
from src.memory_state_update import StateUpdate
from src.task import CompactGoal


@dataclass
class Context:
    memory: Memory  # полная память
    user_goal: Optional[str]  # что агент должен сделать сейчас
    compact_goal: Optional[CompactGoal]
    rag: Optional[str]
    last_observation: Observation  # последнее наблюдение (может быть None)
    state_update: StateUpdate

    def __init__(self, memory: Memory = None, user_goal: str = None):
        self.memory = memory or Memory()
        self.user_goal = user_goal
        self.last_observation = None
        self.state_update: StateUpdate = StateUpdate()

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
    def format_recent_history(self, num_obs=5, num_full_output_obs=2) -> str:
        lines = []
        history_len = len(self.memory.history)
        for i, obs in enumerate(self.memory.history[-num_obs:]):
            # Сколько событий осталось показать после текущего (включая текущее)
            remaining = min(num_obs, history_len) - i
            if remaining <= num_full_output_obs or history_len <= num_full_output_obs:
                output = obs.output
            else:
                output = obs.output_short
            mark = "[Success]" if obs.success else "[Error]"
            refs = (
                f" refs={','.join([str(a) for a in obs.artifacts])}"
                if obs.artifacts
                else ""
            )
            if obs.action.tool_name == "write_file":
                params = f"{obs.action.params}"[:30]
            else:
                params = obs.action.params
            lines.append(
                f"{mark} {obs.action.tool_name}: {params} - результат: {output}{refs}"
            )
        return "\n".join(lines) if lines else "История пуста"
