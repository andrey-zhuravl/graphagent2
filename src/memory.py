import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

from src.artifacts.models import ArtifactRef

from src.action import Action
from src.task import CompactGoal

@dataclass
class StateUpdate:
    def __init__(self, start_line: str = "unknown",
                 found_end: bool = False,
                 end_line: str = "unknown",
                 notes: Optional[list[str]] = None):
        self.start_line = start_line
        self.found_end = found_end
        self.end_line = end_line
        self.notes = notes

    def to_str(self):
        return json.dumps(
            {
            "state_update": {
                    "start_line": self.start_line,
                    "found_end": self.found_end,
                    "end_line": self.end_line,
                    "notes": self.notes,
                }
            }, ensure_ascii=False
        )


@dataclass
class Observation:
    def __init__(self,
                 action: Action = None,  # к какому действию относится
                 output: any = None,  # описание результата
                 output_short: any = None,  # короткое описание результата
                 success: bool = False,  # прошло ли действие успешно
                 step: int = -1,
                 artifacts: Optional[list[ArtifactRef | str]] = None,
                 ):
        self.action = action
        self.output = output
        self.output_short = output_short
        self.success = success
        self.step = step
        self.artifacts = artifacts or []

    def _output_short(self, output_max_len: int = 100) -> Optional[str]:
        tool_name = getattr(self.action, "tool_name", None)

        # Make output a short string (max 100 chars)
        if self.output is None:
            output_short = None
        else:
            s = str(self.output)
            if len(s) > output_max_len:
                output_short = s[:output_max_len]
            else:
                output_short = s

        return output_short

    def to_json_fields(self, output_max_len: int = 100) -> dict:
        tool_name = getattr(self.action, "tool_name", None)
        output_short = self._output_short(output_max_len)

        return {
            "tool_name": tool_name,
            "success": bool(self.success),
            "output_short": output_short,
            "output": self.output,
            "step": self.step,
            "artifacts": [str(a) for a in self.artifacts],
        }

    def to_json(self, output_max_len: int = 100) -> str:
        return json.dumps(self.to_json_fields(output_max_len), ensure_ascii=False)



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
            remaining = min(num_obs, history_len)  - i
            if remaining <= num_full_output_obs or history_len <= num_full_output_obs:
                output = obs.output
            else:
                output = obs.output_short
            mark = "[Success]" if obs.success else "[Error]"
            refs = f" refs={','.join([str(a) for a in obs.artifacts])}" if obs.artifacts else ""
            if obs.action.tool_name == "write_file":
                params = f"{obs.action.params}"[:30]
            else:
                params = obs.action.params
            lines.append(
                f"{mark} {obs.action.tool_name}: {params} - результат: {output}{refs}"
            )
        return "\n".join(lines) if lines else "История пуста"

@dataclass
class Thought:
    def __init__(self, reasoning: str,
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
        self.state_update: StateUpdate = state_update  # List of reasoning steps leading to this thought

    def add_reasoning(self, reasoning_step: str):
        self.reasoning_chain.append(reasoning_step)

    def adjust_confidence(self, new_confidence: float):
        self.confidence = new_confidence

    def set_priority(self, priority: int):
        self.priority = priority

    def set_action_plan(self, plan: str):
        self.action_plan = plan
