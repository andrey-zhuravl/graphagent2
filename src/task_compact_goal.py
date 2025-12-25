import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from src.task_goal_context import GoalContext
from src.task_subgoal import Subgoal


@dataclass
class CompactGoal:
    title: str = ""
    objective: str = ""
    deliverables: List[str] = field(default_factory=list)
    tool_name_list: List[str] = field(default_factory=list)
    context: GoalContext = field(default_factory=GoalContext)
    success_criteria: List[str] = field(default_factory=list)
    subgoals: List[Subgoal] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    rag_queries: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(json_obj: Dict[str, Any]) -> "CompactGoal":
        data = json_obj.get("compact_goal")
        ctx = GoalContext(**(data.get("context") or {}))
        subgoals = [Subgoal(**sg) for sg in (data.get("subgoals") or [])]
        return CompactGoal(
            title=data.get("title", ""),
            objective=data.get("objective", ""),
            deliverables=list(data.get("deliverables") or []),
            tool_name_list=list(data.get("tool_name_list") or []),
            context=ctx,
            success_criteria=list(data.get("success_criteria") or []),
            subgoals=subgoals,
            unknowns=list(data.get("unknowns") or []),
            assumptions=list(data.get("assumptions") or []),
            rag_queries=list(data.get("rag_queries") or []),
            confidence=data.get("confidence", 0.0),
        )

    def to_json(self, *, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def from_json(text: str) -> "CompactGoal":
        return CompactGoal.from_dict(json.loads(text))
