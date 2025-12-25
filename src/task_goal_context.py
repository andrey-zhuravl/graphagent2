from dataclasses import dataclass, field
from typing import List


@dataclass
class GoalContext:
    domain: str = ""  # e.g. "code_review", "etl_parsing", "docs"
    entities: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    non_goals: List[str] = field(default_factory=list)
