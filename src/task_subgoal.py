from dataclasses import dataclass, field
from typing import List


@dataclass
class Subgoal:
    id: str  # e.g. "S1"
    description: str = ""
    done_when: str = ""  # MUST be evidence-based / checkable
    evidence_needed: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
