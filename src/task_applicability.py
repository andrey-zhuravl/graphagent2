from dataclasses import dataclass, field
from typing import List


@dataclass
class Applicability:
    # Filters for retrieval (so memory doesn't pollute other tasks)
    scope: str = "global"  # "global" or "project:<repo_id>"
    domain: str = ""  # e.g. "java_review", "python_etl", "agent_ops"
    language: str = "any"  # "java" | "python" | "any"
    stack: List[str] = field(default_factory=list)  # e.g. ["spring", "kafka", "pytest"]
    conditions: List[str] = field(default_factory=list)  # free-form "when applicable"
