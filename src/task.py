from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
from datetime import datetime, timezone


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# compact_goal (for ReAct loop)
# -----------------------------

@dataclass
class GoalContext:
    domain: str = ""                       # e.g. "code_review", "etl_parsing", "docs"
    entities: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    non_goals: List[str] = field(default_factory=list)


@dataclass
class Subgoal:
    id: str                                # e.g. "S1"
    description: str = ""
    done_when: str = ""                    # MUST be evidence-based / checkable
    evidence_needed: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class CompactGoal:
    title: str = ""
    objective: str = ""
    deliverables: List[str] = field(default_factory=list)
    tool_name_list : List[str] = field(default_factory=list)
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
            tool_name_list = list(data.get("tool_name_list") or []),
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


# --------------------------------
# memory_record (for long-term RAG)
# --------------------------------

@dataclass
class Applicability:
    # Filters for retrieval (so memory doesn't pollute other tasks)
    scope: str = "global"                  # "global" or "project:<repo_id>"
    domain: str = ""                       # e.g. "java_review", "python_etl", "agent_ops"
    language: str = "any"                  # "java" | "python" | "any"
    stack: List[str] = field(default_factory=list)     # e.g. ["spring", "kafka", "pytest"]
    conditions: List[str] = field(default_factory=list)  # free-form "when applicable"


@dataclass
class PitfallFix:
    symptom: str = ""                      # exact error / log pattern if possible
    cause: str = ""
    fix: str = ""
    verify: str = ""


@dataclass
class Evidence:
    # Optional anchors: paths, commits, urls, ids — anything to re-check later
    kind: str = ""                         # "file" | "commit" | "url" | "ticket" | "log" | ...
    ref: str = ""                          # path/hash/url/identifier
    note: str = ""


@dataclass
class MemoryRecord:
    # Identification / retrieval anchors
    title: str = ""
    problem_signature: str = ""            # short searchable signature (errors/keywords)
    applicability: Applicability = field(default_factory=Applicability)
    tags: List[str] = field(default_factory=list)

    # Core content
    task: str = ""
    solution_outline: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    pitfalls_and_fixes: List[PitfallFix] = field(default_factory=list)

    # Artifacts / verification
    artifacts: List[str] = field(default_factory=list)         # files, commands, outputs
    verification: List[str] = field(default_factory=list)      # how we checked it's done
    evidence: List[Evidence] = field(default_factory=list)     # anchors

    # Reuse
    reusable_patterns: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)
    source: str = "agent"                     # "agent" | "user" | "system"

    def touch(self) -> None:
        self.updated_at = _iso_now()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemoryRecord":
        appl = Applicability(**(data.get("applicability") or {}))
        pfs = [PitfallFix(**x) for x in (data.get("pitfalls_and_fixes") or [])]
        ev = [Evidence(**x) for x in (data.get("evidence") or [])]
        return MemoryRecord(
            title=data.get("title", ""),
            problem_signature=data.get("problem_signature", ""),
            applicability=appl,
            tags=list(data.get("tags") or []),
            task=data.get("task", ""),
            solution_outline=list(data.get("solution_outline") or []),
            key_decisions=list(data.get("key_decisions") or []),
            pitfalls_and_fixes=pfs,
            artifacts=list(data.get("artifacts") or []),
            verification=list(data.get("verification") or []),
            evidence=ev,
            reusable_patterns=list(data.get("reusable_patterns") or []),
            created_at=data.get("created_at", _iso_now()),
            updated_at=data.get("updated_at", data.get("created_at", _iso_now())),
            source=data.get("source", "agent"),
        )

    def to_json(self, *, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def from_json(text: str) -> "MemoryRecord":
        return MemoryRecord.from_dict(json.loads(text))