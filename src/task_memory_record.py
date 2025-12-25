import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.task_applicability import Applicability
from src.task_evidence import Evidence
from src.task_pitfall_fix import PitfallFix


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    # Identification / retrieval anchors
    title: str = ""
    problem_signature: str = ""  # short searchable signature (errors/keywords)
    applicability: Applicability = field(default_factory=Applicability)
    tags: List[str] = field(default_factory=list)

    # Core content
    task: str = ""
    solution_outline: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    pitfalls_and_fixes: List[PitfallFix] = field(default_factory=list)

    # Artifacts / verification
    artifacts: List[str] = field(default_factory=list)  # files, commands, outputs
    verification: List[str] = field(default_factory=list)  # how we checked it's done
    evidence: List[Evidence] = field(default_factory=list)  # anchors

    # Reuse
    reusable_patterns: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)
    source: str = "agent"  # "agent" | "user" | "system"

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
