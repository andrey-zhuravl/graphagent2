from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ArtifactRef:
    """Lightweight reference to an artifact stored on disk."""

    artifact_id: str

    def __str__(self) -> str:
        return f"artifact:{self.artifact_id}"

    @classmethod
    def from_ref(cls, ref: str) -> "ArtifactRef":
        if ref.startswith("artifact:"):
            ref = ref.split(":", 1)[1]
        return cls(artifact_id=ref)


@dataclass
class ArtifactMeta:
    """Metadata describing a stored artifact."""

    artifact_id: str
    path: Path
    content_type: str
    size: int
    created_at: datetime
    summary: Optional[str] = None
    content: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "content_type": self.content_type,
            "size": self.size,
            "created_at": self.created_at.isoformat(),
            "summary": self.summary,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArtifactMeta":
        return cls(
            artifact_id=data["artifact_id"],
            path=Path(data["path"]),
            content_type=data["content_type"],
            size=int(data["size"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            summary=data.get("summary"),
            content=data.get("content"),
        )
