import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.artifacts.models import ArtifactMeta, ArtifactRef


class ArtifactStore:
    """Simple filesystem-backed artifact storage."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self) -> str:
        return uuid.uuid4().hex

    def _meta_path(self, artifact_id: str) -> Path:
        return self.base_path / f"{artifact_id}.meta.json"

    def _data_path(self, artifact_id: str, content_type: str) -> Path:
        suffix = "bin" if content_type == "bytes" else "txt"
        return self.base_path / f"{artifact_id}.{suffix}"

    def save_text(self, text: str, summary: Optional[str] = None) -> ArtifactRef:
        artifact_id = self._generate_id()
        data_path = self._data_path(artifact_id, "text")
        data_path.write_text(text, encoding="utf-8")
        meta = ArtifactMeta(
            artifact_id=artifact_id,
            path=data_path,
            content_type="text",
            size=len(text.encode("utf-8")),
            created_at=datetime.utcnow(),
            summary=summary,
        )
        self._save_meta(meta)
        return ArtifactRef(artifact_id)

    def save_bytes(self, blob: bytes, summary: Optional[str] = None) -> ArtifactRef:
        artifact_id = self._generate_id()
        data_path = self._data_path(artifact_id, "bytes")
        data_path.write_bytes(blob)
        meta = ArtifactMeta(
            artifact_id=artifact_id,
            path=data_path,
            content_type="bytes",
            size=len(blob),
            created_at=datetime.utcnow(),
            summary=summary,
        )
        self._save_meta(meta)
        return ArtifactRef(artifact_id)

    def save_json(self, data: Any, summary: Optional[str] = None) -> ArtifactRef:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return self.save_text(text, summary=summary)

    def save(self, data: Any, summary: Optional[str] = None) -> ArtifactRef:
        if isinstance(data, bytes):
            return self.save_bytes(data, summary=summary)
        if isinstance(data, (dict, list)):
            return self.save_json(data, summary=summary)
        return self.save_text(str(data), summary=summary)

    def _save_meta(self, meta: ArtifactMeta) -> None:
        meta_path = self._meta_path(meta.artifact_id)
        meta_path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, ref: ArtifactRef | str) -> tuple[ArtifactMeta, Any]:
        artifact_ref = ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_ref(ref)
        meta_path = self._meta_path(artifact_ref.artifact_id)
        meta = ArtifactMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.content_type == "bytes":
            content: Any = meta.path.read_bytes()
        else:
            content = meta.path.read_text(encoding="utf-8")
        return meta, content

    def read_slice(self, ref: ArtifactRef | str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> tuple[ArtifactMeta, Any]:
        meta, content = self.load(ref)
        if meta.content_type == "bytes":
            # For bytes, slicing means byte ranges
            start = start_line or 0
            end = end_line if end_line is not None else len(content)
            return meta, content[start:end]

        if start_line is None and end_line is None:
            return meta, content

        lines = content.splitlines()
        # Convert to 0-based slice
        start_idx = max((start_line - 1) if start_line else 0, 0)
        end_idx = end_line if end_line is not None else len(lines)
        sliced = "\n".join(lines[start_idx:end_idx])
        return meta, sliced
