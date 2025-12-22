from typing import Any, List, Tuple

from src.artifacts.models import ArtifactRef
from src.artifacts.store import ArtifactStore


class ArtifactPolicy:
    """Defines when and how to persist tool results as artifacts."""

    def __init__(self, store: ArtifactStore, max_inline_length: int = 2000, summary_length: int = 200):
        self.store = store
        self.max_inline_length = max_inline_length
        self.summary_length = summary_length

    def _summarize(self, value: Any) -> str:
        text = str(value)
        if len(text) > self.summary_length:
            return text[: self.summary_length]
        return text

    def _should_persist(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (dict, list, bytes)):
            return True
        text = str(value)
        return len(text) > self.max_inline_length

    def maybe_persist(self, value: Any) -> Tuple[str | None, List[str]]:
        if value is None:
            return None, []

        if self._should_persist(value):
            summary = self._summarize(value)
            ref: ArtifactRef = self.store.save(value, summary=summary)
            return summary, [str(ref)]

        text_value = str(value)
        return self._summarize(text_value), []
