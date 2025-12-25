import json
from dataclasses import dataclass
from typing import Optional

from src.action import Action
from src.artifacts.models import ArtifactRef


@dataclass
class Observation:
    def __init__(
        self,
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
