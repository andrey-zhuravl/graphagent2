import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class StateUpdate:
    def __init__(
        self,
        start_line: str = "unknown",
        found_end: bool = False,
        end_line: str = "unknown",
        notes: Optional[list[str]] = None,
    ):
        self.start_line = start_line
        self.found_end = found_end
        self.end_line = end_line
        self.notes = notes

    def to_str(self):
        return json.dumps(
            {
                "state_update": {
                    "start_line": self.start_line,
                    "found_end": self.found_end,
                    "end_line": self.end_line,
                    "notes": self.notes,
                }
            },
            ensure_ascii=False,
        )
