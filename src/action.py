from dataclasses import dataclass, field


@dataclass
class Action:
    tool_name: str        # имя инструмента, который нужно вызвать
    params: dict