from src.action import Action


class Observation:
    action: Action        # к какому действию относится
    output: any           # результат инструмента (текст, структура, лог)
    success: bool         # прошло ли действие успешно

class Memory:
    history: list[Observation]    # вся последовательность наблюдений
    scratchpad: dict              # временные данные между шагами

class Context:
    memory: Memory                   # полная память
    target_class: any                # описание класса, который тестируем
    user_goal: str                   # что агент должен сделать сейчас
    last_observation: Observation    # последнее наблюдение (может быть None)