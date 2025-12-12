import re
from typing import Tuple

MAX_EMBED_CHARS = 2000  # or tune for your embedder / DB
MAX_LONG_CHARS = 16000

def _normalize_for_embedding(s: str) -> str:
    # 1) убрать лишние пробелы / новые строки
    s = re.sub(r'\s+', ' ', s).strip()
    # 2) заменить абсолютные пути на <PATH>
    s = re.sub(r'(/[A-Za-z0-9_\-./]+)+', '<PATH>', s)
    # 3) заменить UUIDs / long hex on <ID>
    s = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<ID>', s, flags=re.I)
    s = re.sub(r'\b0x[0-9a-fA-F]+\b', '<HEX>', s)
    # 4) truncate
    if len(s) > MAX_EMBED_CHARS:
        s = s[:MAX_EMBED_CHARS].rsplit(' ', 1)[0]
    return s

def _shorten_long(s: str) -> str:
    if len(s) <= MAX_LONG_CHARS:
        return s
    return s[:MAX_LONG_CHARS].rsplit('\n', 1)[0] + "\n...[truncated]"

# Внутри класса Agent — переписанный build_situation:
def build_situation(self) -> Tuple[str, str]:
    parts = []

    # 1. Главная цель — всегда наверху
    if self.context.user_goal:
        parts.append(f"ЦЕЛЬ: {self.context.user_goal}")

    # 2. Последнее действие и его результат
    if self.context.last_observation:
        obs = self.context.last_observation
        status = "УСПЕХ" if obs.success else "ОШИБКА"
        parts.append(f"ПОСЛЕДНЕЕ ДЕЙСТВИЕ ({status}): {obs.action.action_type}")
        # подробный текст для long
        if not obs.success:
            # оставить только последнюю строку ошибки для краткой версии,
            # а full output — в длинной
            full_output = str(obs.output or "")
            last_line = full_output.strip().split('\n')[-1]
            parts.append(f"ОШИБКА: {last_line}")
        else:
            parts.append(f"РЕЗУЛЬТАТ: {str(obs.output)[:240]}")

    # 3. Краткая история (последние 3–5 шагов)
    recent = self.context.memory.history[-6:]
    if len(recent) > 1:
        parts.append("ИСТОРИЯ:")
        for obs in recent[-5:]:
            mark = "Успех" if obs.success else "Провал"
            cmd = getattr(obs.action, "command", None) or getattr(obs.action, "description", None) or str(obs.action)
            short = cmd.split('\n')[0][:120]
            parts.append(f"{mark} {obs.action.action_type}: {short}")

    # 4. Текущая среда — если есть (cwd, branch, python)
    env = self.context.scratchpad.get("env")  # предполагается, что ты кладёшь сюда snapshot среды
    if env:
        # env должен быть коротким словарём: {"cwd": "...", "git_branch": "...", "python": "..."}
        env_parts = []
        for k in ("cwd", "git_branch", "python"):
            if env.get(k):
                env_parts.append(f"{k}={env[k]}")
        if env_parts:
            parts.append("ENV: " + "; ".join(env_parts))

    # 5. План (если есть) — очень важно включить в short/embedding, если он уже есть
    plan = self.context.scratchpad.get("plan")
    if plan:
        parts.append(f"ПЛАН: {plan}")

    long_text = "\n".join(parts)

    # --- build short (target for embedding) ---
    # Сжимаем: берем цель, последнее действие+ошибку (одна строка), краткую историю (1-3 пункта) и план
    short_parts = []
    if self.context.user_goal:
        short_parts.append(f"{self.context.user_goal}")
    if self.context.last_observation:
        obs = self.context.last_observation
        status = "OK" if obs.success else "ERR"
        last_line = ""
        if not obs.success:
            out = str(obs.output or "")
            last_line = out.strip().split('\n')[-1]
        else:
            last_line = (getattr(obs.action, "command", None) or getattr(obs.action, "description", "") )[:120]
        short_parts.append(f"{status}: {obs.action.action_type} -> {last_line}")
    # последние 2 шага истории (коротко)
    for obs in recent[-2:]:
        cmd = getattr(obs.action, "command", None) or getattr(obs.action, "description", None) or str(obs.action)
        short_parts.append(f"{'+' if obs.success else '-'} {obs.action.action_type}: {cmd.splitlines()[0][:100]}")
    if plan:
        short_parts.append(f"PLAN: {plan}")

    short_text_raw = " | ".join(short_parts)
    short_text = _normalize_for_embedding(short_text_raw)

    long_text = _shorten_long(long_text)
    return short_text, long_text
