from typing import Any, Dict, Optional

from src.artifacts.store import ArtifactStore


def get_retrieve_artifact_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "retrieve_artifact",
            "description": "Возвращает содержимое сохранённого артефакта по ref (artifact:<id>). Можно запросить диапазон строк.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Ссылка вида artifact:<id>",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Начальная строка (1-индексация) для извлечения части текста.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Конечная строка (включительно) для извлечения части текста.",
                    },
                },
                "required": ["ref"],
            },
        },
    }


def retrieve_artifact(store: ArtifactStore, ref: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Dict[str, Any]:
    meta, content = store.read_slice(ref, start_line=start_line, end_line=end_line)
    return {
        "ref": str(ref),
        "content_type": meta.content_type,
        "size": meta.size,
        "summary": meta.summary,
        "content": content,
        "slice": {
            "start_line": start_line,
            "end_line": end_line,
        },
    }
