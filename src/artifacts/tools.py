from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import re
from src.artifacts.store import ArtifactStore

@dataclass
class ToolResult:
    ok: bool
    result_ref: Optional[str] = None
    matches: Optional[List[dict]] = None
    error: Optional[str] = None

class IncompleteArtifactRange(Exception):
    pass

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

def get_retrieve_artifact_mini_tool_schema() -> Dict[str, Any]:
    return {
        "name": "retrieve_artifact",
        "description": "Возвращает содержимое сохранённого артефакта по ref (artifact:<id>). Можно запросить диапазон строк.",
        "parameters": ["ref", "start_line", "end_line"]
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

def tool_slice_artifact_lines(
    store: ArtifactStore,
    source_ref: str,
    start_line: int,
    end_line: int,
) -> ToolResult:
    try:
        art, src = store.read_slice(source_ref, start_line=start_line, end_line=end_line)
        sliced = slice_lines_from_numbered_block(src, start_line, end_line)
        ref = store.save_text(sliced)
        return ToolResult(ok=True, result_ref=ref)
    except Exception as e:
        return ToolResult(ok=False, error=str(e))

_LINE_RE = re.compile(r"^\s*(\d+):\s?(.*)$")

def parse_numbered_lines(block_text: str) -> Dict[int, str]:
    """
    Парсит формат:
      '  14: Статья 1....'
    Возвращает {14: 'Статья 1....', 15: '', ...}
    Игнорирует шапки/мусорные строки, которые не начинаются с 'N:'.
    """
    lines: Dict[int, str] = {}
    for raw in block_text.splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            continue
        n = int(m.group(1))
        lines[n] = m.group(2)  # может быть пустой строкой
    return lines

def slice_lines_from_numbered_block(
    block_text: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    Возвращает текст строк start_line..end_line (включительно) без 'N:' префиксов.
    Если в артефакте не хватает нужных строк (обрезка/лимит), кидает IncompleteArtifactRange.
    """
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")

    m = parse_numbered_lines(block_text)
    missing = [i for i in range(start_line, end_line + 1) if i not in m]
    if missing:
        # значит артефакт не содержит весь диапазон (лимит вывода или кусок не тот)
        raise IncompleteArtifactRange(f"Missing lines: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    out = []
    for i in range(start_line, end_line + 1):
        out.append(m[i])
    return "\n".join(out).rstrip() + "\n"


def find_matches_in_numbered_block(
    block_text: str,
    pattern: str,
    flags: int = 0,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """
    Ищет regex по "чистым" строкам (без 'N:').
    Возвращает список (line_no, matched_text_line).
    """
    rx = re.compile(pattern, flags)
    m = parse_numbered_lines(block_text)

    keys = sorted(m.keys())
    if not keys:
        return []

    lo = start_line if start_line is not None else keys[0]
    hi = end_line if end_line is not None else keys[-1]

    hits: List[Tuple[int, str]] = []
    for ln in range(lo, hi + 1):
        if ln not in m:
            continue
        line = m[ln]
        if rx.search(line):
            hits.append((ln, line))
    return hits

def tool_find_matches_in_artifact(
    store: ArtifactStore,
    source_ref: str,
    pattern: str,
    flags: int = 0,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> ToolResult:
    try:
        src = store.get(source_ref)
        hits = find_matches_in_numbered_block(src, pattern, flags=flags, start_line=start_line, end_line=end_line)
        return ToolResult(
            ok=True,
            matches=[{"line": ln, "text": text} for ln, text in hits]
        )
    except Exception as e:
        return ToolResult(ok=False, error=str(e))


def tool_write_file_from_artifact(
    store: ArtifactStore,
    content_ref: str,
    file_path: str,
    encoding: str = "utf-8",
    mkdirs: bool = True,
) -> ToolResult:
    try:
        text = store.load(content_ref)
        p = Path(file_path)
        if mkdirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding=encoding)
        return ToolResult(ok=True)
    except Exception as e:
        return ToolResult(ok=False, error=str(e))
