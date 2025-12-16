# agent.py

import hashlib
import sys
from pathlib import Path
from typing import List, Dict

from src.rag.pgvector_rag import PgVectorRAG
from src.rag.models import Chunk, MemoryChunk
from src.utils.config import get_config_dict

sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
from tqdm import tqdm
import psycopg2

DB_URL = "postgresql+psycopg2://agent:agent123@localhost:5422/agentdb"
TABLE_NAME = "code_chunks"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
CHUNK_SIZE = 1000      # символов
CHUNK_OVERLAP = 200    # символов
IGNORE_DIRS = {".git", "__pycache__", "node_modules", "build", "dist", ".idea", ".venv", "chroma_db"}
IGNORE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".lock", ".log"}
# Адрес твоей запущенной Nomic-embed-text-v1.5
# LM Studio по умолчанию: http://localhost:1234/v1/embeddings
# ollama:                 http://localhost:11434/api/embeddings
EMBEDDING_URL = "http://192.168.1.12:1234/v1/embeddings"

# Заголовки и тело запроса для LM Studio
HEADERS = {"Content-Type": "application/json"}

# Base = declarative_base()


class Embedder:
    def __init__(self, model: str = "embedding_llm1"):
        self.pgVectorRAG = PgVectorRAG()
        self.config = get_config_dict()
        self.model = model

    # --------------------------------------------------------------
    # Функция получения эмбеддинга через твою запущенную модель
    # --------------------------------------------------------------
    def get_embedding(self, text: str) -> List[float]:
        payload = {
            "model": self.config[self.model]["model"],   # имя может быть любым, главное совпадает с тем, что в LM Studio
            "input": text
        }
        try:
            r = requests.post(EMBEDDING_URL, headers=HEADERS, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Ошибка эмбеддинга: {e}")
            return None

    def find_chunks(self, text: str, top_k: int = 3, max_distance: float = 0.1) -> List[Chunk]:
        embedding: List[float] = self.get_embedding(text)
        result = self.pgVectorRAG.search(embedding, top_k=top_k, max_distance = max_distance)
        print(f"Нашли {len(result)} чанков")
        return result

    def find_memory_chunks(self, query: str, top_k: int = 5, max_distance: float = 0.15) -> List[MemoryChunk]:
        embedding: List[float] = self.get_embedding(query)
        result = self.pgVectorRAG.search_memory(embedding, top_k=top_k, max_distance=max_distance)
        print(f"Нашли {len(result)} чанков")
        return result

    # def find_memory_chunks1(self, query: str, top_k: int = 5, max_distance: float = 0.15) -> List[MemoryChunk]:
    #     self.pgVectorRAG.memory_init_db()
    #     emb = self.get_embedding(query)
    #     if not emb:
    #         return []
    #
    #     with Session(self.engine) as session:
    #         distance_expr = MemoryChunk.embedding.cosine_distance(emb)
    #         stmt = (
    #             select(MemoryChunk)
    #             .order_by(distance_expr)
    #             .where(distance_expr <= max_distance)
    #             .limit(top_k)
    #         )
    #         rows = session.execute(stmt).scalars().all()
    #         return rows

    def save_memory_chunk(
            self,
            situation: str,  # Краткое описание текущей ситуации (для поиска)
            action_description: str,  # Что сделал: "вызвал create_file", "подумал о архитектуре"
            result_summary: str,  # "Успех: файл создан", "Провал: синтаксическая ошибка"
            reasoning: str = None,  # Полный текст рассуждений (опционально)
            action_plan: str = None,  # План действий из Thought
            success: bool = True
    ):
        emb = self.get_embedding(situation)
        if not emb:
            print("Не удалось получить эмбеддинг для памяти")
            return

        memory_chunk = MemoryChunk(
            situation=situation,
            action_description=action_description,
            result_summary=result_summary,
            reasoning=reasoning,
            action_plan=action_plan,
            embedding=emb,
            success=success
        )

        # Используем существующий pgVectorRAG
        self.pgVectorRAG.save_memory_chunk(memory_chunk)  # у тебя уже есть этот метод
        print(f"Сохранена память: {situation[:80]}...")

    # --------------------------------------------------------------
    # Утилиты
    # --------------------------------------------------------------
    def should_ignore(self, path: Path) -> bool:
        return any(part.startswith('.') for part in path.parts) or \
               path.suffix.lower() in IGNORE_EXT or \
               any(ignored in path.parts for ignored in IGNORE_DIRS)

    def text_to_chunk(self, text: str) -> list[str]:
        chunks = []
        i = 0
        while i < len(text):
            j = i + CHUNK_SIZE
            chunks.append(text[i:j])
            i = j - CHUNK_OVERLAP
            if j >= len(text): break
        return chunks

    def file_id(self, filepath: str) -> str:
        """Детерминированный ID по содержимому + пути (чтобы не дублировать)"""
        hasher = hashlib.md5()
        hasher.update(filepath.encode('utf-8'))
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

# --------------------------------------------------------------
# Основная функция сканирования
# --------------------------------------------------------------
# --------------------------------------------------------------
    def scan_directory(self, root_path: str):
        pgVectorRAG = PgVectorRAG(DB_URL)
        pgVectorRAG.init_db()
        root = Path(root_path).resolve()
        if not root.is_dir():
            print("Путь не папка")
            return

        added = files = 0

        for file_path in tqdm(list(root.rglob("*")), desc="Сканирование"):
            if not file_path.is_file() or self.should_ignore(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except:
                continue

            for idx, chunk_text in enumerate(self.text_to_chunk(content)):
                if len(chunk_text.strip()) < 50:
                    continue
                emb = self.get_embedding(chunk_text)
                if not emb:
                    continue

                pgVectorRAG.merge(Chunk(                 # merge = upsert, чтобы можно было пересканировать
                    file_path=str(file_path),
                    source=str(file_path.relative_to(root)),
                    chunk_index=idx,
                    content=chunk_text,
                    embedding=emb
                ))
                added += 1
                if added % 30 == 0:
                    pgVectorRAG.commit()

            files += 1

        pgVectorRAG.commit()
        pgVectorRAG.close()
        print(f"\nГотово! Файлов обработано: {files}, чанков добавлено/обновлено: {added}")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def print_help():
    print("Использование:")
    print("  python agent.py scan <путь_к_репозиторию>   — просканировать и добавить в Chroma")
    print("  python agent.py count                        — показать количество документов")

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "scan":
        print("Использование: python agent_pg.py scan ./путь_к_репозиторию")
        sys.exit(1)
    embedder = Embedder()
    embedder.scan_directory(sys.argv[2])