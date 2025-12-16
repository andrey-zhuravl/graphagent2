# pgvector_rag.py (новая версия)
from typing import List, Dict, Any

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from src.rag.models import Chunk, Base, MemoryChunk
import toml

TABLE_NAME = "code_chunks"
MEMORY_TABLE_NAME = "memory_chunks"

from dataclasses import dataclass

class PgVectorRAG:

    def __init__(self, db_url: str = None):
        if not db_url:
            config = toml.load("config.toml")
            db_url = config["memory"]["embedding_db_url"]  # "postgresql+psycopg2://..."

        self.engine = create_engine(db_url, future=True)
        self.session = Session(self.engine)

    def init_db(self):
        Base.metadata.create_all(self.engine)
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{TABLE_NAME}_embedding 
                ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
            """))
            conn.commit()

    def memory_init_db(self):
        Base.metadata.create_all(self.engine)
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{MEMORY_TABLE_NAME}_embedding 
                ON {MEMORY_TABLE_NAME} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
            """))
            conn.commit()

    def search(self, query_embedding: List[float], top_k: int = 10, max_distance: float = 0.1) -> List[Chunk]:
        """
        Поиск по косинусному расстоянию с использованием SQLAlchemy ORM
        """
        with Session(self.engine) as session:
            # 1. Создаем запрос с ORM-методом cosine_distance
            #    Это не raw SQL, а нативная конструкция pgvector.sqlalchemy
            distance_expr = Chunk.embedding.cosine_distance(query_embedding)

            stmt = (
                select(
                    Chunk.id,
                    Chunk.file_path,
                    Chunk.source,
                    Chunk.content,
                    Chunk.chunk_index,
                    distance_expr.label("distance")  # Присваиваем имя для доступа к результату
                )
                .order_by(distance_expr)  # Сортируем по расстоянию
                .limit(top_k)
            )

            # <-- добавляем фильтр по порогу, если он задан
            if max_distance is not None:
                stmt = stmt.where(distance_expr <= max_distance)

            # 2. Выполняем и получаем результаты
            rows = session.execute(stmt).all()

            # 3. Преобразуем в dict
            return [
                Chunk(
                    id = row.id,
                    file_path = row.file_path,
                    source = row.source,
                    content = row.content,
                    chunk_index = row.chunk_index,
                )
                for row in rows
            ]

    def search_memory(self, query_embedding: List[float], top_k: int = 10, max_distance: float = 0.1) -> List[MemoryChunk]:
        """
        Поиск по косинусному расстоянию с использованием SQLAlchemy ORM
        """
        with Session(self.engine) as session:
            # 1. Создаем запрос с ORM-методом cosine_distance
            #    Это не raw SQL, а нативная конструкция pgvector.sqlalchemy
            distance_expr = MemoryChunk.embedding.cosine_distance(query_embedding)

            stmt = (
                select(
                    MemoryChunk.id,
                    MemoryChunk.situation,
                    MemoryChunk.action_description,
                    MemoryChunk.result_summary,
                    MemoryChunk.reasoning,
                    MemoryChunk.action_plan,
                    MemoryChunk.embedding,
                    MemoryChunk.success,
                    MemoryChunk.created_at,
                    distance_expr.label("distance")  # Присваиваем имя для доступа к результату
                )
                .order_by(distance_expr)  # Сортируем по расстоянию
                .limit(top_k)
            )

            # <-- добавляем фильтр по порогу, если он задан
            if max_distance is not None:
                stmt = stmt.where(distance_expr <= max_distance)

            # 2. Выполняем и получаем результаты
            rows = session.execute(stmt).all()

            # 3. Преобразуем в dict
            return [
                MemoryChunk(
                    situation=row.situation,
                    action_description=row.action_description,
                    result_summary=row.result_summary,
                    reasoning=row.reasoning,
                    action_plan=row.action_plan,
                    embedding=row.embedding,
                    success=row.success
                )
                for row in rows
            ]


    def save_memory_chunk(self, chunk: MemoryChunk):
        self.session.merge(chunk)
        self.session.commit()

    def close(self):
        self.session.close()

    def merge(self, chunk: Chunk):
        self.session.merge(chunk)

    def commit(self):
        self.session.commit()
