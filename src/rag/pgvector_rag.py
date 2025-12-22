# pgvector_rag.py (новая версия)
from typing import List, Dict, Any

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from src.rag.models import Chunk, Base, MemoryChunk, MemoryRecordRow, MemoryRecordRow1
import toml

from src.task import MemoryRecord

TABLE_NAME = "code_chunks"
MEMORY_TABLE_NAME = "memory_chunks"
MEMORY_RECORD_TABLE_NAME = "memory_records"

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

    def memory_record_init_db(self):
        Base.metadata.create_all(self.engine)
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{MEMORY_RECORD_TABLE_NAME}_embedding 
                ON {MEMORY_RECORD_TABLE_NAME} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
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

    def search_memory_record(self,
                             query_embedding: List[float],
                             top_k: int = 10,
                             max_distance: float = 0.1) -> List[MemoryRecordRow]:
        """
        Поиск по косинусному расстоянию с использованием SQLAlchemy ORM
        """
        with Session(self.engine) as session:
            distance_expr = MemoryRecordRow.embedding.cosine_distance(query_embedding)

            stmt = (
                select(
                    MemoryRecordRow.id,
                    MemoryRecordRow.title,
                    MemoryRecordRow.task,
                    MemoryRecordRow.solution_outline,
                    MemoryRecordRow.key_decisions,
                    MemoryRecordRow.artifacts,
                    MemoryRecordRow.verification,
                    MemoryRecordRow.reusable_patterns,
                    MemoryRecordRow.embedding,
                    MemoryRecordRow.success,
                    MemoryRecordRow.created_at,
                    MemoryRecordRow.updated_at,
                    distance_expr.label("distance")
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
                MemoryRecordRow(
                    id = row.id,
                    title = row.title,
                    task = row.task,
                    solution_outline = row.solution_outline,
                    key_decisions = row.key_decisions,
                    artifacts = row.artifacts,
                    verification = row.verification,
                    reusable_patterns = row.reusable_patterns,
                    embedding = row.embedding,
                    success = row.success,
                    created_at = row.created_at,
                    updated_at = row.updated_at,
                )
                for row in rows
            ]

    def save_memory_record(self, record: MemoryRecordRow):
        self.session.merge(record)
        self.session.commit()

    def search_memory_record1(self,
                                 query_embedding: List[float],
                                 top_k: int = 10,
                                 max_distance: float = 0.1) -> List[MemoryRecordRow1]:
            """
            Поиск по косинусному расстоянию с использованием SQLAlchemy ORM
            """
            with Session(self.engine) as session:
                distance_expr = MemoryRecordRow1.embedding.cosine_distance(query_embedding)

                stmt = (
                    select(
                        MemoryRecordRow1.id,
                        MemoryRecordRow1.title,
                        MemoryRecordRow1.problem_signature,
                        MemoryRecordRow1.scope,
                        MemoryRecordRow1.domain,
                        MemoryRecordRow1.language,
                        MemoryRecordRow1.tags,
                        MemoryRecordRow1.stack,
                        MemoryRecordRow1.record_json,
                        MemoryRecordRow1.embedding,
                        MemoryRecordRow1.success,
                        MemoryRecordRow1.source,
                        MemoryRecordRow1.created_at,
                        MemoryRecordRow1.updated_at,
                        distance_expr.label("distance")
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
                    MemoryRecordRow1(
                        id = row.id,
                        title = row.title,
                        problem_signature = row.problem_signature,
                        scope = row.scope,
                        domain = row.domain,
                        language = row.language,
                        tags = row.tags,
                        stack = row.stack,
                        record_json = row.record_json,
                        embedding = row.embedding,
                        success = row.success,
                        source = row.source,
                        created_at = row.created_at,
                        updated_at = row.updated_at,
                    )
                    for row in rows
                ]

    def save_memory_record1(self, record1: MemoryRecordRow1):
        self.session.merge(record1)
        self.session.commit()

    def close(self):
        self.session.close()

    def merge(self, chunk: Chunk):
        self.session.merge(chunk)

    def commit(self):
        self.session.commit()
