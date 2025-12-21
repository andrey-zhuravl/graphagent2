# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Index, Boolean, ARRAY, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "code_chunks"
    id = Column(Integer, primary_key=True)
    file_path = Column(String, index=True)  # абсолютный путь
    source = Column(String, index=True)  # относительный путь
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768))  # nomic-embed-text-v1.5
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Для детерминированных обновлений
    __table_args__ = (
        Index('ix_unique_chunk', 'file_path', 'chunk_index', unique=True),
    )

class MemoryChunk(Base):
    __tablename__ = "memory_chunks"
    id = Column(Integer, primary_key=True)
    # Короткое описание ситуации (сырой текст ситуационного контекста)
    situation = Column(Text, nullable=False)
    # Что агент сделал (action + описание)
    action_description = Column(Text, nullable=False)
    # Что получилось — кратко
    result_summary = Column(Text, nullable=False)
    # Вся мысль (reasoning + chain)
    reasoning = Column(Text)
    # План действий, который привёл к успеху
    action_plan = Column(Text)
    # Чанк для поиска похожих рассуждений
    embedding = Column(Vector(768))
    # Метаданные
    success = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class MemoryRecordRow(Base):
    __tablename__ = "memory_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # anchors
    title = Column(String, nullable=False, default="")
    task = Column(String, nullable=False, default="")
    solution_outline = Column(ARRAY(String), nullable=False)
    key_decisions = Column(ARRAY(String), nullable=False)
    artifacts = Column(ARRAY(String), nullable=False)
    verification = Column(ARRAY(String), nullable=False)
    reusable_patterns = Column(ARRAY(String), nullable=False)
    tags = Column(ARRAY(String), nullable=False, default=list)
    embedding = Column(Vector(768), nullable=False)
    success = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class MemoryRecordRow1(Base):
    __tablename__ = "memory_records1"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # anchors
    title = Column(String, nullable=False, default="")
    problem_signature = Column(String, nullable=False, default="")

    scope = Column(String, nullable=False, default="global")
    domain = Column(String, nullable=False, default="")
    language = Column(String, nullable=False, default="any")

    # Важно: default должен быть callable, чтобы не шарить один список между объектами
    tags = Column(ARRAY(String), nullable=False, default=list)
    stack = Column(ARRAY(String), nullable=False, default=list)

    record_json = Column(JSONB, nullable=False)
    embedding = Column(Vector(768), nullable=False)

    success = Column(Boolean, nullable=False, default=True)
    source = Column(String, nullable=False, default="agent")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("problem_signature", "scope", "domain",
                         name="uq_memrec_signature_scope_domain"),
        Index("ix_memrec_scope_domain", "scope", "domain"),
        Index("ix_memrec_language", "language"),
    )