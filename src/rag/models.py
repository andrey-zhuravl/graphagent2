# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Index, Boolean
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