"""SQLAlchemy ORM models for query telemetry and document tracking."""
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    """One row per /query request for metrics and debugging."""
    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    total_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieve_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    generate_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QAPair(Base):
    """Auto-generated Q&A pairs from ingested document chunks for evaluation."""
    __tablename__ = "qa_pair"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    knowledge_base: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    chunk_id: Mapped[str] = mapped_column(String(512), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Tracks ingested documents and their knowledge base membership."""
    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("doc_name", "knowledge_base", name="uq_doc_kb"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_name: Mapped[str] = mapped_column(String(512), nullable=False)
    knowledge_base: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chunks_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
