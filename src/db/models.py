"""RAG 审计库表结构（SQLAlchemy 2.x）。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RagRequest(Base):
    """每次 /upload、/search、/chat 对应一条请求记录。"""

    __tablename__ = "rag_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_type: Mapped[str] = mapped_column(String(16), nullable=False)  # upload | search | chat
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    error_detail: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    query_log: Mapped[QueryLog | None] = relationship(back_populates="request", uselist=False)
    upload_log: Mapped[UploadLog | None] = relationship(back_populates="request", uselist=False)
    chat_answer: Mapped[ChatAnswer | None] = relationship(back_populates="request", uselist=False)
    refusal_record: Mapped[RefusalRecord | None] = relationship(back_populates="request", uselist=False)
    retrieval_hits: Mapped[list[RetrievalHit]] = relationship(back_populates="request")


class Document(Base):
    """文档元数据（上传时 upsert）。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), default="")
    industry: Mapped[str] = mapped_column(String(64), default="")
    industry_label: Mapped[str] = mapped_column(String(64), default="")
    source_pdf_path: Mapped[str] = mapped_column(String(512), default="")
    display_name: Mapped[str] = mapped_column(String(256), default="")
    company_name: Mapped[str] = mapped_column(String(128), default="")
    stock_code: Mapped[str] = mapped_column(String(32), default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    retrievable_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    milvus_total_rows: Mapped[int] = mapped_column(Integer, default=0)
    bm25_total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    chunks: Mapped[list[ChunkRecord]] = relationship(back_populates="document")


class ChunkRecord(Base):
    """分块元数据（上传入库时同步；正文仅存预览）。"""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunks_chunk_id"),
        Index("ix_chunks_doc_id", "doc_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False)
    doc_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    company_name: Mapped[str] = mapped_column(String(128), default="")
    stock_code: Mapped[str] = mapped_column(String(32), default="")
    section_title: Mapped[str] = mapped_column(String(512), default="")
    page_start: Mapped[int] = mapped_column(Integer, default=0)
    page_end: Mapped[int] = mapped_column(Integer, default=0)
    content_type: Mapped[str] = mapped_column(String(64), default="")
    is_retrievable: Mapped[bool] = mapped_column(Boolean, default=True)
    text_preview: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class QueryLog(Base):
    """用户问题与检索配置（search / chat）。"""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rag_requests.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    original_query: Mapped[str] = mapped_column(Text, default="")
    rewritten_query: Mapped[str] = mapped_column(Text, default="")
    bm25_query: Mapped[str] = mapped_column(Text, default="")
    stock_code: Mapped[str] = mapped_column(String(32), default="")
    query_type: Mapped[str] = mapped_column(String(32), default="factual")
    recall_route: Mapped[str] = mapped_column(String(16), default="hybrid")
    recall_top_k: Mapped[int] = mapped_column(Integer, default=0)
    rerank_top_k: Mapped[int] = mapped_column(Integer, default=0)
    refusal_threshold: Mapped[float] = mapped_column(Float, default=0.0)
    hybrid_vector_weight: Mapped[float] = mapped_column(Float, default=0.0)

    request: Mapped[RagRequest] = relationship(back_populates="query_log")


class RetrievalHit(Base):
    """召回或重排片段（含页码与分数）。"""

    __tablename__ = "retrieval_hits"
    __table_args__ = (
        Index("ix_retrieval_hits_request_stage", "request_id", "stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rag_requests.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(16), nullable=False)  # recall | rerank
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(128), default="")
    doc_id: Mapped[str] = mapped_column(String(128), default="")
    company_name: Mapped[str] = mapped_column(String(128), default="")
    section_title: Mapped[str] = mapped_column(String(512), default="")
    page_start: Mapped[int] = mapped_column(Integer, default=0)
    page_end: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    score_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_rerank: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_vector: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_bm25: Mapped[float | None] = mapped_column(Float, nullable=True)
    text_preview: Mapped[str] = mapped_column(Text, default="")

    request: Mapped[RagRequest] = relationship(back_populates="retrieval_hits")


class ChatAnswer(Base):
    """问答最终答案与引用（仅 chat）。"""

    __tablename__ = "chat_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rag_requests.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    answer: Mapped[str] = mapped_column(Text, default="")
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    refusal_reason: Mapped[str] = mapped_column(String(64), default="")
    refusal_message: Mapped[str] = mapped_column(Text, default="")
    top_rerank_score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    evidence_check_json: Mapped[str] = mapped_column(Text, default="{}")

    request: Mapped[RagRequest] = relationship(back_populates="chat_answer")


class RefusalRecord(Base):
    """低分拒答专表，便于分析与报表。"""

    __tablename__ = "refusal_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rag_requests.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    query_text: Mapped[str] = mapped_column(Text, default="")
    refusal_reason: Mapped[str] = mapped_column(String(64), default="")
    refusal_message: Mapped[str] = mapped_column(Text, default="")
    top_rerank_score: Mapped[float] = mapped_column(Float, default=0.0)
    refusal_threshold: Mapped[float] = mapped_column(Float, default=0.0)
    recall_route: Mapped[str] = mapped_column(String(16), default="")
    evidence_check_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    request: Mapped[RagRequest] = relationship(back_populates="refusal_record")


class UploadLog(Base):
    """上传入库过程与阶段状态。"""

    __tablename__ = "upload_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rag_requests.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    doc_id: Mapped[str] = mapped_column(String(128), index=True, default="")
    filename: Mapped[str] = mapped_column(String(512), default="")
    industry: Mapped[str] = mapped_column(String(64), default="")
    industry_label: Mapped[str] = mapped_column(String(64), default="")
    source_pdf_path: Mapped[str] = mapped_column(String(512), default="")
    replaced_existing: Mapped[bool] = mapped_column(Boolean, default=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    retrievable_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    milvus_rows_inserted: Mapped[int] = mapped_column(Integer, default=0)
    milvus_total_rows: Mapped[int] = mapped_column(Integer, default=0)
    bm25_total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    stages_json: Mapped[str] = mapped_column(Text, default="[]")

    request: Mapped[RagRequest] = relationship(back_populates="upload_log")
