"""审计写入：将 upload / search / chat 结果持久化到数据库。"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterator

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from db.config import is_audit_enabled
from db.engine import get_session_factory, init_db
from db.models import (
    ChatAnswer,
    ChunkRecord,
    Document,
    QueryLog,
    RagRequest,
    RefusalRecord,
    RetrievalHit,
    UploadLog,
)

if TYPE_CHECKING:
    from pipeline.ingest import IngestResult
    from rag_pipeline import RAGPipelineConfig
    from rag_types import Citation, RAGPipelineResult, RAGQuery, RAGSearchResult, RetrievedChunk

logger = logging.getLogger(__name__)

TEXT_PREVIEW_LIMIT = 800


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _preview(text: str, limit: int = TEXT_PREVIEW_LIMIT) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _chunk_text(record: dict) -> str:
    for key in ("embedding_text", "text", "content"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


class AuditTracker:
    """RAG 请求审计写入器（线程内每次请求独立 Session）。"""

    def __init__(self) -> None:
        self._initialized = False

    def ensure_ready(self) -> bool:
        if not is_audit_enabled():
            return False
        if not self._initialized:
            init_db()
            self._initialized = True
        return True

    @contextmanager
    def request_context(self, request_type: str) -> Iterator[int | None]:
        """开始一条请求记录；退出时标记 success / error。"""
        if not self.ensure_ready():
            yield None
            return

        started = time.perf_counter()
        session = get_session_factory()()
        request_id: int | None = None
        exc: BaseException | None = None
        try:
            row = RagRequest(request_type=request_type, status="running")
            session.add(row)
            session.commit()
            session.refresh(row)
            request_id = row.id
            yield request_id
        except BaseException as error:
            exc = error
            raise
        finally:
            try:
                if request_id is not None:
                    row = session.get(RagRequest, request_id)
                    if row is not None:
                        row.finished_at = _utcnow()
                        row.duration_ms = int((time.perf_counter() - started) * 1000)
                        if exc is not None:
                            row.status = "error"
                            row.error_detail = str(exc)[:4000]
                        elif row.status == "running":
                            row.status = "success"
                        session.commit()
            except Exception:
                session.rollback()
                logger.exception("结束请求审计失败 request_id=%s", request_id)
            finally:
                session.close()

    def mark_error(self, request_id: int | None, error_detail: str) -> None:
        if request_id is None or not self.ensure_ready():
            return
        session = get_session_factory()()
        try:
            row = session.get(RagRequest, request_id)
            if row is None:
                return
            row.status = "error"
            row.error_detail = (error_detail or "")[:4000]
            row.finished_at = _utcnow()
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("写入审计错误状态失败 request_id=%s", request_id)
        finally:
            session.close()

    def log_upload(
        self,
        request_id: int | None,
        result: IngestResult,
        *,
        chunk_records: list[dict] | None = None,
    ) -> None:
        if request_id is None or not self.ensure_ready():
            return

        session = get_session_factory()()
        try:
            stages_payload = [
                {"name": s.name, "status": s.status, "detail": s.detail} for s in result.stages
            ]
            session.add(
                UploadLog(
                    request_id=request_id,
                    doc_id=result.doc_id,
                    filename=result.filename,
                    industry=result.industry,
                    industry_label=result.industry_label,
                    source_pdf_path=result.source_pdf_path,
                    replaced_existing=result.replaced_existing,
                    chunk_count=result.chunk_count,
                    retrievable_chunk_count=result.retrievable_chunk_count,
                    milvus_rows_inserted=result.milvus_rows_inserted,
                    milvus_total_rows=result.milvus_total_rows,
                    bm25_total_chunks=result.bm25_total_chunks,
                    stages_json=json.dumps(stages_payload, ensure_ascii=False),
                )
            )
            records = (
                result.chunk_records
                if chunk_records is None
                else chunk_records
            )
            self._upsert_document(session, result, records)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("log_upload 失败 request_id=%s doc_id=%s", request_id, result.doc_id)
        finally:
            session.close()

    def _upsert_document(
        self,
        session: Session,
        result: IngestResult,
        chunk_records: list[dict],
    ) -> None:
        doc = session.execute(
            select(Document).where(Document.doc_id == result.doc_id)
        ).scalar_one_or_none()
        now = _utcnow()
        if doc is None:
            doc = Document(
                doc_id=result.doc_id,
                created_at=now,
                updated_at=now,
            )
            session.add(doc)

        doc.filename = result.filename
        doc.industry = result.industry
        doc.industry_label = result.industry_label
        doc.source_pdf_path = result.source_pdf_path
        doc.display_name = result.display_name
        doc.company_name = result.company_name
        doc.stock_code = result.stock_code
        doc.chunk_count = result.chunk_count
        doc.retrievable_chunk_count = result.retrievable_chunk_count
        doc.milvus_total_rows = result.milvus_total_rows
        doc.bm25_total_chunks = result.bm25_total_chunks
        doc.updated_at = now

        session.flush()
        session.execute(delete(ChunkRecord).where(ChunkRecord.doc_id == result.doc_id))
        for record in chunk_records:
            session.add(
                ChunkRecord(
                    chunk_id=str(record.get("chunk_id", "")),
                    doc_id=result.doc_id,
                    company_name=str(record.get("company_name", "")),
                    stock_code=str(record.get("stock_code", "")),
                    section_title=str(record.get("section_title", "")),
                    page_start=int(record.get("page_start") or 0),
                    page_end=int(record.get("page_end") or 0),
                    content_type=str(record.get("content_type", "")),
                    is_retrievable=bool(record.get("is_retrievable", True)),
                    text_preview=_preview(_chunk_text(record)),
                    updated_at=now,
                )
            )

    def log_search(
        self,
        request_id: int | None,
        rag_query: RAGQuery,
        config: RAGPipelineConfig,
        result: RAGSearchResult,
    ) -> None:
        if request_id is None or not self.ensure_ready():
            return
        session = get_session_factory()()
        try:
            rewrite = result.query_rewrite
            session.add(
                QueryLog(
                    request_id=request_id,
                    original_query=rewrite.original_query,
                    rewritten_query=rewrite.query,
                    bm25_query=rewrite.bm25_query,
                    stock_code=rewrite.stock_code or rag_query.stock_code,
                    query_type=rewrite.query_type or rag_query.query_type,
                    recall_route=config.recall_route.value
                    if hasattr(config.recall_route, "value")
                    else str(config.recall_route),
                    recall_top_k=config.recall_top_k,
                    rerank_top_k=config.rerank_top_k,
                    refusal_threshold=config.refusal_threshold,
                    hybrid_vector_weight=config.hybrid_vector_weight,
                )
            )
            self._add_hits(session, request_id, "recall", result.recall_hits)
            self._add_hits(session, request_id, "rerank", result.rerank_hits)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("log_search 失败 request_id=%s", request_id)
        finally:
            session.close()

    def log_chat(
        self,
        request_id: int | None,
        rag_query: RAGQuery,
        config: RAGPipelineConfig,
        result: RAGPipelineResult,
    ) -> None:
        if request_id is None or not self.ensure_ready():
            return
        session = get_session_factory()()
        try:
            rewrite = result.query_rewrite
            if rewrite is not None:
                session.add(
                    QueryLog(
                        request_id=request_id,
                        original_query=rewrite.original_query,
                        rewritten_query=rewrite.query,
                        bm25_query=rewrite.bm25_query,
                        stock_code=rewrite.stock_code or rag_query.stock_code,
                        query_type=rewrite.query_type or rag_query.query_type,
                        recall_route=config.recall_route.value
                        if hasattr(config.recall_route, "value")
                        else str(config.recall_route),
                        recall_top_k=config.recall_top_k,
                        rerank_top_k=config.rerank_top_k,
                        refusal_threshold=config.refusal_threshold,
                        hybrid_vector_weight=config.hybrid_vector_weight,
                    )
                )

            self._add_hits(session, request_id, "recall", result.recall_hits)
            self._add_hits(session, request_id, "rerank", result.rerank_hits)

            citations_json = json.dumps(
                [asdict(c) for c in result.citations],
                ensure_ascii=False,
            )
            evidence = result.evidence_check
            evidence_json = json.dumps(
                evidence.to_dict() if evidence else {},
                ensure_ascii=False,
            )
            refusal_message = (
                evidence.refusal_message if evidence and result.refused else ""
            )
            session.add(
                ChatAnswer(
                    request_id=request_id,
                    answer=result.answer,
                    refused=result.refused,
                    refusal_reason=result.refusal_reason,
                    refusal_message=refusal_message,
                    top_rerank_score=result.top_rerank_score,
                    evidence_passed=evidence.passed if evidence else not result.refused,
                    citation_count=len(result.citations),
                    citations_json=citations_json,
                    evidence_check_json=evidence_json,
                )
            )

            if result.refused:
                session.add(
                    RefusalRecord(
                        request_id=request_id,
                        query_text=result.query,
                        refusal_reason=result.refusal_reason,
                        refusal_message=refusal_message,
                        top_rerank_score=result.top_rerank_score,
                        refusal_threshold=config.refusal_threshold,
                        recall_route=config.recall_route.value
                        if hasattr(config.recall_route, "value")
                        else str(config.recall_route),
                        evidence_check_json=evidence_json,
                    )
                )

            session.commit()
        except Exception:
            session.rollback()
            logger.exception("log_chat 失败 request_id=%s", request_id)
        finally:
            session.close()

    def _add_hits(
        self,
        session: Session,
        request_id: int,
        stage: str,
        hits: list[RetrievedChunk],
    ) -> None:
        for hit in hits:
            session.add(
                RetrievalHit(
                    request_id=request_id,
                    stage=stage,
                    rank=hit.rank,
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    company_name=hit.company_name,
                    section_title=hit.section_title,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    score=hit.score,
                    score_recall=hit.score_recall,
                    score_rerank=hit.score_rerank,
                    score_vector=hit.score_vector,
                    score_bm25=hit.score_bm25,
                    text_preview=_preview(hit.text),
                )
            )


_tracker: AuditTracker | None = None


def get_tracker() -> AuditTracker:
    global _tracker
    if _tracker is None:
        _tracker = AuditTracker()
    return _tracker
