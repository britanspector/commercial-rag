"""API 层审计挂钩：在 /upload、/search、/chat 成功后写入数据库。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from db.config import is_audit_enabled
from db.tracker import get_tracker

if TYPE_CHECKING:
    from api.schemas import RAGRequest
    from pipeline.ingest import IngestResult
    from rag_pipeline import RAGPipeline, RAGPipelineConfig
    from rag_types import RAGPipelineResult, RAGQuery, RAGSearchResult

logger = logging.getLogger(__name__)


def init_audit_on_startup() -> None:
    if not is_audit_enabled():
        logger.info("RAG 审计已关闭（RAG_AUDIT_ENABLED=0）")
        return
    try:
        get_tracker().ensure_ready()
        logger.info("RAG 审计库已就绪")
    except Exception:
        logger.exception("RAG 审计库初始化失败，后续请求将跳过审计写入")


def log_upload_success(request_id: int | None, result: IngestResult) -> None:
    if request_id is None:
        return
    get_tracker().log_upload(request_id, result)


def log_upload_error(request_id: int | None, error: Exception) -> None:
    get_tracker().mark_error(request_id, str(error))


def log_search_success(
    request_id: int | None,
    body: RAGRequest,
    pipeline: RAGPipeline,
    req_config: RAGPipelineConfig,
    result: RAGSearchResult,
) -> None:
    if request_id is None:
        return
    rag_query = _body_to_rag_query(body)
    get_tracker().log_search(request_id, rag_query, req_config, result)


def log_chat_success(
    request_id: int | None,
    body: RAGRequest,
    req_config: RAGPipelineConfig,
    result: RAGPipelineResult,
) -> None:
    if request_id is None:
        return
    rag_query = _body_to_rag_query(body)
    get_tracker().log_chat(request_id, rag_query, req_config, result)


def log_request_error(request_id: int | None, error: Exception) -> None:
    get_tracker().mark_error(request_id, str(error))


def _body_to_rag_query(body: RAGRequest) -> RAGQuery:
    from rag_types import RAGQuery

    return RAGQuery(
        query=body.query.strip(),
        stock_code=body.stock_code.strip(),
        query_type=body.query_type,
    )
