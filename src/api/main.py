"""FastAPI 应用：/search 与 /chat 复用 RAGPipeline。"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

CURRENT_DIR = Path(__file__).parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from api.deps import close_pipeline, get_pipeline, pipeline_status
from api.schemas import ChatResponse, HealthResponse, RAGRequest, SearchResponse
from api.serializers import chat_result_to_response, search_result_to_response
from rag_constants import (
    DEFAULT_RERANK_REFUSAL_THRESHOLD,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RECALL_TOP_K,
)
from rag_types import RAGQuery

if TYPE_CHECKING:
    from rag_pipeline import RAGPipeline, RAGPipelineConfig

API_TITLE = "commercial-rag API"
API_VERSION = "0.1.0"


def _build_request_config(pipeline: RAGPipeline, body: RAGRequest) -> RAGPipelineConfig:
    from rag_pipeline import RAGPipelineConfig
    from retrieval import RecallRoute

    base = pipeline.config
    return RAGPipelineConfig(
        recall_top_k=body.recall_top_k if body.recall_top_k is not None else base.recall_top_k,
        rerank_top_k=body.rerank_top_k if body.rerank_top_k is not None else base.rerank_top_k,
        refusal_threshold=(
            body.refusal_threshold if body.refusal_threshold is not None else base.refusal_threshold
        ),
        recall_route=RecallRoute(body.recall_route),
        hybrid_vector_weight=base.hybrid_vector_weight,
        hybrid_pool_size=base.hybrid_pool_size,
    )


def _build_rag_query(body: RAGRequest) -> RAGQuery:
    return RAGQuery(
        query=body.query.strip(),
        stock_code=body.stock_code.strip(),
        query_type=body.query_type,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    try:
        yield
    finally:
        close_pipeline(app)


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    status = pipeline_status(request.app)
    return HealthResponse(
        status="ok",
        pipeline_ready=status["pipeline_initialized"],
        models_loaded=status["models_loaded"],
        defaults={
            "recall_route": "hybrid",
            "recall_top_k": DEFAULT_RECALL_TOP_K,
            "rerank_top_k": DEFAULT_RERANK_TOP_K,
            "refusal_threshold": DEFAULT_RERANK_REFUSAL_THRESHOLD,
        },
    )


@app.post("/search", response_model=SearchResponse)
async def search(body: RAGRequest, request: Request) -> SearchResponse:
    """检索 + 重排，用于调试召回质量（不生成答案）。"""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    pipeline = get_pipeline(request.app)
    req_config = _build_request_config(pipeline, body)
    rag_query = _build_rag_query(body)

    try:
        result = await run_in_threadpool(pipeline.run_search, rag_query, config=req_config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(
            status_code=503,
            detail="内存不足，无法加载 Embedding/Reranker 模型。建议扩容至 ≥8GB 或配置 swap。",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"检索失败：{exc}") from exc

    return search_result_to_response(result)


@app.post("/chat", response_model=ChatResponse)
async def chat(body: RAGRequest, request: Request) -> ChatResponse:
    """完整问答：检索 → 重排 → 证据校验 → 生成答案与引用。"""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    pipeline = get_pipeline(request.app)
    req_config = _build_request_config(pipeline, body)
    rag_query = _build_rag_query(body)

    try:
        result = await run_in_threadpool(pipeline.run, rag_query, config=req_config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(
            status_code=503,
            detail="内存不足，无法加载 Embedding/Reranker 模型。建议扩容至 ≥8GB 或配置 swap。",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc

    return chat_result_to_response(result)
