"""FastAPI 应用：/search 与 /chat 复用 RAGPipeline。"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

CURRENT_DIR = Path(__file__).parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()

from api.audit import (
    init_audit_on_startup,
    log_chat_success,
    log_request_error,
    log_search_success,
    log_upload_error,
    log_upload_success,
)
from api.deps import close_pipeline, get_pipeline, pipeline_status, reload_pipeline
from db.config import is_audit_enabled
from db.engine import db_status
from db.tracker import get_tracker
from api.jobs import create_job, get_job, job_to_dict, run_job_async
from api.schemas import (
    ChatResponse,
    EvalJobRequest,
    HealthResponse,
    JobStatusResponse,
    RAGRequest,
    SearchResponse,
    UploadResponse,
)
from api.serializers import chat_result_to_response, ingest_result_to_response, search_result_to_response
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
    init_audit_on_startup()
    try:
        yield
    finally:
        close_pipeline(app)


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    status = pipeline_status(request.app)
    audit = {"enabled": is_audit_enabled(), **db_status()} if is_audit_enabled() else {"enabled": False}
    return HealthResponse(
        status="ok",
        pipeline_ready=status["pipeline_initialized"],
        models_loaded=status["models_loaded"],
        audit=audit,
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

    tracker = get_tracker()
    with tracker.request_context("search") as request_id:
        try:
            result = await run_in_threadpool(pipeline.run_search, rag_query, config=req_config)
        except FileNotFoundError as exc:
            log_request_error(request_id, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except MemoryError as exc:
            log_request_error(request_id, exc)
            raise HTTPException(
                status_code=503,
                detail="内存不足，无法加载 Embedding/Reranker 模型。建议扩容至 ≥8GB 或配置 swap。",
            ) from exc
        except Exception as exc:
            log_request_error(request_id, exc)
            raise HTTPException(status_code=500, detail=f"检索失败：{exc}") from exc

        log_search_success(request_id, body, pipeline, req_config, result)
        return search_result_to_response(result)


@app.post("/chat", response_model=ChatResponse)
async def chat(body: RAGRequest, request: Request) -> ChatResponse:
    """完整问答：检索 → 重排 → 证据校验 → 生成答案与引用。"""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    pipeline = get_pipeline(request.app)
    req_config = _build_request_config(pipeline, body)
    rag_query = _build_rag_query(body)

    tracker = get_tracker()
    with tracker.request_context("chat") as request_id:
        try:
            result = await run_in_threadpool(pipeline.run, rag_query, config=req_config)
        except FileNotFoundError as exc:
            log_request_error(request_id, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except MemoryError as exc:
            log_request_error(request_id, exc)
            raise HTTPException(
                status_code=503,
                detail="内存不足，无法加载 Embedding/Reranker 模型。建议扩容至 ≥8GB 或配置 swap。",
            ) from exc
        except Exception as exc:
            log_request_error(request_id, exc)
            raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc

        log_chat_success(request_id, body, req_config, result)
        return chat_result_to_response(result)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """查询后台任务状态（/upload?background=1 或 POST /eval）。"""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{job_id}")
    return JobStatusResponse(**job_to_dict(job))


@app.post("/eval", response_model=JobStatusResponse)
async def start_eval(body: EvalJobRequest) -> JobStatusResponse:
    """
    异步启动批量评测（等价 CLI：eval_generation / eval_retrieval / eval_ragas）。
    立即返回 job_id，通过 GET /jobs/{job_id} 查询进度与输出路径。
    """
    from eval_runner import run_generation_eval_job, run_ragas_eval_job, run_retrieval_eval_job

    job = create_job(f"eval_{body.eval_type}")

    def _worker() -> dict:
        if body.eval_type == "generation":
            return run_generation_eval_job(
                limit=body.limit,
                skip_ragas=body.skip_ragas,
                save_detail=body.save_detail,
                resume=body.resume,
            )
        if body.eval_type == "retrieval":
            return run_retrieval_eval_job(
                compare_routes=body.compare_routes,
                route=body.route,
                top_k=body.top_k,
                legacy_retriever=body.legacy_retriever,
                pipeline_stage=body.pipeline_stage,
            )
        return run_ragas_eval_job(resume=body.resume, limit=body.limit)

    run_job_async(job.id, _worker)
    return JobStatusResponse(**job_to_dict(job))


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(..., description="研报 PDF 文件"),
    industry: str = Form(default="", description="行业子目录，如 semi-conductor；默认 uploads"),
    industry_label: str = Form(default="", description="行业中文标签，如 半导体"),
    replace_existing: bool = Form(default=True, description="同 doc_id 是否覆盖旧数据"),
    background: bool = Form(default=False, description="true 时异步入库，返回 job_id"),
) -> UploadResponse:
    """
    上传 PDF 并自动完成：MinerU 解析 → 分块 → 向量化 → Milvus + BM25 入库。

    大文件建议 background=true，通过 GET /jobs/{job_id} 查询进度。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件过大（上限 100MB）")

    from pipeline.ingest import ingest_pdf_bytes

    if background:
        job = create_job("upload")
        filename = file.filename
        app_ref = request.app

        def _upload_worker() -> dict:
            result = ingest_pdf_bytes(
                content,
                filename,
                industry=industry,
                industry_label=industry_label,
                replace_existing=replace_existing,
            )
            reload_pipeline(app_ref)
            payload = ingest_result_to_response(result)
            return {
                "doc_id": payload.doc_id,
                "chunk_count": payload.chunk_count,
                "milvus_rows_inserted": payload.milvus_rows_inserted,
            }

        run_job_async(job.id, _upload_worker)
        return UploadResponse(
            doc_id="",
            filename=filename,
            industry=industry,
            industry_label=industry_label,
            source_pdf_path="",
            chunk_count=0,
            retrievable_chunk_count=0,
            milvus_rows_inserted=0,
            milvus_total_rows=0,
            bm25_total_chunks=0,
            replaced_existing=replace_existing,
            stages=[],
            job_id=job.id,
            async_mode=True,
        )

    tracker = get_tracker()
    with tracker.request_context("upload") as request_id:
        try:
            result = await run_in_threadpool(
                ingest_pdf_bytes,
                content,
                file.filename,
                industry=industry,
                industry_label=industry_label,
                replace_existing=replace_existing,
            )
        except ValueError as exc:
            log_upload_error(request_id, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            log_upload_error(request_id, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            log_upload_error(request_id, exc)
            raise HTTPException(status_code=500, detail=f"PDF 解析失败：{exc}") from exc
        except MemoryError as exc:
            log_upload_error(request_id, exc)
            raise HTTPException(
                status_code=503,
                detail="内存不足，无法完成解析或向量化。建议扩容或改用 CPU 解析。",
            ) from exc
        except Exception as exc:
            log_upload_error(request_id, exc)
            raise HTTPException(status_code=500, detail=f"入库失败：{exc}") from exc

        log_upload_success(request_id, result)
        reload_pipeline(request.app)
        return ingest_result_to_response(result)
