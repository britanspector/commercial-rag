"""API 依赖：懒加载 RAGPipeline，避免 uvicorn 启动时拉起重模型导致 OOM。"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from rag_pipeline import RAGPipeline


_load_lock = threading.Lock()


def get_pipeline(app: FastAPI) -> RAGPipeline:
    """首次调用时加载 Pipeline 单例（模型仍在首次检索/问答时懒加载）。"""
    with _load_lock:
        if app.state.pipeline is None:
            from rag_pipeline import RAGPipeline, RAGPipelineConfig

            app.state.pipeline = RAGPipeline(RAGPipelineConfig())
        return app.state.pipeline


def close_pipeline(app: FastAPI) -> None:
    with _load_lock:
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None:
            pipeline.close()
            app.state.pipeline = None


def pipeline_status(app: FastAPI) -> dict[str, bool]:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        return {"pipeline_initialized": False, "models_loaded": False}
    return {
        "pipeline_initialized": True,
        "models_loaded": pipeline.is_loaded,
    }
