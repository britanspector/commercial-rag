"""后台任务：评测与上传（内存 Job 表，单进程）。"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

_lock = threading.Lock()
_jobs: dict[str, JobRecord] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    id: str
    job_type: str  # eval_generation | eval_retrieval | eval_ragas | upload
    status: str = "pending"  # pending | running | success | failed
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def create_job(job_type: str) -> JobRecord:
    job_id = uuid.uuid4().hex[:12]
    record = JobRecord(id=job_id, job_type=job_type)
    with _lock:
        _jobs[job_id] = record
    return record


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        return _jobs.get(job_id)


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in kwargs.items():
            setattr(job, key, value)


def run_job_async(job_id: str, worker: Callable[[], dict[str, Any]]) -> None:
    def _run() -> None:
        _set_job(job_id, status="running", started_at=_utcnow(), progress="running")
        try:
            result = worker()
            _set_job(
                job_id,
                status="success",
                finished_at=_utcnow(),
                result=result,
                progress="done",
            )
        except Exception as exc:
            _set_job(
                job_id,
                status="failed",
                finished_at=_utcnow(),
                error=str(exc),
                progress="failed",
                result={"traceback": traceback.format_exc()[-2000:]},
            )

    thread = threading.Thread(target=_run, name=f"job-{job_id}", daemon=True)
    thread.start()


def job_to_dict(job: JobRecord) -> dict[str, Any]:
    duration_ms = None
    if job.started_at and job.finished_at:
        duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000)
    return {
        "job_id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "duration_ms": duration_ms,
        "result": job.result,
        "error": job.error,
    }


def wait_job(job_id: str, timeout_s: float = 0.0) -> JobRecord | None:
    """可选阻塞等待（测试用）。"""
    deadline = time.time() + timeout_s
    while True:
        job = get_job(job_id)
        if job is None:
            return None
        if job.status in {"success", "failed"}:
            return job
        if timeout_s <= 0:
            return job
        if time.time() >= deadline:
            return job
        time.sleep(0.5)
