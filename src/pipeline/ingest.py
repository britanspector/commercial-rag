"""
ingest：单份 PDF 上传入库（解析 → 分块 → 向量化 → Milvus + BM25）。

复用 parse_pdf_mineru、chunk_mineru、embed_chunks、build_bm25_index 的核心逻辑，
增量更新 JSONL 元数据与索引，不重建全库。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache

bootstrap_hf_cache()

from bm25_store import BM25ChunkIndex, DEFAULT_CHUNKS_JSONL, DEFAULT_INDEX_PATH
from chunk_mineru import OUTPUT_CHUNKS_JSONL, chunk_single_document, get_tokenizer
from embed_chunks import embed_chunk_records, insert_vectors_for_doc
from parse_pdf_mineru import (
    OUTPUT_DOCUMENTS_JSONL,
    OUTPUT_SUMMARY_CSV,
    build_summary_row,
    find_content_list_v2,
    load_summary_by_doc_id,
    parse_single_pdf,
    resolve_runtime_config,
    save_summary_csv,
)
from pdf_paths import (
    DOC_MANIFEST_JSONL,
    INDUSTRY_LABELS,
    INPUT_PDF_DIR,
    PROJECT_ROOT,
    PdfSource,
    get_doc_manifest,
)

DEFAULT_UPLOAD_INDUSTRY = "uploads"
_FILENAME_SAFE_PATTERN = re.compile(r"[^\w.\-()\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class IngestStage:
    name: str
    status: str  # success | failed | skipped
    detail: str = ""


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    industry: str
    industry_label: str
    source_pdf_path: str
    display_name: str = ""
    company_name: str = ""
    stock_code: str = ""
    chunk_count: int = 0
    retrievable_chunk_count: int = 0
    milvus_rows_inserted: int = 0
    milvus_total_rows: int = 0
    bm25_total_chunks: int = 0
    replaced_existing: bool = False
    stages: list[IngestStage] = field(default_factory=list)
    chunk_records: list[dict] = field(default_factory=list)


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name.lower().endswith(".pdf"):
        raise ValueError("仅支持 PDF 文件（.pdf）")
    stem = Path(name).stem
    suffix = Path(name).suffix
    safe_stem = _FILENAME_SAFE_PATTERN.sub("_", stem).strip("._")
    if not safe_stem:
        raise ValueError("文件名无效")
    return f"{safe_stem}{suffix}"


def _resolve_industry(industry: str, industry_label: str) -> tuple[str, str]:
    folder = (industry or DEFAULT_UPLOAD_INDUSTRY).strip().strip("/")
    label = (industry_label or INDUSTRY_LABELS.get(folder, folder)).strip()
    return folder, label


def _load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _upsert_jsonl_by_doc_id(path: Path, record: dict, doc_id: str) -> bool:
    """写入 JSONL；若 doc_id 已存在则替换。返回是否替换了旧记录。"""
    records = _load_jsonl_records(path)
    replaced = any(item.get("doc_id") == doc_id for item in records)
    records = [item for item in records if item.get("doc_id") != doc_id]
    records.append(record)
    _write_jsonl_records(path, records)
    return replaced


def _replace_doc_chunks(doc_id: str, new_chunks: list[dict]) -> bool:
    """替换 chunks.jsonl 中指定 doc_id 的分块。返回是否替换了旧分块。"""
    records = _load_jsonl_records(OUTPUT_CHUNKS_JSONL)
    replaced = any(item.get("doc_id") == doc_id for item in records)
    kept = [item for item in records if item.get("doc_id") != doc_id]
    kept.extend(new_chunks)
    _write_jsonl_records(OUTPUT_CHUNKS_JSONL, kept)
    return replaced


def _upsert_doc_manifest(source: PdfSource) -> None:
    record = {
        "doc_id": source.doc_id,
        "filename": source.filename,
        "industry": source.industry,
        "industry_label": source.industry_label,
        "source_pdf_path": source.source_pdf_path,
    }
    _upsert_jsonl_by_doc_id(DOC_MANIFEST_JSONL, record, source.doc_id)


def _doc_exists(doc_id: str) -> bool:
    manifest = get_doc_manifest()
    if doc_id in manifest:
        return True
    return any(item.get("doc_id") == doc_id for item in _load_jsonl_records(OUTPUT_DOCUMENTS_JSONL))


def save_pdf_bytes(
    content: bytes,
    filename: str,
    *,
    industry: str = "",
    industry_label: str = "",
) -> PdfSource:
    """保存上传 PDF 到 data/raw_pdfs/<industry>/。"""
    if not content:
        raise ValueError("上传文件为空")

    safe_name = _sanitize_filename(filename)
    industry_folder, resolved_label = _resolve_industry(industry, industry_label)
    target_dir = INPUT_PDF_DIR / industry_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name
    target_path.write_bytes(content)

    rel_path = target_path.relative_to(PROJECT_ROOT).as_posix()
    return PdfSource(
        path=target_path,
        doc_id=target_path.stem,
        filename=safe_name,
        industry=industry_folder,
        industry_label=resolved_label,
        source_pdf_path=rel_path,
    )


def ingest_pdf_file(
    pdf_path: Path,
    *,
    industry: str = "",
    industry_label: str = "",
    replace_existing: bool = True,
) -> IngestResult:
    """从本地 PDF 路径执行完整入库流程。"""
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")

    industry_folder, resolved_label = _resolve_industry(industry, industry_label)
    rel_path = pdf_path.relative_to(PROJECT_ROOT).as_posix() if pdf_path.is_relative_to(PROJECT_ROOT) else str(pdf_path)
    source = PdfSource(
        path=pdf_path,
        doc_id=pdf_path.stem,
        filename=pdf_path.name,
        industry=industry_folder,
        industry_label=resolved_label,
        source_pdf_path=rel_path,
    )
    return ingest_pdf_source(source, replace_existing=replace_existing)


def ingest_pdf_bytes(
    content: bytes,
    filename: str,
    *,
    industry: str = "",
    industry_label: str = "",
    replace_existing: bool = True,
) -> IngestResult:
    """保存上传 PDF 并执行完整入库流程。"""
    source = save_pdf_bytes(
        content,
        filename,
        industry=industry,
        industry_label=industry_label,
    )
    return ingest_pdf_source(source, replace_existing=replace_existing)


def ingest_pdf_source(
    source: PdfSource,
    *,
    replace_existing: bool = True,
) -> IngestResult:
    """对已落盘的 PdfSource 执行 parse → chunk → embed → 索引更新。"""
    result = IngestResult(
        doc_id=source.doc_id,
        filename=source.filename,
        industry=source.industry,
        industry_label=source.industry_label,
        source_pdf_path=source.source_pdf_path,
        replaced_existing=_doc_exists(source.doc_id) if replace_existing else False,
    )

    # --- 1. 解析 ---
    try:
        device, backend, mineru_env, _runtime = resolve_runtime_config()
        doc_record = parse_single_pdf(source, backend, device, mineru_env)
        _upsert_jsonl_by_doc_id(OUTPUT_DOCUMENTS_JSONL, doc_record, source.doc_id)

        summary_by_doc_id = load_summary_by_doc_id()
        summary_by_doc_id[source.doc_id] = build_summary_row(
            source,
            status="success",
            backend=backend,
            device=device,
            text_char_count=doc_record.get("text_char_count", 0),
            markdown_path=doc_record.get("markdown_path", ""),
            metadata=doc_record.get("metadata") or {},
        )
        save_summary_csv(summary_by_doc_id)
        _upsert_doc_manifest(source)

        result.stages.append(
            IngestStage(
                name="parse",
                status="success",
                detail=f"backend={backend}, device={device}",
            )
        )
    except Exception as error:
        result.stages.append(IngestStage(name="parse", status="failed", detail=str(error)))
        raise

    # --- 2. 分块 ---
    try:
        content_list_path = find_content_list_v2(source.doc_id)
        if content_list_path is None:
            raise FileNotFoundError(f"MinerU 未生成 content_list_v2：{source.doc_id}")

        get_tokenizer()
        doc_manifest = get_doc_manifest()
        doc_manifest[source.doc_id] = {
            "doc_id": source.doc_id,
            "filename": source.filename,
            "industry": source.industry,
            "industry_label": source.industry_label,
            "source_pdf_path": source.source_pdf_path,
        }
        chunk_records = chunk_single_document(content_list_path, doc_manifest)
        _replace_doc_chunks(source.doc_id, chunk_records)

        retrievable = [record for record in chunk_records if record.get("is_retrievable", True)]
        result.chunk_records = chunk_records
        result.chunk_count = len(chunk_records)
        result.retrievable_chunk_count = len(retrievable)
        if chunk_records:
            result.display_name = str(chunk_records[0].get("display_name", ""))
            result.company_name = str(chunk_records[0].get("company_name", ""))
            result.stock_code = str(chunk_records[0].get("stock_code", ""))

        result.stages.append(
            IngestStage(
                name="chunk",
                status="success",
                detail=f"total={len(chunk_records)}, retrievable={len(retrievable)}",
            )
        )
    except Exception as error:
        result.stages.append(IngestStage(name="chunk", status="failed", detail=str(error)))
        raise

    if not retrievable:
        result.stages.append(
            IngestStage(name="embed", status="skipped", detail="无可检索 chunk")
        )
        result.stages.append(
            IngestStage(name="bm25", status="skipped", detail="无可检索 chunk")
        )
        return result

    # --- 3. 向量化 + Milvus ---
    try:
        milvus_rows, _embedder = embed_chunk_records(retrievable)
        total_rows = insert_vectors_for_doc(
            milvus_rows,
            doc_id=source.doc_id,
            replace_existing=replace_existing,
        )
        result.milvus_rows_inserted = len(milvus_rows)
        result.milvus_total_rows = total_rows
        result.stages.append(
            IngestStage(
                name="embed",
                status="success",
                detail=f"inserted={len(milvus_rows)}, milvus_total={total_rows}",
            )
        )
    except Exception as error:
        result.stages.append(IngestStage(name="embed", status="failed", detail=str(error)))
        raise

    # --- 4. BM25 全量重建（基于更新后的 chunks.jsonl）---
    try:
        if not DEFAULT_CHUNKS_JSONL.exists():
            raise FileNotFoundError(f"chunks 文件不存在：{DEFAULT_CHUNKS_JSONL}")
        index = BM25ChunkIndex.build_from_chunks(DEFAULT_CHUNKS_JSONL)
        index.save(DEFAULT_INDEX_PATH)
        result.bm25_total_chunks = len(index.chunk_ids)
        result.stages.append(
            IngestStage(
                name="bm25",
                status="success",
                detail=f"total_chunks={len(index.chunk_ids)}",
            )
        )
    except Exception as error:
        result.stages.append(IngestStage(name="bm25", status="failed", detail=str(error)))
        raise

    try:
        from cache.invalidate_hooks import on_corpus_updated

        on_corpus_updated(doc_id=source.doc_id)
    except Exception as error:
        import logging

        logging.getLogger(__name__).warning(
            "cache invalidation after ingest skipped doc_id=%s: %s",
            source.doc_id,
            error,
        )

    return result
