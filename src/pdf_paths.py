"""
raw_pdfs 目录结构与 PDF 发现逻辑。

目录约定：
    data/raw_pdfs/<industry_folder>/*.pdf

industry_folder 示例：semi-conductor / power-electronics / e-commercial
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

INPUT_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
DOC_MANIFEST_JSONL = PROJECT_ROOT / "data" / "parsed" / "doc_manifest.jsonl"

INDUSTRY_LABELS = {
    "semi-conductor": "半导体",
    "power-electronics": "电力",
    "e-commercial": "互联网电商",
}


@dataclass(frozen=True)
class PdfSource:
    path: Path
    doc_id: str
    filename: str
    industry: str
    industry_label: str
    source_pdf_path: str


def discover_pdf_files(root: Path | None = None) -> list[PdfSource]:
    root = root or INPUT_PDF_DIR
    if not root.exists():
        return []

    sources: list[PdfSource] = []
    for pdf_path in sorted(root.rglob("*.pdf")):
        industry = "" if pdf_path.parent == root else pdf_path.parent.name
        industry_label = INDUSTRY_LABELS.get(industry, industry)
        rel_path = pdf_path.relative_to(PROJECT_ROOT).as_posix()
        sources.append(
            PdfSource(
                path=pdf_path,
                doc_id=pdf_path.stem,
                filename=pdf_path.name,
                industry=industry,
                industry_label=industry_label,
                source_pdf_path=rel_path,
            )
        )
    return sources


def load_doc_manifest() -> dict[str, dict]:
    if not DOC_MANIFEST_JSONL.exists():
        return {}

    manifest: dict[str, dict] = {}
    with open(DOC_MANIFEST_JSONL, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            manifest[record["doc_id"]] = record
    return manifest


def get_doc_manifest() -> dict[str, dict]:
    manifest = load_doc_manifest()
    if manifest:
        return manifest

    return {
        source.doc_id: {
            "doc_id": source.doc_id,
            "filename": source.filename,
            "industry": source.industry,
            "industry_label": source.industry_label,
            "source_pdf_path": source.source_pdf_path,
        }
        for source in discover_pdf_files()
    }


def write_doc_manifest(sources: list[PdfSource]) -> None:
    DOC_MANIFEST_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC_MANIFEST_JSONL, "w", encoding="utf-8") as output_file:
        for source in sources:
            record = {
                "doc_id": source.doc_id,
                "filename": source.filename,
                "industry": source.industry,
                "industry_label": source.industry_label,
                "source_pdf_path": source.source_pdf_path,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_doc_display_name(
    *,
    company_name: str = "",
    stock_code: str = "",
    report_title: str = "",
    broker: str = "",
    report_date: str = "",
    rating: str = "",
    industry_label: str = "",
    filename: str = "",
) -> str:
    """生成面向用户的文档展示名（不依赖编号 filename）。"""
    headline_parts: list[str] = []
    if company_name:
        headline = company_name
        if stock_code:
            headline += f"（{stock_code}）"
        headline_parts.append(headline)

    if report_title and report_title not in headline_parts:
        if not headline_parts or report_title != company_name:
            headline_parts.append(report_title)

    title = " — ".join(headline_parts) if headline_parts else ""

    meta_parts: list[str] = []
    if industry_label:
        meta_parts.append(industry_label)
    if broker:
        meta_parts.append(broker)
    if report_date:
        meta_parts.append(report_date)
    if rating:
        meta_parts.append(rating)

    if title and meta_parts:
        return f"{title} [{', '.join(meta_parts)}]"
    if title:
        return title
    if meta_parts:
        return f"{filename or '未知文档'} [{', '.join(meta_parts)}]"
    return filename or "未知文档"
