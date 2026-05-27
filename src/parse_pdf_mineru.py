"""
方案 B：使用 MinerU 解析 PDF，输出 Markdown + documents.jsonl。

GPU 说明（重要）：
1. 新版 MinerU CLI 已移除 `-d/--device` 参数，设备通过环境变量 `MINERU_DEVICE_MODE` 控制。
2. 仅设置 MINERU_DEVICE=cuda 不够，还需安装带 CUDA 的 PyTorch（例如 2.12.0+cu124，而非 2.12.0+cpu）。
3. GPU 推理建议使用 `hybrid-auto-engine` 或 `pipeline` 后端。

依赖安装：
    pip install -r requirements.txt

GPU 额外步骤（仅示例，需按本机 GPU/驱动/CUDA 匹配版本）：
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pdf_paths import DOC_MANIFEST_JSONL, INPUT_PDF_DIR, PdfSource, discover_pdf_files, write_doc_manifest

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

MINERU_OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed" / "mineru"
OUTPUT_DOCUMENTS_JSONL = PROJECT_ROOT / "data" / "parsed" / "documents.jsonl"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "data" / "parsed" / "parse_summary.csv"

PARSE_METHOD = "mineru"

# 设备：cuda / cpu / auto（auto 会检测 PyTorch 是否支持 CUDA）
MINERU_DEVICE = "cuda"

# 后端：auto / pipeline / hybrid-auto-engine
# - auto：有 CUDA 时用 hybrid-auto-engine，否则 pipeline
# - pipeline：兼容性好，支持 CPU；有 CUDA 时也可走 GPU
# - hybrid-auto-engine：精度更高，需要 GPU + 足够显存（建议 8GB+）
MINERU_BACKEND = "pipeline"

MINERU_METHOD = "auto"
MINERU_LANG = "ch"

# 断点续跑：已有 content_list_v2 的 PDF 跳过；单份失败继续下一本
RESUME_SKIP_PARSED = True


def find_content_list_v2(doc_id: str) -> Path | None:
    doc_dir = MINERU_OUTPUT_DIR / doc_id
    if not doc_dir.exists():
        return None
    matches = sorted(doc_dir.rglob("*_content_list_v2.json"))
    return matches[0] if matches else None


def is_pdf_parse_complete(doc_id: str) -> bool:
    return find_content_list_v2(doc_id) is not None


def find_markdown_output(doc_id: str, pdf_stem: str) -> Path | None:
    doc_dir = MINERU_OUTPUT_DIR / doc_id
    if not doc_dir.exists():
        return None
    markdown_files = sorted(doc_dir.rglob("*.md"))
    if not markdown_files:
        return None
    preferred = [
        candidate
        for candidate in markdown_files
        if candidate.stem == pdf_stem or pdf_stem in candidate.stem
    ]
    return preferred[0] if preferred else markdown_files[0]


def load_parsed_doc_ids_from_jsonl() -> set[str]:
    if not OUTPUT_DOCUMENTS_JSONL.exists():
        return set()

    doc_ids: set[str] = set()
    with open(OUTPUT_DOCUMENTS_JSONL, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            doc_ids.add(json.loads(line)["doc_id"])
    return doc_ids


def append_document_record(record: dict) -> None:
    with open(OUTPUT_DOCUMENTS_JSONL, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_summary_by_doc_id() -> dict[str, dict]:
    if not OUTPUT_SUMMARY_CSV.exists():
        return {}

    summary_by_doc_id: dict[str, dict] = {}
    dataframe = pd.read_csv(OUTPUT_SUMMARY_CSV, encoding="utf-8-sig")
    for row in dataframe.to_dict(orient="records"):
        filename = str(row.get("filename", ""))
        doc_id = Path(filename).stem if filename else ""
        if doc_id:
            summary_by_doc_id[doc_id] = row
    return summary_by_doc_id


def save_summary_csv(summary_by_doc_id: dict[str, dict]) -> None:
    if not summary_by_doc_id:
        return
    rows = sorted(summary_by_doc_id.values(), key=lambda row: str(row.get("filename", "")))
    pd.DataFrame(rows).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")


def build_summary_row(
    pdf_source: PdfSource,
    *,
    status: str,
    backend: str,
    device: str,
    text_char_count: int = 0,
    markdown_path: str = "",
    metadata: dict | None = None,
    error: str = "",
) -> dict:
    return {
        "doc_id": pdf_source.doc_id,
        "filename": pdf_source.filename,
        "industry": pdf_source.industry_label or pdf_source.industry,
        "source_pdf_path": pdf_source.source_pdf_path,
        "status": status,
        "parse_method": PARSE_METHOD,
        "mineru_backend": backend,
        "mineru_device": device,
        "total_pages": "",
        "total_text_chars": text_char_count,
        "markdown_path": markdown_path,
        "metadata": json.dumps(metadata or {}, ensure_ascii=False),
        "error": error,
    }


def rebuild_record_from_cache(
    pdf_source: PdfSource,
    backend: str,
    device: str,
) -> dict:
    markdown_path = find_markdown_output(pdf_source.doc_id, pdf_source.path.stem)
    if markdown_path is None:
        raise FileNotFoundError(f"未找到已缓存的 Markdown：{pdf_source.doc_id}")

    markdown_text = normalize_text(markdown_path.read_text(encoding="utf-8"))
    metadata = extract_doc_metadata(markdown_text)
    return {
        "doc_id": pdf_source.doc_id,
        "filename": pdf_source.filename,
        "industry": pdf_source.industry,
        "industry_label": pdf_source.industry_label,
        "source_pdf_path": pdf_source.source_pdf_path,
        "parse_method": PARSE_METHOD,
        "markdown_path": str(markdown_path.relative_to(PROJECT_ROOT)),
        "text": markdown_text,
        "text_char_count": len(markdown_text),
        "metadata": metadata,
        "mineru_backend": backend,
        "mineru_device": device,
        "mineru_method": MINERU_METHOD,
    }


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_torch_cuda_status() -> tuple[bool, str, str | None]:
    try:
        import torch
    except ImportError:
        return False, "not_installed", None

    cuda_available = torch.cuda.is_available()
    version = torch.__version__
    cuda_build = torch.version.cuda
    return cuda_available, version, cuda_build


def resolve_runtime_config() -> tuple[str, str, dict[str, str], dict[str, str]]:
    cuda_available, torch_version, torch_cuda_build = get_torch_cuda_status()

    if MINERU_DEVICE == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = MINERU_DEVICE.strip().lower()

    if MINERU_BACKEND == "auto":
        backend = "hybrid-auto-engine" if device == "cuda" and cuda_available else "pipeline"
    else:
        backend = MINERU_BACKEND

    if device.startswith("cuda") and not cuda_available:
        print(
            "\n[警告] 你配置了 GPU，但当前 PyTorch 不可用 CUDA。\n"
            f"  torch 版本: {torch_version}\n"
            f"  torch.version.cuda: {torch_cuda_build}\n"
            "  常见原因: 安装的是 CPU 版 PyTorch（例如 2.12.0+cpu）。\n"
            "  修复示例:\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  将自动回退到 CPU + pipeline 后端。\n"
        )
        device = "cpu"
        if backend.startswith("hybrid-"):
            backend = "pipeline"

    mineru_env = os.environ.copy()
    mineru_env["MINERU_DEVICE_MODE"] = device

    runtime_info = {
        "requested_device": MINERU_DEVICE,
        "resolved_device": device,
        "requested_backend": MINERU_BACKEND,
        "resolved_backend": backend,
        "torch_version": torch_version,
        "torch_cuda_available": str(cuda_available),
        "torch_cuda_build": torch_cuda_build or "",
    }
    return device, backend, mineru_env, runtime_info


def extract_doc_metadata(markdown_text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}

    stock_match = re.search(r"[\(（](\d{6})[\)）]", markdown_text)
    if stock_match:
        metadata["stock_code"] = stock_match.group(1)

    date_match = re.search(
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        markdown_text,
    )
    if date_match:
        year, month, day = date_match.groups()
        metadata["report_date"] = f"{year}-{int(month):02d}-{int(day):02d}"

    rating_match = re.search(r"(买入|增持|持有|减持|卖出)(?:[/／](维持|首次))?", markdown_text)
    if rating_match:
        metadata["rating"] = rating_match.group(0)

    company_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]+)\(\d{6}\)", markdown_text)
    if company_match:
        metadata["company_name"] = company_match.group(1)

    return metadata


def find_mineru_cli() -> str:
    cli_path = shutil.which("mineru")
    if cli_path:
        return cli_path

    python_dir = Path(sys.executable).resolve().parent
    for candidate in (
        python_dir / "Scripts" / "mineru.exe",
        python_dir / "Scripts" / "mineru",
        python_dir / "mineru.exe",
        python_dir / "mineru",
    ):
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "未找到 mineru 命令。请先安装：pip install -r requirements.txt"
    )


def run_mineru_on_pdf(
    pdf_path: Path,
    output_dir: Path,
    backend: str,
    mineru_env: dict[str, str],
) -> Path:
    cli_path = find_mineru_cli()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        cli_path,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-b",
        backend,
        "-m",
        MINERU_METHOD,
        "-l",
        MINERU_LANG,
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=mineru_env,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"MinerU 解析失败：{stderr}")

    markdown_files = sorted(output_dir.rglob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(
            f"MinerU 未生成 Markdown 文件，请检查输出目录：{output_dir}"
        )

    preferred = [
        candidate
        for candidate in markdown_files
        if candidate.stem == pdf_path.stem or pdf_path.stem in candidate.stem
    ]
    return preferred[0] if preferred else markdown_files[0]


def parse_single_pdf(
    pdf_source: PdfSource,
    backend: str,
    device: str,
    mineru_env: dict[str, str],
) -> dict:
    pdf_path = pdf_source.path
    doc_output_dir = MINERU_OUTPUT_DIR / pdf_path.stem
    if doc_output_dir.exists():
        shutil.rmtree(doc_output_dir)

    markdown_path = run_mineru_on_pdf(pdf_path, doc_output_dir, backend, mineru_env)
    markdown_text = normalize_text(markdown_path.read_text(encoding="utf-8"))
    metadata = extract_doc_metadata(markdown_text)

    return {
        "doc_id": pdf_source.doc_id,
        "filename": pdf_source.filename,
        "industry": pdf_source.industry,
        "industry_label": pdf_source.industry_label,
        "source_pdf_path": pdf_source.source_pdf_path,
        "parse_method": PARSE_METHOD,
        "markdown_path": str(markdown_path.relative_to(PROJECT_ROOT)),
        "text": markdown_text,
        "text_char_count": len(markdown_text),
        "metadata": metadata,
        "mineru_backend": backend,
        "mineru_device": device,
        "mineru_method": MINERU_METHOD,
    }


def parse_all_pdfs() -> None:
    device, backend, mineru_env, runtime_info = resolve_runtime_config()

    MINERU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOCUMENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    pdf_sources = discover_pdf_files(INPUT_PDF_DIR)
    if not pdf_sources:
        raise FileNotFoundError(
            f"没有在以下目录（含子文件夹）发现 PDF 文件：\n{INPUT_PDF_DIR}"
        )

    write_doc_manifest(pdf_sources)

    jsonl_doc_ids = load_parsed_doc_ids_from_jsonl()
    summary_by_doc_id = load_summary_by_doc_id()

    skipped_count = 0
    success_count = 0
    failed_count = 0

    print("=" * 70)
    print("开始解析金融研报 PDF（方案 B：MinerU）")
    print(f"输入根目录：{INPUT_PDF_DIR}")
    print(f"发现 PDF 数量：{len(pdf_sources)}")
    industries = sorted({source.industry_label or source.industry or "未分类" for source in pdf_sources})
    print(f"行业分布：{', '.join(industries)}")
    print(f"断点续跑：{'开启' if RESUME_SKIP_PARSED else '关闭'}")
    if RESUME_SKIP_PARSED:
        already_parsed = sum(1 for source in pdf_sources if is_pdf_parse_complete(source.doc_id))
        print(f"已解析可跳过：{already_parsed} 份")
    print(f"请求配置：device={MINERU_DEVICE}, backend={MINERU_BACKEND}")
    print(f"实际运行：device={device}, backend={backend}")
    print(f"MINERU_DEVICE_MODE={mineru_env.get('MINERU_DEVICE_MODE')}")
    print(
        "PyTorch："
        f"version={runtime_info['torch_version']}, "
        f"cuda_available={runtime_info['torch_cuda_available']}, "
        f"cuda_build={runtime_info['torch_cuda_build'] or 'None'}"
    )
    print(f"原始 MinerU 输出：{MINERU_OUTPUT_DIR}")
    print("=" * 70)

    for pdf_source in tqdm(pdf_sources, desc="MinerU 解析 PDF"):
        if RESUME_SKIP_PARSED and is_pdf_parse_complete(pdf_source.doc_id):
            skipped_count += 1
            tqdm.write(f"[跳过] {pdf_source.filename}（已存在 content_list_v2）")

            try:
                cached = rebuild_record_from_cache(pdf_source, backend, device)
                if pdf_source.doc_id not in jsonl_doc_ids:
                    append_document_record(cached)
                    jsonl_doc_ids.add(pdf_source.doc_id)
                summary_by_doc_id[pdf_source.doc_id] = build_summary_row(
                    pdf_source,
                    status="skipped",
                    backend=backend,
                    device=device,
                    text_char_count=cached["text_char_count"],
                    markdown_path=cached["markdown_path"],
                    metadata=cached["metadata"],
                )
            except Exception as error:
                tqdm.write(f"[警告] 跳过 {pdf_source.filename}，读取缓存失败：{error}")
                summary_by_doc_id[pdf_source.doc_id] = build_summary_row(
                    pdf_source,
                    status="skipped",
                    backend=backend,
                    device=device,
                    error=str(error),
                )
            save_summary_csv(summary_by_doc_id)
            continue

        try:
            record = parse_single_pdf(pdf_source, backend, device, mineru_env)
            append_document_record(record)
            jsonl_doc_ids.add(pdf_source.doc_id)
            success_count += 1

            summary_by_doc_id[pdf_source.doc_id] = build_summary_row(
                pdf_source,
                status="success",
                backend=backend,
                device=device,
                text_char_count=record["text_char_count"],
                markdown_path=record["markdown_path"],
                metadata=record["metadata"],
            )
            save_summary_csv(summary_by_doc_id)
        except Exception as error:
            failed_count += 1
            tqdm.write(f"[失败] {pdf_source.filename}：{error}")
            summary_by_doc_id[pdf_source.doc_id] = build_summary_row(
                pdf_source,
                status="failed",
                backend=backend,
                device=device,
                error=str(error),
            )
            save_summary_csv(summary_by_doc_id)

    print(f"文档清单：{DOC_MANIFEST_JSONL}")

    usable_count = skipped_count + success_count
    print("\n" + "=" * 70)
    print("MinerU PDF 解析完成")
    print(f"本次新解析成功：{success_count}")
    print(f"跳过（已解析）：{skipped_count}")
    print(f"本次失败：{failed_count}")
    print(f"可用于分块：{usable_count} / {len(pdf_sources)}")
    print(f"文档级结果：{OUTPUT_DOCUMENTS_JSONL}")
    print(f"统计结果：{OUTPUT_SUMMARY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        parse_all_pdfs()
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
