"""
方案 B：使用 MinerU 解析 PDF，输出 Markdown + documents.jsonl。

依赖安装：
    pip install -U "mineru[core]"

首次运行会自动下载模型（体积较大，需预留磁盘空间）。
无 GPU 时使用 pipeline 后端 + CPU。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

INPUT_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
MINERU_OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed" / "mineru"
OUTPUT_DOCUMENTS_JSONL = PROJECT_ROOT / "data" / "parsed" / "documents.jsonl"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "data" / "parsed" / "parse_summary.csv"

PARSE_METHOD = "mineru"
MINERU_BACKEND = "pipeline"
MINERU_METHOD = "auto"
MINERU_LANG = "ch"
MINERU_DEVICE = "cpu"


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    raise FileNotFoundError(
        "未找到 mineru 命令。请先安装：pip install -U \"mineru[core]\""
    )


def run_mineru_on_pdf(pdf_path: Path, output_dir: Path) -> Path:
    cli_path = find_mineru_cli()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        cli_path,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-b",
        MINERU_BACKEND,
        "-m",
        MINERU_METHOD,
        "-l",
        MINERU_LANG,
        "-d",
        MINERU_DEVICE,
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
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


def parse_single_pdf(pdf_path: Path) -> dict:
    doc_output_dir = MINERU_OUTPUT_DIR / pdf_path.stem
    if doc_output_dir.exists():
        shutil.rmtree(doc_output_dir)

    markdown_path = run_mineru_on_pdf(pdf_path, doc_output_dir)
    markdown_text = normalize_text(markdown_path.read_text(encoding="utf-8"))
    metadata = extract_doc_metadata(markdown_text)

    return {
        "doc_id": pdf_path.stem,
        "filename": pdf_path.name,
        "parse_method": PARSE_METHOD,
        "markdown_path": str(markdown_path.relative_to(PROJECT_ROOT)),
        "text": markdown_text,
        "text_char_count": len(markdown_text),
        "metadata": metadata,
        "mineru_backend": MINERU_BACKEND,
        "mineru_method": MINERU_METHOD,
    }


def parse_all_pdfs() -> None:
    MINERU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DOCUMENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"没有在以下目录发现 PDF 文件：\n{INPUT_PDF_DIR}"
        )

    document_records = []
    summary_records = []

    print("=" * 70)
    print("开始解析金融研报 PDF（方案 B：MinerU）")
    print(f"输入文件夹：{INPUT_PDF_DIR}")
    print(f"发现 PDF 数量：{len(pdf_files)}")
    print(f"MinerU 后端：{MINERU_BACKEND} / method={MINERU_METHOD} / device={MINERU_DEVICE}")
    print(f"原始 MinerU 输出：{MINERU_OUTPUT_DIR}")
    print("=" * 70)

    for pdf_path in tqdm(pdf_files, desc="MinerU 解析 PDF"):
        try:
            record = parse_single_pdf(pdf_path)
            document_records.append(record)

            summary_records.append(
                {
                    "filename": pdf_path.name,
                    "status": "success",
                    "parse_method": PARSE_METHOD,
                    "total_pages": "",
                    "total_text_chars": record["text_char_count"],
                    "markdown_path": record["markdown_path"],
                    "metadata": json.dumps(record["metadata"], ensure_ascii=False),
                    "error": "",
                }
            )
        except Exception as error:
            summary_records.append(
                {
                    "filename": pdf_path.name,
                    "status": "failed",
                    "parse_method": PARSE_METHOD,
                    "total_pages": "",
                    "total_text_chars": 0,
                    "markdown_path": "",
                    "metadata": "",
                    "error": str(error),
                }
            )

    with open(OUTPUT_DOCUMENTS_JSONL, "w", encoding="utf-8") as output_file:
        for record in document_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_dataframe = pd.DataFrame(summary_records)
    summary_dataframe.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    success_count = sum(record["status"] == "success" for record in summary_records)
    failed_count = len(summary_records) - success_count

    print("\n" + "=" * 70)
    print("MinerU PDF 解析完成")
    print(f"成功：{success_count}，失败：{failed_count}")
    print(f"文档级结果：{OUTPUT_DOCUMENTS_JSONL}")
    print(f"统计结果：{OUTPUT_SUMMARY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        parse_all_pdfs()
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
