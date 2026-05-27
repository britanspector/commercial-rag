import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
from tqdm import tqdm

from pdf_paths import INPUT_PDF_DIR, discover_pdf_files


# ============================================================
# 1. 项目路径配置
# ============================================================

# 获取当前文件所在目录 (src/)
CURRENT_DIR = Path(__file__).parent
# 向上取父目录，得到项目根目录
PROJECT_ROOT = CURRENT_DIR.parent

# 定义各个目录
OUTPUT_PAGE_JSONL = PROJECT_ROOT / "data" / "parsed" / "pages.jsonl"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "data" / "parsed" / "parse_summary.csv"

# 是否提取表格
# 金融研报通常包含大量表格，因此第一版建议设置为 True
EXTRACT_TABLES = True

# 当某一页提取到的文字少于该长度时，标记为需要人工检查
MIN_TEXT_CHAR_COUNT = 30


# ============================================================
# 2. 文本清洗函数
# ============================================================

def normalize_text(text: str) -> str:
    """
    对 PDF 中提取出的文本进行基础清洗。

    主要处理：
    1. 全角空格和特殊空格；
    2. 多余的连续空格；
    3. 过多的空行。
    """
    if not text:
        return ""

    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 合并连续空格，但保留换行
    text = re.sub(r"[ \t]+", " ", text)

    # 将过多空行压缩为两个换行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def table_to_text(table: list) -> str:
    """
    将 pdfplumber 提取出的二维表格转换为普通文本。

    例如：
    [
        ["年份", "收入", "增长率"],
        ["2024", "100亿元", "12.3%"]
    ]

    会被转换为：

    年份 | 收入 | 增长率
    2024 | 100亿元 | 12.3%
    """
    if not table:
        return ""

    text_rows = []

    for row in table:
        if not row:
            continue

        cleaned_cells = []

        for cell in row:
            if cell is None:
                cleaned_cells.append("")
            else:
                cleaned_cells.append(normalize_text(str(cell)))

        if any(cleaned_cells):
            text_rows.append(" | ".join(cleaned_cells))

    return "\n".join(text_rows)


# ============================================================
# 3. 单份 PDF 解析函数
# ============================================================

def parse_single_pdf(pdf_path: Path) -> list[dict]:
    """
    按页解析单份 PDF。

    每一页保存以下信息：
    - doc_id：文档编号，暂时使用文件名；
    - filename：PDF 文件名；
    - pdf_page_number：PDF 文件中的实际页序号；
    - text：正文文本；
    - tables：表格文本；
    - needs_manual_check：该页是否可能存在解析异常。
    """
    page_records = []

    pymupdf_document = fitz.open(pdf_path)
    pdfplumber_document = None

    if EXTRACT_TABLES:
        pdfplumber_document = pdfplumber.open(pdf_path)

    try:
        total_pages = len(pymupdf_document)

        for page_index in range(total_pages):
            pdf_page_number = page_index + 1

            # -------------------------------
            # 3.1 使用 PyMuPDF 提取正文
            # -------------------------------
            page = pymupdf_document[page_index]
            raw_text = page.get_text("text", sort=True)
            cleaned_text = normalize_text(raw_text)

            # -------------------------------
            # 3.2 使用 pdfplumber 提取表格
            # -------------------------------
            extracted_table_texts = []
            table_extraction_error = ""

            if EXTRACT_TABLES and pdfplumber_document is not None:
                try:
                    tables = pdfplumber_document.pages[page_index].extract_tables()

                    for table in tables:
                        table_text = table_to_text(table)

                        if table_text:
                            extracted_table_texts.append(table_text)

                except Exception as error:
                    table_extraction_error = str(error)

            # -------------------------------
            # 3.3 判断是否需要人工检查
            # -------------------------------
            needs_manual_check = len(cleaned_text) < MIN_TEXT_CHAR_COUNT

            page_record = {
                "doc_id": pdf_path.stem,
                "filename": pdf_path.name,
                "pdf_page_number": pdf_page_number,
                "text": cleaned_text,
                "text_char_count": len(cleaned_text),
                "tables": extracted_table_texts,
                "table_count": len(extracted_table_texts),
                "table_extraction_error": table_extraction_error,
                "needs_manual_check": needs_manual_check,
            }

            page_records.append(page_record)

    finally:
        pymupdf_document.close()

        if pdfplumber_document is not None:
            pdfplumber_document.close()

    return page_records


# ============================================================
# 4. 批量解析全部 PDF
# ============================================================

def parse_all_pdfs() -> None:
    """
    解析 raw_pdfs 文件夹中的全部 PDF，并生成：
    1. pages.jsonl：逐页详细文本结果；
    2. parse_summary.csv：每份 PDF 的整体解析统计。
    """
    OUTPUT_PAGE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    pdf_sources = discover_pdf_files(INPUT_PDF_DIR)

    if not pdf_sources:
        raise FileNotFoundError(
            f"没有在以下目录（含子文件夹）发现 PDF 文件：\n{INPUT_PDF_DIR}\n"
            f"请先将 PDF 上传到该目录后再运行脚本。"
        )

    all_page_records = []
    summary_records = []

    print("=" * 70)
    print("开始解析金融研报 PDF")
    print(f"输入根目录：{INPUT_PDF_DIR}")
    print(f"发现 PDF 数量：{len(pdf_sources)}")
    print(f"是否提取表格：{EXTRACT_TABLES}")
    print("=" * 70)

    for pdf_source in tqdm(pdf_sources, desc="正在解析 PDF"):
        pdf_path = pdf_source.path
        try:
            page_records = parse_single_pdf(pdf_path)

            all_page_records.extend(page_records)

            total_pages = len(page_records)
            pages_needing_manual_check = sum(
                record["needs_manual_check"] for record in page_records
            )
            pages_with_tables = sum(
                record["table_count"] > 0 for record in page_records
            )
            total_text_chars = sum(
                record["text_char_count"] for record in page_records
            )
            table_count = sum(
                record["table_count"] for record in page_records
            )

            average_chars_per_page = (
                round(total_text_chars / total_pages, 2)
                if total_pages > 0
                else 0
            )

            summary_records.append(
                {
                    "filename": pdf_path.name,
                    "status": "success",
                    "total_pages": total_pages,
                    "total_text_chars": total_text_chars,
                    "average_chars_per_page": average_chars_per_page,
                    "pages_with_tables": pages_with_tables,
                    "total_table_count": table_count,
                    "pages_needing_manual_check": pages_needing_manual_check,
                    "error": "",
                }
            )

        except Exception as error:
            summary_records.append(
                {
                    "filename": pdf_path.name,
                    "status": "failed",
                    "total_pages": 0,
                    "total_text_chars": 0,
                    "average_chars_per_page": 0,
                    "pages_with_tables": 0,
                    "total_table_count": 0,
                    "pages_needing_manual_check": 0,
                    "error": str(error),
                }
            )

    # -------------------------------
    # 4.1 保存逐页详细解析结果
    # -------------------------------
    with open(OUTPUT_PAGE_JSONL, "w", encoding="utf-8") as output_file:
        for record in all_page_records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # -------------------------------
    # 4.2 保存文档级统计结果
    # -------------------------------
    summary_dataframe = pd.DataFrame(summary_records)
    summary_dataframe.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # -------------------------------
    # 4.3 打印结果说明
    # -------------------------------
    successful_pdf_count = sum(
        record["status"] == "success" for record in summary_records
    )
    failed_pdf_count = sum(
        record["status"] == "failed" for record in summary_records
    )

    print("\n" + "=" * 70)
    print("PDF 解析完成")
    print(f"成功解析 PDF 数量：{successful_pdf_count}")
    print(f"解析失败 PDF 数量：{failed_pdf_count}")
    print(f"成功解析页面数量：{len(all_page_records)}")
    print(f"逐页结果保存位置：{OUTPUT_PAGE_JSONL}")
    print(f"统计结果保存位置：{OUTPUT_SUMMARY_CSV}")
    print("=" * 70)


# ============================================================
# 5. 程序入口
# ============================================================

if __name__ == "__main__":
    parse_all_pdfs()