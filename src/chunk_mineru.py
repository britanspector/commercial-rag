"""
将 MinerU 解析结果切分为适合 RAG 检索的文本块（chunks）。

输入：data/parsed/mineru/<doc_id>/.../auto/*_content_list.json
输出：data/parsed/chunks.jsonl

分块策略（Paper 式）：
- 以 MinerU content_list 中的标题（text_level 1/2）作为章节边界
- 同一章节内的正文、表格合并为一个 chunk
- 过长章节按字符数二次切分
- 过滤免责声明等低价值章节
- 保留 page_idx 映射为 PDF 页码（从 1 开始）
"""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path

import pandas as pd
from tqdm import tqdm


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

MINERU_PARSED_DIR = PROJECT_ROOT / "data" / "parsed" / "mineru"
OUTPUT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "data" / "parsed" / "chunk_summary.csv"

CHUNK_METHOD = "mineru_section"
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 80
OVERLAP_CHARS = 100

SKIP_SECTION_KEYWORDS = [
    "免责声明",
    "投资评级说明",
    "太平洋证券股份有限公司",
]


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unescape(text)
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_table_to_text(html: str) -> str:
    if not html:
        return ""

    text = unescape(html)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = normalize_text(text)
    return text


def should_skip_section(title: str) -> bool:
    if not title:
        return False

    return any(keyword in title for keyword in SKIP_SECTION_KEYWORDS)


def extract_doc_metadata(full_text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}

    stock_match = re.search(r"[\(（](\d{6})[\)）]", full_text)
    if stock_match:
        metadata["stock_code"] = stock_match.group(1)

    date_match = re.search(
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        full_text,
    )
    if date_match:
        year, month, day = date_match.groups()
        metadata["report_date"] = f"{year}-{int(month):02d}-{int(day):02d}"

    rating_match = re.search(r"(买入|增持|持有|减持|卖出)(?:[/／](维持|首次))?", full_text)
    if rating_match:
        metadata["rating"] = rating_match.group(0)

    company_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]+)\(\d{6}\)", full_text)
    if company_match:
        metadata["company_name"] = company_match.group(1)

    return metadata


def item_to_text(item: dict) -> str:
    item_type = item.get("type", "")

    if item_type == "text":
        return normalize_text(item.get("text", ""))

    if item_type == "table":
        table_text = html_table_to_text(item.get("table_body", ""))
        caption = normalize_text(" ".join(item.get("table_caption", [])))
        if caption and table_text:
            return f"{caption}\n{table_text}"
        return table_text

    if item_type == "chart":
        chart_text = normalize_text(item.get("content", ""))
        if chart_text:
            return chart_text
        return ""

    return normalize_text(item.get("text", ""))


def page_number_from_item(item: dict) -> int | None:
    if "page_idx" not in item:
        return None
    return int(item["page_idx"]) + 1


def find_content_list_files() -> list[Path]:
    files = [
        path
        for path in sorted(MINERU_PARSED_DIR.rglob("*_content_list.json"))
        if not path.name.endswith("_content_list_v2.json")
    ]
    return files


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return [text[:max_chars]]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)

        if paragraph_len > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0

            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end])
                if end >= len(paragraph):
                    break
                start = max(end - overlap, start + 1)
            continue

        projected_len = current_len + paragraph_len + (2 if current_parts else 0)
        if projected_len > max_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]
            current_len = paragraph_len
        else:
            current_parts.append(paragraph)
            current_len = projected_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


class SectionBuffer:
    def __init__(self, title: str) -> None:
        self.title = title
        self.parts: list[str] = []
        self.pages: set[int] = set()

    def add_item(self, text: str, page_number: int | None) -> None:
        if text:
            self.parts.append(text)
        if page_number is not None:
            self.pages.add(page_number)

    @property
    def text(self) -> str:
        return normalize_text("\n\n".join(self.parts))

    def is_empty(self) -> bool:
        return not self.text


def build_sections(content_items: list[dict]) -> list[SectionBuffer]:
    sections: list[SectionBuffer] = []
    current_title = "文档摘要"
    current_section = SectionBuffer(current_title)

    for item in content_items:
        if item.get("type") == "text" and item.get("text_level") in {1, 2}:
            heading = normalize_text(item.get("text", ""))
            if heading:
                if not current_section.is_empty():
                    sections.append(current_section)

                current_title = heading
                current_section = SectionBuffer(current_title)
                page_number = page_number_from_item(item)
                if page_number is not None:
                    current_section.pages.add(page_number)
                continue

        text = item_to_text(item)
        page_number = page_number_from_item(item)
        current_section.add_item(text, page_number)

    if not current_section.is_empty():
        sections.append(current_section)

    return sections


def chunk_single_document(content_list_path: Path) -> list[dict]:
    content_items = json.loads(content_list_path.read_text(encoding="utf-8"))
    doc_id = content_list_path.name.replace("_content_list.json", "")
    filename = f"{doc_id}.pdf"

    full_text = "\n".join(item_to_text(item) for item in content_items)
    doc_metadata = extract_doc_metadata(full_text)

    chunk_records: list[dict] = []
    chunk_index = 0

    for section in build_sections(content_items):
        if should_skip_section(section.title):
            continue

        section_text = section.text
        if len(section_text) < MIN_CHUNK_CHARS:
            continue

        text_parts = split_long_text(section_text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
        page_start = min(section.pages) if section.pages else None
        page_end = max(section.pages) if section.pages else None

        for part_index, part_text in enumerate(text_parts, start=1):
            if len(part_text) < MIN_CHUNK_CHARS:
                continue

            chunk_index += 1
            chunk_records.append(
                {
                    "chunk_id": f"{doc_id}_{chunk_index:04d}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_method": CHUNK_METHOD,
                    "section_title": section.title,
                    "section_part_index": part_index,
                    "section_part_count": len(text_parts),
                    "text": part_text,
                    "char_count": len(part_text),
                    "page_start": page_start,
                    "page_end": page_end,
                    "pdf_page_numbers": sorted(section.pages),
                    "metadata": doc_metadata,
                }
            )

    return chunk_records


def chunk_all_documents() -> None:
    content_list_files = find_content_list_files()

    if not content_list_files:
        raise FileNotFoundError(
            f"未找到 MinerU content_list 文件，请先运行 src/parse_pdf_mineru.py\n"
            f"搜索目录：{MINERU_PARSED_DIR}"
        )

    OUTPUT_CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    summary_records: list[dict] = []

    print("=" * 70)
    print("开始将 MinerU 解析结果切分为 RAG chunks")
    print(f"输入目录：{MINERU_PARSED_DIR}")
    print(f"发现文档数量：{len(content_list_files)}")
    print(f"分块方法：{CHUNK_METHOD}")
    print(f"单块最大字符数：{MAX_CHUNK_CHARS}")
    print("=" * 70)

    for content_list_path in tqdm(content_list_files, desc="正在分块"):
        try:
            chunk_records = chunk_single_document(content_list_path)
            all_chunks.extend(chunk_records)

            page_values = [
                page
                for record in chunk_records
                for page in record.get("pdf_page_numbers", [])
            ]

            summary_records.append(
                {
                    "doc_id": content_list_path.name.replace("_content_list.json", ""),
                    "status": "success",
                    "chunk_count": len(chunk_records),
                    "total_chars": sum(record["char_count"] for record in chunk_records),
                    "avg_chars_per_chunk": round(
                        sum(record["char_count"] for record in chunk_records)
                        / len(chunk_records),
                        2,
                    ) if chunk_records else 0,
                    "page_min": min(page_values) if page_values else "",
                    "page_max": max(page_values) if page_values else "",
                    "error": "",
                }
            )
        except Exception as error:
            summary_records.append(
                {
                    "doc_id": content_list_path.name.replace("_content_list.json", ""),
                    "status": "failed",
                    "chunk_count": 0,
                    "total_chars": 0,
                    "avg_chars_per_chunk": 0,
                    "page_min": "",
                    "page_max": "",
                    "error": str(error),
                }
            )

    with open(OUTPUT_CHUNKS_JSONL, "w", encoding="utf-8") as output_file:
        for record in all_chunks:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_dataframe = pd.DataFrame(summary_records)
    summary_dataframe.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    success_count = sum(record["status"] == "success" for record in summary_records)
    failed_count = len(summary_records) - success_count

    print("\n" + "=" * 70)
    print("分块完成")
    print(f"成功文档数：{success_count}")
    print(f"失败文档数：{failed_count}")
    print(f"总 chunk 数：{len(all_chunks)}")
    print(f"结果文件：{OUTPUT_CHUNKS_JSONL}")
    print(f"统计文件：{OUTPUT_SUMMARY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    chunk_all_documents()
