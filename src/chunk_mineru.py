"""
将 MinerU 解析结果切分为适合 RAG 检索的文本块（chunks）。

输入：data/parsed/mineru/<doc_id>/.../auto/*_content_list_v2.json
输出：data/parsed/chunks.jsonl

分块策略（v3）：
- 正文：段落合并 + 句子边界切分，embedding 不超过 512 tokens
- 表格：按行组拆分，每块重复元数据/表头，embedding_text 含自然语言描述
- 噪声：免责声明/分析师联系方式等标记 is_retrievable=false
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from html import unescape
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pdf_paths import build_doc_display_name, get_doc_manifest


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

MINERU_PARSED_DIR = PROJECT_ROOT / "data" / "parsed" / "mineru"
OUTPUT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "data" / "parsed" / "chunk_summary.csv"

CHUNK_METHOD = "mineru_paragraph_v3"
TOKENIZER_MODEL = "BAAI/bge-large-zh-v1.5"
UNITS_JOIN_SEPARATOR = "\n\n"
EMBED_ABSOLUTE_MAX_TOKENS = 512

# 正文 chunk
TEXT_TARGET_MIN_TOKENS = 120
TEXT_TARGET_MAX_TOKENS = 480
TEXT_HARD_MAX_TOKENS = 512
TEXT_OVERLAP_TOKENS = 20

# 表格 chunk（为重复表头/元数据预留空间）
TABLE_TARGET_MAX_TOKENS = 440
TABLE_HARD_MAX_TOKENS = 440
TEXT_MIN_TOKENS = 40

NOISE_SECTION_KEYWORDS = [
    "免责声明",
    "投资评级说明",
    "投资评级标准",
    "资质声明",
    "特别声明",
    "分析师声明",
    "分析师介绍",
    "适当性管理办法",
    "太平洋证券股份有限公司",
    "公司概况",
    "中邮证券研究所",
]

REPORT_TITLE_SKIP = {
    "投资要点",
    "投资建议",
    "风险提示",
    "核心风险提示",
    "投资逻辑",
    "盈利预测",
    "盈利预测、估值和评级",
    "市场表现",
    "投资评级与估值",
    "关键假设",
    "投资逻辑要点",
}

GENERIC_REPORT_TITLES = {
    "证券研究报告",
    "公司研究",
    "公司点评报告",
    "公司深度报告",
    "公司深度研究",
    "文档摘要",
    "个股表现",
    "市场表现",
    "公司基本情况",
    "投资评级",
    "上次评级",
    "研究所",
    "电子组",
}

DISCLAIMER_TITLE_KEYWORDS = [
    "投资评级说明",
    "分析师声明",
    "免责声明",
    "适当性管理办法",
    "公司概况",
]

SKIP_CONTENT_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
    "page_footnote",
}

NOISE_TEXT_PATTERNS = [
    re.compile(r"E-MAIL\s*[:：]|Email\s*[:：]|邮\s*箱\s*[:：]", re.I),
    re.compile(r"@[a-zA-Z0-9._-]+\.(?:com|cn)(?:\.cn)?"),
    re.compile(r"分析师登记编号|执业编号|SAC\s*[:：]|S\d{11}"),
    re.compile(r"^证券分析师\s*[:：]|^分析师\s*[:：]", re.M),
    re.compile(r"^联系人\s*[:：]", re.M),
    re.compile(r"^<<.+>>"),
    re.compile(r"请务必阅读正文之后的免责条款"),
    re.compile(r"本报告仅供.*客户使用"),
    re.compile(r"太平洋证券股份有限公司.*版权所有"),
    re.compile(r"投资评级说明|投资评级标准|适当性管理办法"),
    re.compile(r"分析师声明|免责声明"),
    re.compile(r"^\[Table_", re.M),
    re.compile(r"^上次评级\s*$", re.M),
]

BROKER_RULES = [
    (re.compile(r"太平洋|PACIFIC\s*SECURITIES", re.I), "太平洋证券"),
    (re.compile(r"華源|华源|HUAYUAN", re.I), "华源证券"),
    (re.compile(r"国金|SINOLINK", re.I), "国金证券"),
    (re.compile(r"中邮|CHINA\s*POST", re.I), "中邮证券"),
    (re.compile(r"东海"), "东海证券"),
    (re.compile(r"信达|CINDA", re.I), "信达证券"),
]

STOCK_CODE_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]+)\s*[（(]\s*(\d{6})(?:\.(?:SH|SZ))?\s*[）)]"
)
STOCK_CODE_LOOSE_PATTERN = re.compile(
    r"\[Table_\w*(?P<name>[\u4e00-\u9fff]+)\s*\(\s*(?P<code>\d{6})"
)
REPORT_DATE_HEADER_PATTERN = re.compile(r"发布时间[：:]\s*(\d{4}-\d{2}-\d{2})")
REPORT_DATE_CN_PATTERN = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")

RATING_PATTERNS = [
    re.compile(r"投资评级[：:\s]*(?:\*+)?\s*(买入|增持|持有|减持|卖出)(?:[/／](维持|首次))?"),
    re.compile(r"(?:首次覆盖[,，])?给予[“\"](买入|增持|持有|减持|卖出)[”\"]"),
    re.compile(r"(买入|增持|持有|卖出)\s*[（(](?:首次|维持|首次覆盖|首次评级)"),
    re.compile(r"(买入|增持|持有|卖出)\s*[|｜]\s*(?:维持|首次)"),
    re.compile(r"(?:股票)?投资评级[：:\s]*(买入|增持|持有|减持|卖出)"),
]

UNIT_PATTERNS = [
    re.compile(r"[（(]([^）)]*(?:百万元|万元|亿元|%/pct|倍|元/股)[^）)]*)[）)]"),
    re.compile(r"单位\s*[:：]\s*(\S+)"),
]

YEAR_COL_PATTERN = re.compile(r"^\d{4}[AEFH]?$|^\d{4}\s*[AEFH]$")
FINANCIAL_TABLE_PATTERN = re.compile(
    r"利润表|资产负债表|现金流量表|财务预测|盈利预测|估值|三大报表|附录|损益表|财务状况表|财务指标|比率分析|分产品|分业务"
)
TABLE_SELF_TITLE_PATTERN = re.compile(
    r"图\d+|表\d+|图表\d+|比率分析|分产品|分业务|盈利预测|财务指标|PE估值|三大报表"
)
COMPLEX_TABLE_PATTERN = re.compile(r"可比公司|PE估值")
ROW_METRIC_UNITS = {"百万元", "亿元", "万元", "%", "％", "倍", "元", "元/股"}
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；!?])\s*")

FORBIDDEN_NARRATIVE_PATTERNS = [
    re.compile(r"百万元\)为[\d,.+-]+元(?!/股)"),
    re.compile(r"亿元\)为[\d,.+-]+元(?!/股)"),
    re.compile(r"(?:PE|P/E|市盈率)为[\d,.+-]+亿元"),
    re.compile(r"(?:EPS|BVPS|每股收益|每股盈利)为[\d,.+-]+(?:亿元|倍)(?!/)"),
    re.compile(r"BVPS为[\d,.+-]+倍"),
    re.compile(r"(?:ROE|ROIC|ROA|毛利率|净利率)为[\d,.+-]+倍"),
    re.compile(r"YoY为%|(?:营收|占比)为%|为百万元[；;]?$"),
]

METADATA_FRONT_PAGE_LIMIT = 3

SPAN_FIELD_BY_TYPE = {
    "paragraph": "paragraph_content",
    "title": "title_content",
    "list": "list_items",
    "index": "index_items",
    "equation_interline": "math_content",
}


@dataclass
class DocContext:
    company_name: str = ""
    stock_code: str = ""
    broker: str = ""
    report_title: str = ""
    report_date: str = ""
    rating: str = ""


@dataclass
class ContentUnit:
    text: str
    page_number: int
    section_title: str
    unit_type: str
    table_caption: str = ""
    table_header: str = ""
    table_rows: tuple[str, ...] = field(default_factory=tuple)
    table_footnote: str = ""
    table_id: str = ""
    table_seq: int = 0
    is_noise: bool = False


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def is_table_unit(unit: ContentUnit) -> bool:
    return unit.unit_type == "table"


def group_has_table(units: list[ContentUnit]) -> bool:
    return any(is_table_unit(unit) for unit in units)


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
    return clean_mineru_markers(text.strip())


def clean_mineru_markers(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"\[Table_StockAndRan(?P<name>[^\]\(]+)\(\s*(?P<code>\d{6})",
        r"\g<name>(\g<code>",
        text,
    )
    text = re.sub(r"\[Table_Title\]", "", text)
    text = re.sub(r"\[Table_Author\]\s*", "", text)
    text = re.sub(r"\[T[^\]]*?ummary\]\s*", "", text)
    text = re.sub(r"\[Table_[^\]]+\]", "", text)
    text = re.sub(r"^\[Table_\S+", "", text)
    text = re.sub(r"\[财[^\]]*比率[^\]]*\]", "财务报表和主要财务比率", text)
    text = re.sub(r"^[lL]\s+", "", text)
    return text.strip()


def html_table_to_text(html: str) -> str:
    if not html:
        return ""

    text = unescape(html)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return normalize_text(text)


def table_body_rows(body: str) -> tuple[str, tuple[str, ...]]:
    rows = tuple(row.strip() for row in body.split("\n") if row.strip())
    if not rows:
        return "", ()
    if len(rows) == 1:
        return rows[0], ()
    return rows[0], rows[1:]


def is_noise_section(title: str) -> bool:
    if not title:
        return False
    return any(keyword in title for keyword in NOISE_SECTION_KEYWORDS)


def is_rating_standard_table(text: str, section_title: str, caption: str, header: str) -> bool:
    combined = f"{section_title}\n{caption}\n{header}\n{text[:400]}"
    return bool(
        re.search(r"投资评级标准|投资评级说明|报告中投资建议的评级标准|适当性管理办法", combined)
    )


def clean_numeric_value(value: str) -> str:
    val = value.strip()
    val = re.sub(r"(?<=\d)\s+(?=\d)", "", val)
    val = re.sub(r"(?<=[.,])\s+(?=\d)", "", val)
    val = re.sub(r"(?<=\d)\s+(?=[.%％])", "", val)
    return val


def clean_numeric_in_text(text: str) -> str:
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        return clean_numeric_value(match.group(0))

    return re.sub(r"[\d][\d\s.,%％]*[\d.%％]|[\d]+(?:\s+[\d.%％]+)+", _replace, text)


def strip_analyst_blocks(text: str) -> str:
    if not text:
        return text

    lines: list[str] = []
    for line in text.split("\n"):
        if re.search(
            r"证券分析师|分析师[：:]|执业编号|SAC\s*[:：]|Email\s*[:：]|邮\s*箱\s*[:：]|"
            r"分析师登记编号|联系人\s*[:：]",
            line,
            re.I,
        ):
            continue
        if re.search(r"@[a-zA-Z0-9._-]+\.(?:com|cn)(?:\.cn)?", line, re.I):
            continue
        lines.append(line)

    return normalize_text("\n".join(lines))


def sanitize_embedding_text(text: str) -> str:
    if not text:
        return text

    text = clean_numeric_in_text(text)
    text = re.sub(
        r"((?:PE|P/E|市盈率|PB|P/B|市净率)[^。\n]{0,40}?[\d、，,\s]+)亿元",
        r"\1倍",
        text,
        flags=re.I,
    )
    text = re.sub(r"([\d.]+)\s*倍\s*亿元", r"\1倍", text)
    text = re.sub(r"([\d.]+)\s*元\s*/\s*股\s*倍", r"\1元/股", text)
    return text.strip()


def has_rating_conclusion(text: str) -> bool:
    return bool(
        re.search(
            r"给予[“\"](?:买入|增持|持有|卖出)[”\"]评级|投资评级[：:\s]*(?:\*+)?\s*(?:买入|增持|持有|卖出)|"
            r"首次覆盖.*?(?:买入|增持|持有|卖出)",
            text,
        )
    )


def prepare_text_embedding(text: str, is_noise: bool) -> tuple[str, bool, str]:
    cleaned = sanitize_embedding_text(strip_analyst_blocks(text))
    if is_noise and has_rating_conclusion(text):
        if cleaned.strip():
            return cleaned, True, "text"
        return text, False, "noise"
    if is_noise:
        return text, False, "noise"
    return cleaned, True, "text"


def validate_embedding_units(text: str) -> bool:
    return not any(pattern.search(text) for pattern in FORBIDDEN_NARRATIVE_PATTERNS)


def is_noise_text(text: str, section_title: str) -> bool:
    if "风险提示" in section_title or section_title.strip() in {"核心风险提示", "l 风险提示"}:
        if "风险提示" in text or "风险" in text:
            return False

    if is_noise_section(section_title):
        return True

    if section_title.strip() in {"研究所", "电子组", "分析师介绍"}:
        return True

    if section_title.strip() in {"投资评级", "上次评级"}:
        return not has_rating_conclusion(text) and "评级" not in text

    if "风险提示" in text and "投资评级" not in text:
        return False

    if any(pattern.search(text) for pattern in NOISE_TEXT_PATTERNS):
        if has_rating_conclusion(text):
            return False
        return True

    # 投资评级结论段落保留
    if has_rating_conclusion(text):
        return False

    return False


def extract_unit_from_text(text: str) -> str:
    for pattern in UNIT_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_text(match.group(1))
    return ""


def spans_to_text(spans: list | None) -> str:
    if not spans:
        return ""

    parts: list[str] = []
    for span in spans:
        if not isinstance(span, dict):
            continue

        span_type = span.get("type", "")
        if span_type == "text":
            parts.append(str(span.get("content", "")))
            continue

        if span_type == "hyperlink":
            content = span.get("content", "")
            if content:
                parts.append(str(content))
            elif span.get("children"):
                parts.append(spans_to_text(span["children"]))
            continue

        if "content" in span:
            parts.append(str(span["content"]))

    return normalize_text("".join(parts))


def join_text_parts(parts: list[str]) -> str:
    cleaned = [part for part in parts if part]
    return normalize_text("\n".join(cleaned))


def join_segments(segments: list[str], separator: str = "\n") -> str:
    return separator.join(segment for segment in segments if segment)


def v2_table_parts(content: dict) -> tuple[str, str, str, str]:
    caption = spans_to_text(content.get("table_caption"))
    body = html_table_to_text(content.get("html", ""))
    footnote = spans_to_text(content.get("table_footnote"))
    text = join_text_parts([caption, body, footnote])
    return caption, body, footnote, text


def v2_chart_to_text(content: dict) -> str:
    caption = spans_to_text(content.get("chart_caption"))
    body = normalize_text(str(content.get("content", "")))
    footnote = spans_to_text(content.get("chart_footnote"))
    return join_text_parts([caption, body, footnote])


def v2_list_to_text(content: dict, field_name: str) -> str:
    items = content.get(field_name, [])
    if not items:
        return ""

    if isinstance(items[0], dict):
        return normalize_text("\n".join(spans_to_text([item]) for item in items))

    return normalize_text("\n".join(str(item) for item in items))


def v2_equation_to_text(content: dict) -> str:
    math_content = content.get("math_content", [])
    if isinstance(math_content, list):
        return spans_to_text(math_content)
    return normalize_text(str(math_content or ""))


def v2_title_to_text(content: dict) -> str:
    return spans_to_text(content.get("title_content"))


def find_content_list_files() -> list[Path]:
    return sorted(MINERU_PARSED_DIR.rglob("*_content_list_v2.json"))


def merged_token_count(texts: list[str], separator: str = UNITS_JOIN_SEPARATOR) -> int:
    return count_tokens(separator.join(text for text in texts if text))


def units_to_text(units: list[ContentUnit]) -> str:
    return normalize_text(UNITS_JOIN_SEPARATOR.join(unit.text for unit in units if unit.text))


def split_paragraph_by_sentences(paragraph: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(paragraph) if part.strip()]
    return parts if parts else [paragraph]


def split_single_segment_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        low = start + 1
        high = text_len
        best = low

        while low <= high:
            mid = (low + high) // 2
            segment = text[start:mid]
            if count_tokens(segment) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best <= start:
            best = min(start + 1, text_len)

        chunks.append(text[start:best])
        if best >= text_len:
            break

        overlap_end = best
        overlap_start = best
        while overlap_start > start and count_tokens(text[overlap_start:overlap_end]) < overlap_tokens:
            overlap_start -= 1

        start = max(start + 1, overlap_start)

    return chunks


def split_long_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    if not paragraphs:
        return split_single_segment_by_tokens(text, max_tokens, overlap_tokens)

    chunks: list[str] = []
    current_parts: list[str] = []

    for paragraph in paragraphs:
        if count_tokens(paragraph) > max_tokens:
            if current_parts:
                chunks.append(join_segments(current_parts))
                current_parts = []

            sentences = split_paragraph_by_sentences(paragraph)
            sentence_buffer: list[str] = []

            for sentence in sentences:
                if count_tokens(sentence) > max_tokens:
                    if sentence_buffer:
                        chunks.append(join_segments(sentence_buffer))
                        sentence_buffer = []
                    chunks.extend(
                        split_single_segment_by_tokens(sentence, max_tokens, overlap_tokens)
                    )
                    continue

                candidate = sentence_buffer + [sentence]
                if sentence_buffer and count_tokens(join_segments(candidate)) > max_tokens:
                    chunks.append(join_segments(sentence_buffer))
                    sentence_buffer = [sentence]
                else:
                    sentence_buffer = candidate

            if sentence_buffer:
                if chunks and overlap_tokens > 0:
                    overlap = sentence_buffer[0]
                    prev = chunks[-1]
                    if count_tokens(prev + overlap) <= max_tokens:
                        chunks[-1] = join_segments([prev, overlap])
                        sentence_buffer = sentence_buffer[1:]
                if sentence_buffer:
                    chunks.append(join_segments(sentence_buffer))
            continue

        candidate_parts = current_parts + [paragraph]
        if current_parts and count_tokens(join_segments(candidate_parts)) > max_tokens:
            chunks.append(join_segments(current_parts))
            current_parts = [paragraph]
        else:
            current_parts = candidate_parts

    if current_parts:
        chunks.append(join_segments(current_parts))

    return chunks


def is_generic_report_title(title: str) -> bool:
    if not title:
        return True
    cleaned = clean_mineru_markers(title).strip()
    if cleaned in GENERIC_REPORT_TITLES:
        return True
    return any(keyword in cleaned for keyword in DISCLAIMER_TITLE_KEYWORDS)


def detect_broker(text: str) -> str:
    for pattern, broker_name in BROKER_RULES:
        if pattern.search(text):
            return broker_name
    return ""


def extract_company_from_text(text: str) -> tuple[str, str]:
    match = STOCK_CODE_PATTERN.search(text)
    if match:
        return match.group(1).strip(), match.group(2)

    loose_match = STOCK_CODE_LOOSE_PATTERN.search(text)
    if loose_match:
        return loose_match.group("name").strip(), loose_match.group("code")

    return "", ""


def extract_rating_from_text(text: str) -> str:
    if re.search(r"股东减持|减持计划|减持比例|减持完毕|减持股份", text):
        rating_text = text
        for pattern in RATING_PATTERNS:
            match = pattern.search(rating_text)
            if match and match.group(1) != "减持":
                rating = match.group(1)
                suffix = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                return f"{rating}/{suffix}" if suffix else rating
        return ""

    for pattern in RATING_PATTERNS:
        match = pattern.search(text)
        if match:
            rating = match.group(1)
            suffix = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            return f"{rating}/{suffix}" if suffix else rating

    header_match = re.search(
        r"(买入|增持|持有|卖出)\s*[（(](?:首次|维持|首次评级|首次覆盖)[^）)]*[）)]",
        text,
    )
    if header_match:
        return header_match.group(1)

    return ""


def extract_report_date_from_text(text: str, *, allow_legal_context: bool = False) -> str:
    if not allow_legal_context and any(
        keyword in text for keyword in ("适当性管理办法", "投资评级说明", "免责声明", "分析师声明")
    ):
        return ""

    header_match = REPORT_DATE_HEADER_PATTERN.search(text)
    if header_match:
        return header_match.group(1)

    date_match = REPORT_DATE_CN_PATTERN.search(text)
    if date_match:
        year, month, day = date_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return ""


def merge_split_titles(titles: list[str]) -> list[str]:
    if not titles:
        return []

    merged: list[str] = []
    buffer = ""

    for title in titles:
        cleaned = clean_mineru_markers(title).strip()
        if not cleaned:
            continue
        if is_generic_report_title(cleaned) and not buffer:
            continue

        if buffer:
            if cleaned in REPORT_TITLE_SKIP or is_generic_report_title(cleaned):
                merged.append(buffer)
                buffer = ""
                continue
            merged.append(f"{buffer}{cleaned}")
            buffer = ""
            continue

        if extract_company_from_text(cleaned)[0] and len(cleaned) <= 30:
            continue

        if cleaned.endswith(("，", "、", "先", "的", "与", "及")) or (
            len(cleaned) <= 18 and not cleaned.endswith(("）", ")"))
        ):
            buffer = cleaned
            continue

        merged.append(cleaned)

    if buffer:
        merged.append(buffer)

    return merged


def resolve_table_section_title(current_title: str, caption: str, body: str) -> str:
    caption = clean_mineru_markers(caption).strip()
    current = clean_mineru_markers(current_title).strip()
    first_body_line = body.split("\n", 1)[0].strip() if body else ""

    if caption and FINANCIAL_TABLE_PATTERN.search(caption):
        return caption

    if caption and TABLE_SELF_TITLE_PATTERN.search(caption):
        return caption

    if first_body_line in {"利润表", "资产负债表", "现金流量表"}:
        return first_body_line

    if first_body_line and FINANCIAL_TABLE_PATTERN.search(first_body_line):
        return first_body_line

    if first_body_line and TABLE_SELF_TITLE_PATTERN.search(first_body_line):
        return first_body_line

    if re.search(r"风险提示|核心风险|免责声明|分析师|投资评级说明|比率分析", current):
        if caption:
            return caption
        if first_body_line:
            return first_body_line

    return current or caption


def extract_doc_context(pages: list[list[dict]]) -> DocContext:
    context = DocContext()
    front_pages = pages[:METADATA_FRONT_PAGE_LIMIT]

    header_texts: list[str] = []
    page_one_titles: list[str] = []
    front_paragraphs: list[str] = []

    for page_idx, page_items in enumerate(front_pages):
        page_number = page_idx + 1
        for item in page_items:
            item_type = item.get("type", "")
            content = item.get("content", {})

            if item_type == "page_header":
                header_text = spans_to_text(content.get("page_header_content"))
                if header_text:
                    header_texts.append(header_text)
                    if not context.broker:
                        context.broker = detect_broker(header_text)

            if item_type == "title" and page_number == 1:
                heading = clean_mineru_markers(v2_title_to_text(content))
                if heading:
                    page_one_titles.append(heading)

            if item_type == "paragraph":
                text = spans_to_text(content.get("paragraph_content"))
                if not text:
                    continue
                if page_number <= METADATA_FRONT_PAGE_LIMIT:
                    front_paragraphs.append(text)
                if not context.broker:
                    context.broker = detect_broker(text)

    for text in header_texts + front_paragraphs + page_one_titles:
        company_name, stock_code = extract_company_from_text(text)
        if company_name and not context.company_name:
            context.company_name = company_name
        if stock_code and not context.stock_code:
            context.stock_code = stock_code

    for text in header_texts:
        if not context.report_date:
            context.report_date = extract_report_date_from_text(text)

    for text in front_paragraphs:
        if not context.report_date:
            context.report_date = extract_report_date_from_text(text)

    for text in header_texts + front_paragraphs[:12]:
        if not context.rating:
            context.rating = extract_rating_from_text(text)

    for title in page_one_titles:
        if not context.rating:
            context.rating = extract_rating_from_text(title)

    title_candidates = merge_split_titles(page_one_titles)
    title_candidates = [
        title
        for title in title_candidates
        if not is_generic_report_title(title)
        and title not in REPORT_TITLE_SKIP
        and not (extract_company_from_text(title)[0] and len(title) <= 30)
    ]
    if title_candidates:
        context.report_title = max(title_candidates, key=len)

    if not context.company_name and context.report_title:
        company_name, stock_code = extract_company_from_text(context.report_title)
        if company_name:
            context.company_name = company_name
        if stock_code and not context.stock_code:
            context.stock_code = stock_code

    if not context.report_title and context.company_name:
        context.report_title = context.company_name

    return context


def parse_table_columns(header: str) -> list[str]:
    cells = [cell.strip() for cell in header.split("|") if cell.strip()]
    if len(cells) <= 1:
        return []
    first = cells[0]
    if YEAR_COL_PATTERN.match(first.replace(" ", "")):
        return cells
    return cells[1:]


def value_has_unit(value: str) -> bool:
    return bool(
        re.search(
            r"[%％]|百万元|亿元|万元|元/股|元$|倍$|倍\s|万股|亿股|百万股",
            value.strip(),
        )
    )


def is_numeric_metric_value(value: str) -> bool:
    val = value.strip()
    if not val or val in {"-", "—", "--"}:
        return False
    return bool(re.search(r"\d", val))


def infer_indicator_unit(indicator: str, table_unit: str, value: str) -> str:
    name = indicator.strip()
    val = value.strip()

    if not is_numeric_metric_value(val):
        return ""

    if value_has_unit(val):
        return ""

    embedded_match = re.search(
        r"[（(]([^）)]*(?:百万元|万元|亿元|元/股|元|倍|%|％)[^）)]*)[）)]",
        name,
    )
    if embedded_match:
        embedded = embedded_match.group(1)
        if "百万元" in embedded:
            return "百万元"
        if "亿元" in embedded:
            return "亿元"
        if "万元" in embedded:
            return "万元"
        if "%" in embedded or "％" in embedded:
            return "%" if not re.search(r"[%％]", val) else ""
        if "倍" in embedded:
            return "倍"
        if "元/股" in embedded:
            return "元/股"
        if embedded.endswith("元") or embedded == "元":
            return "元"
        return embedded

    if re.search(
        r"(%|％|增长率|增速|yoy|YoY|同比|毛利率|净利率|ROE|ROIC|ROA|"
        r"资产负债率|负债率|占比|周转率|费用率|税.*?率|股息率|利用率|occupancy|"
        r"增.*?%|pct)",
        name,
        re.I,
    ):
        return "%" if not re.search(r"[%％]", val) else ""

    if re.search(r"^(BVPS|EPS|每股|摊薄)", name, re.I):
        return "元/股" if "股" in name or name.upper() in {"EPS", "BVPS"} else "元"

    if re.search(r"^(PE|P/E|PB|P/B|PEG|EV/)", name, re.I):
        return "倍"

    if re.search(r"(PE|P/E|PB|P/B|PEG|市净率|市盈率|EV/EBITDA)", name, re.I):
        return "倍"

    if re.search(r"(收盘价|目标价|市价|股价|最高.?最低|52周)", name):
        return "元"

    if re.search(r"(总股本|流通股本|股本|百万股|万股|亿股)", name):
        if "亿股" in name:
            return "亿股"
        if "万股" in name:
            return "万股"
        if "百万股" in name:
            return "百万股"
        return ""

    if re.search(
        r"(第一大股东|股东名称|公司名称|说明|类型|备注|资料来源)",
        name,
    ):
        return ""

    if re.search(
        r"(营业收入|营业总收入|主营收入|营收|收入|净利润|归母|净利|毛利|成本|费用|"
        r"资产|负债|现金流|现金|存货|市值|资本|税收|税金|薪酬|折旧|摊销|"
        r"EBIT|EBITDA|利润|补贴|投资|支出|金额|营亏|所得税|营业利润|归母净利润|"
        r"货币资金|应收|应付|预收|预付|固定资产|无形资产)",
        name,
    ):
        return table_unit or ""

    return table_unit or ""


def normalize_indicator_label(indicator: str) -> str:
    """财务指标行名 → 检索友好的自然语言标签（P2 表行语义化）。"""
    label = indicator.strip()
    if re.search(r"每股盈利|每股收益", label, re.I) and "EPS" not in label.upper():
        return f"{label}(EPS)"
    if re.search(r"归母净利润|归属于.*净利润", label):
        return "归母净利润" if "归母" not in label else label
    if label in {"营业总收入", "营业收入"}:
        return "营业收入"
    return label


def format_narrative_sentence(column_name: str, indicator: str, value: str, unit_suffix: str) -> str:
    label = normalize_indicator_label(indicator)
    value = clean_numeric_value(value)
    if column_name and not YEAR_COL_PATTERN.match(column_name.replace(" ", "")):
        sentence = f"{column_name} {label}为{value}"
    else:
        sentence = f"{column_name} {label}为{value}" if column_name else f"{label}为{value}"

    if unit_suffix and not value_has_unit(value):
        sentence = f"{sentence}{unit_suffix}"

    return sentence


def is_valid_narrative_sentence(sentence: str, value: str) -> bool:
    if not is_numeric_metric_value(value):
        return False
    if re.search(r"为[%百万元亿元倍]+[；;]?$", sentence):
        return False
    if re.search(r"(?:YoY|营收|占比|毛利率)为%[；;]?$", sentence):
        return False
    return validate_embedding_units(sentence)


def validate_narrative_text(narrative: str) -> bool:
    return validate_embedding_units(narrative)


def parse_year_columns(header: str) -> list[str]:
    cells = [cell.strip() for cell in header.split("|") if cell.strip()]
    if not cells:
        return []
    if cells[0] in {"单位", "指标", "项目"}:
        cells = cells[1:]
    return [cell for cell in cells if YEAR_COL_PATTERN.match(cell.replace(" ", ""))]


def is_metric_unit_cell(cell: str) -> bool:
    return clean_numeric_value(cell.strip()) in ROW_METRIC_UNITS


def is_unit_column_table(header: str) -> bool:
    cells = [cell.strip() for cell in header.split("|") if cell.strip()]
    return bool(cells) and cells[0] == "单位"


def has_corrupted_table_cells(rows: list[str]) -> bool:
    for row in rows:
        for cell in row.split("|"):
            cell = cell.strip()
            if not cell:
                continue
            if re.search(r"[\u4e00-\u9fff]{2,}.*[\d,.]+|[\d,.]+.*[\u4e00-\u9fff]{2,}", cell):
                if not re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9（）().\-\s]+", cell):
                    return True
    return False


def is_comparable_valuation_table(caption: str, header: str) -> bool:
    return bool(re.search(r"可比公司|PE估值", f"{caption} {header}"))


def is_complex_table(caption: str, header: str, rows: list[str], section_title: str = "") -> bool:
    if is_comparable_valuation_table(caption, header):
        return False
    if is_unit_column_table(header):
        return False

    label = f"{caption} {section_title} {header}"
    if COMPLEX_TABLE_PATTERN.search(label):
        return True
    if re.search(r"股票代码.*公司简称|EPS\s*[（(].*PE\s*[（(]", header):
        return True
    if rows and re.search(r"股票代码", rows[0]):
        return True
    if has_corrupted_table_cells(rows):
        return True
    for row in rows[:2]:
        year_count = sum(
            1
            for cell in row.split("|")
            if YEAR_COL_PATTERN.match(cell.strip().replace(" ", ""))
        )
        if year_count >= 4 and ("EPS" in header or "PE" in header):
            return True
    return False


def rows_to_comparable_valuation_narrative(header: str, rows: list[str]) -> str:
    year_row: list[str] | None = None
    header_cells = [cell.strip() for cell in header.split("|") if cell.strip()]
    if sum(1 for cell in header_cells if YEAR_COL_PATTERN.match(cell.replace(" ", ""))) >= 4:
        year_row = header_cells

    sentences: list[str] = []

    for row in rows:
        cells = [cell.strip() for cell in row.split("|") if cell.strip()]
        if not cells:
            continue

        year_cells = [cell for cell in cells if YEAR_COL_PATTERN.match(cell.replace(" ", ""))]
        if len(year_cells) >= 4 and not any(re.search(r"\.(SH|SZ)", cell, re.I) for cell in cells):
            if year_row is None:
                year_row = cells
            continue

        if cells[0] in {"股票代码"} or cells[0].startswith("资料来源"):
            continue

        name = cells[1] if len(cells) > 1 and re.search(r"\.(SH|SZ)", cells[0], re.I) else cells[0]
        if name in {"可比公司均值", "算术平均"}:
            if len(cells) >= 7 and year_row and len(year_row) >= 6:
                eps_years, pe_years = year_row[:3], year_row[3:6]
                eps_vals, pe_vals = cells[-6:-3], cells[-3:]
                for year, value in zip(eps_years, eps_vals):
                    value = clean_numeric_value(value)
                    if is_numeric_metric_value(value):
                        sentences.append(f"可比公司均值 {year} EPS为{value}元/股")
                for year, value in zip(pe_years, pe_vals):
                    value = clean_numeric_value(value)
                    if is_numeric_metric_value(value):
                        sentences.append(f"可比公司均值 {year} PE为{value}倍")
            continue

        if len(cells) >= 9 and year_row and len(year_row) >= 6:
            eps_years, pe_years = year_row[:3], year_row[3:6]
            eps_vals, pe_vals = cells[3:6], cells[6:9]
            for year, value in zip(eps_years, eps_vals):
                value = clean_numeric_value(value)
                if is_numeric_metric_value(value):
                    sentences.append(f"{name} {year} EPS为{value}元/股")
            for year, value in zip(pe_years, pe_vals):
                value = clean_numeric_value(value)
                if is_numeric_metric_value(value):
                    sentences.append(f"{name} {year} PE为{value}倍")

    return "；".join(sentences)


def rows_to_unit_column_narrative(header: str, rows: list[str], product_context: str = "") -> str:
    columns = parse_year_columns(header)
    if not columns:
        return ""

    sentences: list[str] = []
    current_product = product_context

    for row in rows:
        raw_cells = [cell.strip() for cell in row.split("|")]
        while raw_cells and not raw_cells[0]:
            raw_cells.pop(0)
        while raw_cells and not raw_cells[-1]:
            raw_cells.pop()

        if not raw_cells:
            continue

        indicator = raw_cells[0]
        if indicator in {"|", "合计", "来源：ifind、国金证券研究所"}:
            continue

        if len(raw_cells) == 1 and not YEAR_COL_PATTERN.match(indicator.replace(" ", "")):
            if indicator not in {"营收", "YoY", "占比", "毛利率", "销售毛利率"}:
                current_product = indicator
            continue

        if len(raw_cells) < 3 or not is_metric_unit_cell(raw_cells[1]):
            continue

        row_unit = raw_cells[1].strip()
        values = [clean_numeric_value(cell.strip()) for cell in raw_cells[2:] if cell.strip()]

        for column_name, value in zip(columns, values):
            if not value or value in {"-", "—", "--"}:
                continue
            if not is_numeric_metric_value(value):
                continue

            unit_suffix = ""
            if row_unit in {"%", "％"}:
                unit_suffix = "%"
            elif row_unit in ROW_METRIC_UNITS:
                unit_suffix = row_unit

            label = f"{current_product}{indicator}" if current_product else indicator
            sentence = format_narrative_sentence(column_name, label, value, unit_suffix)
            if is_valid_narrative_sentence(sentence, value):
                sentences.append(sentence)

    return "；".join(sentences)


def table_has_year_columns(header: str, rows: list[str]) -> bool:
    columns = parse_table_columns(header)
    if any(YEAR_COL_PATTERN.match(col.replace(" ", "")) for col in columns):
        return True
    return any(
        YEAR_COL_PATTERN.match(row.split("|")[0].strip().replace(" ", ""))
        for row in rows
        if row.strip()
    )


def rows_to_kv_narrative(rows: list[str], table_unit: str) -> str:
    sentences: list[str] = []

    for row in rows:
        cells = [cell.strip() for cell in row.split("|") if cell.strip()]
        if len(cells) < 2:
            if len(cells) == 1 and re.search(r"[\d.]+\s*$", cells[0]):
                continue
            continue

        indicator = cells[0]
        value = clean_numeric_value(cells[-1])
        if not is_numeric_metric_value(value):
            continue
        unit_suffix = infer_indicator_unit(indicator, table_unit, value)
        sentence = format_narrative_sentence("", indicator, value, unit_suffix)
        if is_valid_narrative_sentence(sentence, value):
            sentences.append(sentence)

    return "；".join(sentences)


def rows_to_narrative(
    header: str,
    rows: list[str],
    unit: str,
    caption: str = "",
    section_title: str = "",
) -> str:
    if is_comparable_valuation_table(caption, header):
        narrative = rows_to_comparable_valuation_narrative(header, rows)
        if narrative and validate_embedding_units(narrative):
            return narrative
        return ""

    if is_unit_column_table(header):
        narrative = rows_to_unit_column_narrative(header, rows, product_context=caption)
        if narrative and validate_embedding_units(narrative):
            return narrative
        return ""

    if is_complex_table(caption, header, rows, section_title):
        return ""

    if not parse_year_columns(header) and not table_has_year_columns(header, rows):
        return ""

    if not table_has_year_columns(header, rows):
        return rows_to_kv_narrative(rows, unit)

    columns = parse_table_columns(header)
    if not columns:
        return ""

    sentences: list[str] = []

    for row in rows:
        cells = [cell.strip() for cell in row.split("|") if cell.strip()]
        if len(cells) < 2:
            continue

        indicator = cells[0]
        if YEAR_COL_PATTERN.match(indicator.replace(" ", "")):
            continue

        values = cells[1:]
        if len(values) >= len(columns):
            values = values[: len(columns)]
        elif len(cells) > len(columns):
            values = cells[1 : 1 + len(columns)]
        else:
            values = cells[1:]

        for column_name, value in zip(columns, values):
            if not value or value in {"-", "—", "--"}:
                continue
            value = clean_numeric_value(value)
            if not is_numeric_metric_value(value):
                continue
            unit_suffix = infer_indicator_unit(indicator, unit, value)
            sentence = format_narrative_sentence(column_name, indicator, value, unit_suffix)
            if is_valid_narrative_sentence(sentence, value):
                sentences.append(sentence)

    return "；".join(sentences)


def trim_embedding_text(embedding_text: str, max_tokens: int) -> str:
    if count_tokens(embedding_text) <= max_tokens:
        return embedding_text

    parts = embedding_text.split("\n\n", 1)
    if len(parts) == 2:
        context_block, body = parts
        context_tokens = count_tokens(context_block)
        body_budget = max(max_tokens - context_tokens - 1, 64)
        trimmed_body = split_long_text_by_tokens(body, body_budget, overlap_tokens=0)[0]
        return join_segments([context_block, trimmed_body], separator="\n\n")

    return split_long_text_by_tokens(embedding_text, max_tokens, overlap_tokens=0)[0]


def build_table_context_block(
    doc_ctx: DocContext,
    section_title: str,
    caption: str,
    unit: str,
    header: str,
    page_number: int,
) -> str:
    company_line = doc_ctx.company_name or doc_ctx.report_title
    if doc_ctx.stock_code:
        company_line = (
            f"{doc_ctx.company_name}({doc_ctx.stock_code})"
            if doc_ctx.company_name
            else f"{doc_ctx.report_title}({doc_ctx.stock_code})" if doc_ctx.report_title else doc_ctx.stock_code
        )

    lines = [
        f"公司：{company_line}" if company_line else "",
        f"券商：{doc_ctx.broker}" if doc_ctx.broker else "",
        f"报告：{doc_ctx.report_title}" if doc_ctx.report_title else "",
        f"章节：{section_title}" if section_title else "",
        f"表格：{caption}" if caption else "",
        f"单位：{unit}" if unit else "",
        f"列头：{header}" if header else "",
        f"页码：第{page_number}页",
    ]
    return join_segments([line for line in lines if line])


def build_table_raw_text(
    caption: str,
    header: str,
    rows: list[str],
    footnote: str = "",
    include_footnote: bool = False,
) -> str:
    parts = []
    if caption:
        parts.append(clean_numeric_in_text(caption))
    if header:
        parts.append(clean_numeric_in_text(header))
    parts.extend(clean_numeric_in_text(row) for row in rows)
    if include_footnote and footnote:
        parts.append(footnote)
    return join_segments(parts)


def build_table_embedding_text(
    doc_ctx: DocContext,
    section_title: str,
    caption: str,
    unit: str,
    header: str,
    rows: list[str],
    page_number: int,
) -> str:
    context_block = build_table_context_block(
        doc_ctx, section_title, caption, unit, header, page_number
    )
    table_raw = build_table_raw_text(caption, header, rows)
    narrative = rows_to_narrative(header, rows, unit, caption=caption, section_title=section_title)
    if narrative and validate_embedding_units(narrative):
        body = narrative
    else:
        body = table_raw
    return join_segments([context_block, body], separator="\n\n")


@dataclass
class TableChunkPart:
    unit: ContentUnit
    table_raw: str
    embedding_text: str
    part_index: int
    part_count: int


def split_table_unit(
    unit: ContentUnit,
    doc_ctx: DocContext,
) -> list[TableChunkPart]:
    caption = unit.table_caption
    header = unit.table_header
    data_rows = list(unit.table_rows)
    footnote = unit.table_footnote
    unit_text = extract_unit_from_text(caption) or extract_unit_from_text(header) or extract_unit_from_text(unit.text)

    if not data_rows:
        table_raw = unit.text
        embedding_text = build_table_embedding_text(
            doc_ctx,
            unit.section_title,
            caption,
            unit_text,
            header,
            [header] if header else [],
            unit.page_number,
        )
        return [TableChunkPart(unit, table_raw, embedding_text, 1, 1)]

    prefix_tokens = count_tokens(
        build_table_context_block(doc_ctx, unit.section_title, caption, unit_text, header, unit.page_number)
    )
    if prefix_tokens >= TABLE_HARD_MAX_TOKENS:
        table_raw = build_table_raw_text(caption, header, data_rows, footnote, include_footnote=True)
        embedding_text = build_table_embedding_text(
            doc_ctx, unit.section_title, caption, unit_text, header, data_rows, unit.page_number
        )
        return [TableChunkPart(unit, table_raw, embedding_text, 1, 1)]

    batch_rows: list[str] = []
    raw_parts: list[tuple[list[str], bool]] = []

    def flush_batch(include_footnote: bool = False) -> None:
        if batch_rows:
            raw_parts.append((batch_rows.copy(), include_footnote))
            batch_rows.clear()

    for row in data_rows:
        candidate_rows = batch_rows + [row]
        candidate_embedding = build_table_embedding_text(
            doc_ctx,
            unit.section_title,
            caption,
            unit_text,
            header,
            candidate_rows,
            unit.page_number,
        )
        if count_tokens(candidate_embedding) <= TABLE_HARD_MAX_TOKENS:
            batch_rows.append(row)
            continue

        if batch_rows:
            flush_batch()
            batch_rows.append(row)
            single_embedding = build_table_embedding_text(
                doc_ctx, unit.section_title, caption, unit_text, header, batch_rows, unit.page_number
            )
            if count_tokens(single_embedding) > TABLE_HARD_MAX_TOKENS:
                flush_batch()
                raw_parts.append(([row], False))
                batch_rows.clear()
            continue

        raw_parts.append(([row], False))

    flush_batch(include_footnote=True)

    if not raw_parts and batch_rows:
        raw_parts.append((batch_rows, True))

    part_count = len(raw_parts)
    parts: list[TableChunkPart] = []
    for index, (rows, include_footnote) in enumerate(raw_parts, start=1):
        table_raw = build_table_raw_text(caption, header, rows, footnote, include_footnote)
        embedding_text = build_table_embedding_text(
            doc_ctx, unit.section_title, caption, unit_text, header, rows, unit.page_number
        )
        embedding_text = trim_embedding_text(embedding_text, TABLE_HARD_MAX_TOKENS)
        parts.append(TableChunkPart(unit, table_raw, embedding_text, index, part_count))

    return parts or [
        TableChunkPart(
            unit,
            build_table_raw_text(caption, header, data_rows, footnote, True),
            build_table_embedding_text(
                doc_ctx, unit.section_title, caption, unit_text, header, data_rows, unit.page_number
            ),
            1,
            1,
        )
    ]


def clone_unit(unit: ContentUnit, text: str) -> ContentUnit:
    return ContentUnit(
        text=text,
        page_number=unit.page_number,
        section_title=unit.section_title,
        unit_type=unit.unit_type,
        table_caption=unit.table_caption,
        table_header=unit.table_header,
        table_rows=unit.table_rows,
        table_footnote=unit.table_footnote,
        table_id=unit.table_id,
        table_seq=unit.table_seq,
        is_noise=unit.is_noise,
    )


def expand_oversized_text_unit(unit: ContentUnit) -> list[ContentUnit]:
    if count_tokens(unit.text) <= TEXT_HARD_MAX_TOKENS:
        return [unit]

    return [
        clone_unit(unit, part)
        for part in split_long_text_by_tokens(unit.text, TEXT_HARD_MAX_TOKENS, TEXT_OVERLAP_TOKENS)
        if count_tokens(part) >= TEXT_MIN_TOKENS or unit.is_noise
    ]


def extract_units_from_v2(pages: list[list[dict]], doc_id: str) -> list[ContentUnit]:
    units: list[ContentUnit] = []
    current_title = "文档摘要"
    table_seq = 0

    for page_idx, page_items in enumerate(pages):
        page_number = page_idx + 1

        for item in page_items:
            item_type = item.get("type", "")
            content = item.get("content", {})

            if item_type == "title":
                heading = clean_mineru_markers(v2_title_to_text(content))
                if heading:
                    current_title = heading
                continue

            if item_type in SKIP_CONTENT_TYPES:
                continue

            text = ""
            table_caption = ""
            table_header = ""
            table_rows: tuple[str, ...] = ()
            table_footnote = ""
            table_id = ""
            effective_section = current_title

            if item_type == "paragraph":
                text = spans_to_text(content.get("paragraph_content"))
            elif item_type == "table":
                table_seq += 1
                table_id = f"{doc_id}_table_{table_seq:03d}"
                table_caption, body, table_footnote, text = v2_table_parts(content)
                table_caption = clean_mineru_markers(table_caption)
                table_header, table_rows = table_body_rows(body)
                if not parse_year_columns(table_header) and table_rows:
                    promoted = parse_year_columns(table_rows[0])
                    if promoted:
                        table_header = table_rows[0]
                        table_rows = table_rows[1:]
                effective_section = resolve_table_section_title(
                    current_title, table_caption, body
                )
            elif item_type == "chart":
                text = v2_chart_to_text(content)
            elif item_type in {"list", "index"}:
                text = v2_list_to_text(content, SPAN_FIELD_BY_TYPE[item_type])
            elif item_type == "equation_interline":
                text = v2_equation_to_text(content)
            elif item_type in {"code", "algorithm"}:
                field_prefix = item_type
                text = join_text_parts([
                    spans_to_text(content.get(f"{field_prefix}_caption")),
                    spans_to_text(content.get(f"{field_prefix}_content")),
                    spans_to_text(content.get(f"{field_prefix}_footnote")),
                ])
            else:
                span_field = SPAN_FIELD_BY_TYPE.get(item_type)
                if span_field:
                    text = spans_to_text(content.get(span_field))

            text = normalize_text(text)
            if not text or len(text) < 2:
                continue

            units.append(
                ContentUnit(
                    text=text,
                    page_number=page_number,
                    section_title=effective_section if item_type == "table" else current_title,
                    unit_type=item_type,
                    table_caption=table_caption if item_type == "table" else "",
                    table_header=table_header if item_type == "table" else "",
                    table_rows=table_rows if item_type == "table" else (),
                    table_footnote=table_footnote if item_type == "table" else "",
                    table_id=table_id if item_type == "table" else "",
                    table_seq=table_seq if item_type == "table" else 0,
                    is_noise=(
                        is_noise_text(text, effective_section if item_type == "table" else current_title)
                        or (
                            item_type == "table"
                            and is_rating_standard_table(
                                text,
                                effective_section,
                                table_caption,
                                table_header,
                            )
                        )
                    ),
                )
            )

    return units


def pack_units(units: list[ContentUnit]) -> list[list[ContentUnit]]:
    if not units:
        return []

    groups: list[list[ContentUnit]] = []
    current: list[ContentUnit] = []

    def flush_current() -> None:
        if current:
            groups.append(current.copy())
            current.clear()

    for unit in units:
        if unit.is_noise:
            flush_current()
            groups.append([unit])
            continue

        if is_table_unit(unit):
            flush_current()
            groups.append([unit])
            continue

        if count_tokens(unit.text) > TEXT_HARD_MAX_TOKENS:
            flush_current()
            groups.extend([[part] for part in expand_oversized_text_unit(unit)])
            continue

        if current and unit.section_title != current[-1].section_title:
            flush_current()

        projected_tokens = merged_token_count([item.text for item in current] + [unit.text])
        if current and projected_tokens > TEXT_HARD_MAX_TOKENS:
            flush_current()

        projected_tokens = merged_token_count([item.text for item in current] + [unit.text])
        if current and projected_tokens > TEXT_TARGET_MAX_TOKENS:
            current_tokens = merged_token_count([item.text for item in current])
            if current_tokens >= TEXT_TARGET_MIN_TOKENS:
                flush_current()
                current.append(unit)
                continue
            if projected_tokens <= TEXT_HARD_MAX_TOKENS:
                current.append(unit)
                continue
            flush_current()
            current.append(unit)
            continue

        current.append(unit)

    flush_current()
    return merge_short_text_groups(groups)


def merge_short_text_groups(groups: list[list[ContentUnit]]) -> list[list[ContentUnit]]:
    if len(groups) <= 1:
        return groups

    merged = [groups[0]]
    for group in groups[1:]:
        if group[0].is_noise or is_table_unit(group[0]):
            merged.append(group)
            continue

        group_tokens = merged_token_count([unit.text for unit in group])
        if group_tokens >= TEXT_TARGET_MIN_TOKENS:
            merged.append(group)
            continue

        prev_group = merged[-1]
        if prev_group[0].is_noise or is_table_unit(prev_group[0]):
            merged.append(group)
            continue

        combined_tokens = merged_token_count(
            [unit.text for unit in prev_group] + [unit.text for unit in group]
        )
        same_section = prev_group[-1].section_title == group[0].section_title
        if combined_tokens <= TEXT_HARD_MAX_TOKENS and same_section:
            prev_group.extend(group)
        else:
            merged.append(group)

    return merged


def split_text_group(group: list[ContentUnit]) -> list[list[ContentUnit]]:
    text = units_to_text(group)
    if count_tokens(text) <= TEXT_HARD_MAX_TOKENS:
        return [group]

    section_title = group[0].section_title
    page_number = group[0].page_number
    unit_type = group[0].unit_type
    is_noise = group[0].is_noise

    split_groups: list[list[ContentUnit]] = []
    for part_text in split_long_text_by_tokens(text, TEXT_HARD_MAX_TOKENS, TEXT_OVERLAP_TOKENS):
        if count_tokens(part_text) < TEXT_MIN_TOKENS and not is_noise:
            continue
        split_groups.append([
            ContentUnit(
                text=part_text,
                page_number=page_number,
                section_title=section_title,
                unit_type=unit_type,
                is_noise=is_noise,
            )
        ])
    return split_groups or [group]


def build_chunk_record(
    *,
    chunk_id: str,
    doc_id: str,
    filename: str,
    doc_ctx: DocContext,
    group: list[ContentUnit],
    text: str,
    embedding_text: str,
    content_type: str,
    is_retrievable: bool,
    display_name: str = "",
    industry: str = "",
    industry_label: str = "",
    source_pdf_path: str = "",
    table_raw: str = "",
    table_id: str = "",
    table_part_index: int = 0,
    table_part_count: int = 0,
) -> dict:
    page_numbers = sorted({unit.page_number for unit in group})
    section_title = group[0].section_title

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "filename": filename,
        "display_name": display_name,
        "industry": industry,
        "industry_label": industry_label,
        "source_pdf_path": source_pdf_path,
        "chunk_method": CHUNK_METHOD,
        "content_type": content_type,
        "is_retrievable": is_retrievable,
        "company_name": doc_ctx.company_name,
        "stock_code": doc_ctx.stock_code,
        "broker": doc_ctx.broker,
        "report_title": doc_ctx.report_title,
        "report_date": doc_ctx.report_date,
        "rating": doc_ctx.rating,
        "section_title": section_title,
        "page_start": page_numbers[0],
        "page_end": page_numbers[-1],
        "pdf_page_numbers": page_numbers,
        "text": text,
        "embedding_text": embedding_text,
        "table_raw": table_raw,
        "table_id": table_id,
        "table_part_index": table_part_index,
        "table_part_count": table_part_count,
        "unit_count": len(group),
        "unit_types": sorted({unit.unit_type for unit in group}),
        "contains_table": content_type == "table",
        "char_count": len(text),
        "embedding_char_count": len(embedding_text),
        "token_count": count_tokens(text),
        "embedding_token_count": count_tokens(embedding_text),
    }


APPENDIX_SECTION_PATTERN = re.compile(r"附录|三大报表|财务报表预测|盈利预测与估值")


def build_rating_headline_record(
    *,
    doc_id: str,
    filename: str,
    doc_ctx: DocContext,
    doc_fields: dict,
) -> dict | None:
    """P2：封面/元数据评级单独成可检索块。"""
    if not doc_ctx.rating:
        return None

    company = doc_ctx.company_name or doc_ctx.report_title or "该公司"
    stock = f"（{doc_ctx.stock_code}）" if doc_ctx.stock_code else ""
    text = f"{company}{stock} 研报投资评级：{doc_ctx.rating}。"
    if doc_ctx.broker:
        text += f" 券商：{doc_ctx.broker}。"
    embedding_text = sanitize_embedding_text(strip_analyst_blocks(text))

    unit = ContentUnit(
        text=text,
        page_number=1,
        section_title="投资评级摘要",
        unit_type="rating_headline",
        is_noise=False,
    )
    return build_chunk_record(
        chunk_id=f"{doc_id}_rating_headline",
        doc_id=doc_id,
        filename=filename,
        doc_ctx=doc_ctx,
        group=[unit],
        text=text,
        embedding_text=embedding_text,
        content_type="rating_headline",
        is_retrievable=True,
        **doc_fields,
    )


def _flush_appendix_buffer(buffer: list[dict]) -> list[dict]:
    if not buffer:
        return []
    if len(buffer) == 1:
        return buffer

    first = buffer[0]
    merged_text = join_segments([record.get("text", "") for record in buffer])
    merged_embedding = join_segments([record.get("embedding_text", "") for record in buffer])
    page_starts = [int(record.get("page_start") or 0) for record in buffer if record.get("page_start")]
    page_ends = [int(record.get("page_end") or 0) for record in buffer if record.get("page_end")]

    merged = dict(first)
    merged["text"] = merged_text
    merged["embedding_text"] = merged_embedding
    merged["page_start"] = min(page_starts) if page_starts else first.get("page_start", 0)
    merged["page_end"] = max(page_ends) if page_ends else first.get("page_end", 0)
    merged["pdf_page_numbers"] = sorted(
        {
            page
            for record in buffer
            for page in (record.get("pdf_page_numbers") or [])
        }
    )
    merged["unit_count"] = sum(int(record.get("unit_count") or 1) for record in buffer)
    merged["char_count"] = len(merged_text)
    merged["embedding_char_count"] = len(merged_embedding)
    merged["token_count"] = count_tokens(merged_text)
    merged["embedding_token_count"] = count_tokens(merged_embedding)
    merged["section_title"] = first.get("section_title", "") or "附录"
    return [merged]


def merge_appendix_chunks(records: list[dict]) -> list[dict]:
    """P2：同一文档内连续附录小节合并，减少碎片化。"""
    if len(records) <= 1:
        return records

    merged: list[dict] = []
    buffer: list[dict] = []
    current_doc = records[0].get("doc_id", "")

    def flush() -> None:
        nonlocal buffer
        if buffer:
            merged.extend(_flush_appendix_buffer(buffer))
            buffer = []

    for record in records:
        doc_id = record.get("doc_id", "")
        section = str(record.get("section_title", ""))
        is_appendix = bool(APPENDIX_SECTION_PATTERN.search(section))

        if doc_id != current_doc:
            flush()
            current_doc = doc_id

        if is_appendix and record.get("is_retrievable", True):
            buffer.append(record)
            continue

        flush()
        merged.append(record)

    flush()
    return merged


def renumber_chunk_ids(records: list[dict], doc_id: str) -> list[dict]:
    """评级摘要块保留固定 id，其余按序重编号。"""
    rating = [record for record in records if record.get("chunk_id", "").endswith("_rating_headline")]
    body = [record for record in records if not record.get("chunk_id", "").endswith("_rating_headline")]
    ordered = rating + body
    body_index = 0
    for record in ordered:
        if record.get("chunk_id", "").endswith("_rating_headline"):
            continue
        body_index += 1
        record["chunk_id"] = f"{doc_id}_{body_index:04d}"
    return ordered


def post_process_chunk_records(records: list[dict], doc_id: str) -> list[dict]:
    records = merge_appendix_chunks(records)
    return renumber_chunk_ids(records, doc_id)


def chunk_single_document(
    content_list_path: Path,
    doc_manifest: dict[str, dict] | None = None,
) -> list[dict]:
    pages = json.loads(content_list_path.read_text(encoding="utf-8"))
    doc_id = content_list_path.name.replace("_content_list_v2.json", "")
    meta = (doc_manifest or {}).get(doc_id, {})
    filename = meta.get("filename", f"{doc_id}.pdf")
    industry = meta.get("industry", "")
    industry_label = meta.get("industry_label", "")
    source_pdf_path = meta.get("source_pdf_path", "")
    doc_ctx = extract_doc_context(pages)
    display_name = build_doc_display_name(
        company_name=doc_ctx.company_name,
        stock_code=doc_ctx.stock_code,
        report_title=doc_ctx.report_title,
        broker=doc_ctx.broker,
        report_date=doc_ctx.report_date,
        rating=doc_ctx.rating,
        industry_label=industry_label,
        filename=filename,
    )
    doc_fields = {
        "display_name": display_name,
        "industry": industry,
        "industry_label": industry_label,
        "source_pdf_path": source_pdf_path,
    }
    units = extract_units_from_v2(pages, doc_id)
    groups = pack_units(units)

    chunk_records: list[dict] = []
    chunk_index = 0

    rating_headline = build_rating_headline_record(
        doc_id=doc_id,
        filename=filename,
        doc_ctx=doc_ctx,
        doc_fields=doc_fields,
    )
    if rating_headline:
        chunk_records.append(rating_headline)

    for group in groups:
        if is_table_unit(group[0]):
            table_parts = split_table_unit(group[0], doc_ctx)
            is_comparable = is_comparable_valuation_table(
                group[0].table_caption,
                group[0].table_header,
            )
            for part in table_parts:
                chunk_index += 1
                if part.unit.is_noise:
                    table_content_type = "noise"
                elif is_comparable:
                    table_content_type = "comparable_table"
                else:
                    table_content_type = "table"
                chunk_records.append(
                    build_chunk_record(
                        chunk_id=f"{doc_id}_{chunk_index:04d}",
                        doc_id=doc_id,
                        filename=filename,
                        doc_ctx=doc_ctx,
                        group=[part.unit],
                        text=part.table_raw,
                        embedding_text=part.embedding_text,
                        content_type=table_content_type,
                        is_retrievable=not part.unit.is_noise,
                        **doc_fields,
                        table_raw=part.table_raw,
                        table_id=part.unit.table_id,
                        table_part_index=part.part_index,
                        table_part_count=part.part_count,
                    )
                )
                if chunk_records[-1]["embedding_token_count"] > EMBED_ABSOLUTE_MAX_TOKENS:
                    trimmed = trim_embedding_text(
                        chunk_records[-1]["embedding_text"],
                        EMBED_ABSOLUTE_MAX_TOKENS,
                    )
                    chunk_records[-1]["embedding_text"] = trimmed
                    chunk_records[-1]["embedding_token_count"] = count_tokens(trimmed)
                    chunk_records[-1]["embedding_char_count"] = len(trimmed)
            continue

        for sub_group in split_text_group(group):
            chunk_index += 1
            text = units_to_text(sub_group)
            raw_noise = sub_group[0].is_noise
            embedding_text, is_retrievable, content_type = prepare_text_embedding(text, raw_noise)
            chunk_records.append(
                build_chunk_record(
                    chunk_id=f"{doc_id}_{chunk_index:04d}",
                    doc_id=doc_id,
                    filename=filename,
                    doc_ctx=doc_ctx,
                    group=sub_group,
                    text=text,
                    embedding_text=embedding_text,
                    content_type=content_type,
                    is_retrievable=is_retrievable,
                    **doc_fields,
                )
            )
            if chunk_records[-1]["embedding_token_count"] > EMBED_ABSOLUTE_MAX_TOKENS:
                trimmed = trim_embedding_text(embedding_text, EMBED_ABSOLUTE_MAX_TOKENS)
                chunk_records[-1]["embedding_text"] = trimmed
                chunk_records[-1]["embedding_token_count"] = count_tokens(trimmed)
                chunk_records[-1]["embedding_char_count"] = len(trimmed)

    return post_process_chunk_records(chunk_records, doc_id)


def print_validation_stats(records: list[dict]) -> None:
    over_512 = sum(
        1 for record in records if record.get("embedding_token_count", 0) > EMBED_ABSOLUTE_MAX_TOKENS
    )

    unit_error_counts = {idx: 0 for idx in range(len(FORBIDDEN_NARRATIVE_PATTERNS))}
    for record in records:
        if not record.get("is_retrievable"):
            continue
        body = record.get("embedding_text", "")
        if record.get("content_type") == "table":
            parts = body.split("\n\n", 1)
            body = parts[1] if len(parts) == 2 else body
        for idx, pattern in enumerate(FORBIDDEN_NARRATIVE_PATTERNS):
            if pattern.search(body):
                unit_error_counts[idx] += 1

    noise_patterns = [
        re.compile(r"@[a-zA-Z0-9._-]+\.(?:com|cn)(?:\.cn)?", re.I),
        re.compile(r"执业编号|分析师声明|投资评级说明|投资评级标准|适当性管理办法"),
    ]
    noisy_retrievable = 0
    for record in records:
        if not record.get("is_retrievable"):
            continue
        text = record.get("embedding_text", "")
        if any(pattern.search(text) for pattern in noise_patterns):
            noisy_retrievable += 1

    print("\n" + "=" * 70)
    print("质量检查")
    print(f"超过 {EMBED_ABSOLUTE_MAX_TOKENS} tokens：{over_512}")
    print(f"百万元)为xx元 类错误：{unit_error_counts.get(0, 0)}")
    print(f"亿元)为xx元 类错误：{unit_error_counts.get(1, 0)}")
    print(f"PE+亿元 类错误：{unit_error_counts.get(2, 0)}")
    print(f"EPS/BVPS+亿元/倍 类错误：{unit_error_counts.get(3, 0)}")
    print(f"BVPS+倍 类错误：{unit_error_counts.get(4, 0)}")
    print(f"ROE/毛利率+倍 类错误：{unit_error_counts.get(5, 0)}")
    print(f"空值/错误单位 类错误：{unit_error_counts.get(6, 0)}")
    print(f"仍可检索但含邮箱/分析师声明/评级标准：{noisy_retrievable}")


def print_sample_chunks(records: list[dict]) -> None:
    comparable = next(
        (
            record
            for record in records
            if record.get("content_type") == "table"
            and record.get("is_retrievable")
            and ("可比公司" in record.get("section_title", "") or "可比公司" in record.get("text", ""))
        ),
        None,
    )
    product_revenue = next(
        (
            record
            for record in records
            if record.get("content_type") == "table"
            and record.get("is_retrievable")
            and ("分产品" in record.get("section_title", "") + record.get("text", ""))
        ),
        None,
    )

    print("\n" + "=" * 70)
    print("复杂表格转写样例")
    for label, sample in [("可比公司估值表", comparable), ("分产品营收预测表", product_revenue)]:
        print(f"\n--- {label} ---")
        if not sample:
            print("未找到样例")
            continue
        print(f"chunk_id={sample['chunk_id']} embed_tokens={sample['embedding_token_count']}")
        print(sample["embedding_text"][:1000])


def chunk_all_documents() -> None:
    content_list_files = find_content_list_files()
    if not content_list_files:
        raise FileNotFoundError(
            f"未找到 MinerU content_list_v2 文件，请先运行 src/parse_pdf_mineru.py\n"
            f"搜索目录：{MINERU_PARSED_DIR}"
        )

    get_tokenizer()
    OUTPUT_CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    summary_records: list[dict] = []

    print("=" * 70)
    print("开始将 MinerU 解析结果切分为 RAG chunks (v3)")
    print(f"输入目录：{MINERU_PARSED_DIR}")
    print(f"发现文档数量：{len(content_list_files)}")
    print(f"正文 embedding 上限：{TEXT_HARD_MAX_TOKENS} tokens")
    print(f"表格 embedding 上限：{TABLE_HARD_MAX_TOKENS} tokens")
    print("=" * 70)

    doc_manifest = get_doc_manifest()
    print(f"文档清单：{len(doc_manifest)} 份")

    for content_list_path in tqdm(content_list_files, desc="正在分块"):
        try:
            chunk_records = chunk_single_document(content_list_path, doc_manifest)
            all_chunks.extend(chunk_records)

            embed_tokens = [record["embedding_token_count"] for record in chunk_records]
            summary_records.append(
                {
                    "doc_id": content_list_path.name.replace("_content_list_v2.json", ""),
                    "status": "success",
                    "chunk_count": len(chunk_records),
                    "text_chunk_count": sum(1 for r in chunk_records if r["content_type"] == "text"),
                    "table_chunk_count": sum(1 for r in chunk_records if r["content_type"] == "table"),
                    "noise_chunk_count": sum(1 for r in chunk_records if r["content_type"] == "noise"),
                    "retrievable_count": sum(1 for r in chunk_records if r["is_retrievable"]),
                    "max_embedding_tokens": max(embed_tokens) if embed_tokens else 0,
                    "over_512_embedding_tokens": sum(
                        1 for value in embed_tokens if value > EMBED_ABSOLUTE_MAX_TOKENS
                    ),
                    "error": "",
                }
            )
        except Exception as error:
            summary_records.append(
                {
                    "doc_id": content_list_path.name.replace("_content_list_v2.json", ""),
                    "status": "failed",
                    "chunk_count": 0,
                    "text_chunk_count": 0,
                    "table_chunk_count": 0,
                    "noise_chunk_count": 0,
                    "retrievable_count": 0,
                    "max_embedding_tokens": 0,
                    "over_512_embedding_tokens": 0,
                    "error": str(error),
                }
            )

    with open(OUTPUT_CHUNKS_JSONL, "w", encoding="utf-8") as output_file:
        for record in all_chunks:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    pd.DataFrame(summary_records).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    embed_tokens = [record["embedding_token_count"] for record in all_chunks]
    text_count = sum(1 for record in all_chunks if record["content_type"] == "text")
    table_count = sum(1 for record in all_chunks if record["content_type"] == "table")
    noise_count = sum(1 for record in all_chunks if record["content_type"] == "noise")
    over_512 = sum(1 for value in embed_tokens if value > EMBED_ABSOLUTE_MAX_TOKENS)

    print("\n" + "=" * 70)
    print("分块完成")
    print(f"总 chunk 数：{len(all_chunks)}")
    print(f"  正文：{text_count}  表格：{table_count}  噪声：{noise_count}")
    print(f"  可检索：{sum(1 for r in all_chunks if r['is_retrievable'])}")
    if embed_tokens:
        print(
            f"embedding token：min={min(embed_tokens)}, "
            f"max={max(embed_tokens)}, "
            f"avg={round(sum(embed_tokens)/len(embed_tokens), 1)}"
        )
    print(f"超过 {EMBED_ABSOLUTE_MAX_TOKENS} tokens：{over_512}")
    print(f"结果文件：{OUTPUT_CHUNKS_JSONL}")
    print(f"统计文件：{OUTPUT_SUMMARY_CSV}")
    print_validation_stats(all_chunks)
    print_sample_chunks(all_chunks)
    print("=" * 70)


if __name__ == "__main__":
    chunk_all_documents()
