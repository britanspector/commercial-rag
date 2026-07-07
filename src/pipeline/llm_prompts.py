"""
LLM 答案生成 Prompt：按 query_type 组织检索上下文与用户指令。
"""

from __future__ import annotations

import hashlib
import os

from generation_config import GenerationConfig, resolve_generation_config
from rag_types import Citation
from reranker import hit_passage_text

_SYSTEM_PROMPT = """你是中文金融研报问答助手。请仅依据用户提供的编号资料 [1]...[n] 作答。

规则：
1. 禁止编造资料中不存在的数据、评级或结论。
2. 数字、评级、报告期（如 H1、全年、2026E）必须与原文一致；找不到则按「未找到」模板说明。
3. 正文关键事实后标注引用序号，如 [1]、[2]。
4. 不要输出【参考文献】或参考文献列表（由系统追加）。
5. 用简洁中文回答，避免大段复制表格行。
6. 不要做单位换算或推导（如 百万元→亿元）除非资料中明确给出或同时给出换算依据并引用。
7. 若资料未覆盖问题所问要点，必须友好说明「未找到」并简要概括检索到的相关内容类型（章节/主题），不要只写「未找到」三个字。"""

PROMPT_VERSION = os.environ.get("GEN_PROMPT_VERSION", "").strip() or hashlib.sha256(
    _SYSTEM_PROMPT.encode("utf-8")
).hexdigest()[:12]

_TYPE_HINTS = {
    "factual": (
        "请直接回答所问指标、评级或风险点；注意问题中的报告期（H1/全年/E 等），"
        "优先使用对应期间的数值。"
    ),
    "comparative": (
        "这是对比题：请分别给出各主体的关键数值，最后一行必须以「结论：」开头，"
        "明确说明谁更高/更低或无法比较。"
    ),
    "summary": "请用 3–5 句话归纳要点，避免堆砌表格数据。",
}

_OUTPUT_TEMPLATES = {
    "factual": """输出格式（严格遵循）：
【直接回答】1–3 句话给出问题所问指标/评级/风险结论，关键数字后标 [n]。
若资料未覆盖所问要点，写：
「资料中未找到关于「{问题要点}」的明确信息；检索到的主要内容为：{章节或主题简述} [n]。」""",
    "comparative": """输出格式（严格遵循）：
【{主体A}】关键指标与数值 [n]
【{主体B}】关键指标与数值 [n]
结论：明确说明谁更高/更低，或「资料不足以比较」并说明原因。""",
    "summary": """输出格式（严格遵循）：
1. 要点一 [n]
2. 要点二 [n]
3. 要点三 [n]
（可选 4–5 条）最后一句总结。""",
}

_NOT_FOUND_HINT = (
    "若无法从资料回答问题，请说明：① 问题所问的具体槽位（如年份、指标名）；"
    "② 资料实际提供了什么（如仅有评级说明而无风险因素）；③ 不要编造。"
)


def _format_context_block(index: int, hit: dict, citation: Citation, max_chars: int) -> str:
    passage = hit_passage_text(hit)[:max_chars]
    company = citation.company_name or str(hit.get("company_name", "")).strip() or "未知公司"
    section = citation.section_title or str(hit.get("section_title", "")).strip() or "未知章节"
    page = citation.page_label()
    return (
        f"[{index}] 公司：{company} | 章节：{section} | 页码：{page}\n"
        f"{passage}"
    )


def build_generation_prompt(
    query: str,
    hits: list[dict],
    citations: list[Citation],
    *,
    query_type: str = "factual",
    compare_entities: list[str] | None = None,
    cfg: GenerationConfig | None = None,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    cfg = cfg or resolve_generation_config()
    blocks: list[str] = []
    for citation, hit in zip(citations, hits[: len(citations)]):
        blocks.append(
            _format_context_block(citation.index, hit, citation, cfg.max_context_chars)
        )

    type_hint = _TYPE_HINTS.get(query_type, _TYPE_HINTS["factual"])
    output_tpl = _OUTPUT_TEMPLATES.get(query_type, _OUTPUT_TEMPLATES["factual"])

    entity_line = ""
    if compare_entities:
        entity_line = f"涉及主体：{'、'.join(compare_entities)}\n"
        if query_type == "comparative" and len(compare_entities) >= 2:
            output_tpl = output_tpl.format(
                主体A=compare_entities[0],
                主体B=compare_entities[1],
            )

    user_prompt = (
        f"问题：{query}\n"
        f"题型：{query_type}\n"
        f"{entity_line}"
        f"作答要求：{type_hint}\n"
        f"{output_tpl}\n"
        f"{_NOT_FOUND_HINT}\n\n"
        f"资料：\n"
        + "\n\n".join(blocks)
    )
    return _SYSTEM_PROMPT, user_prompt
