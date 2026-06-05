"""
answer_generate：基于检索证据 + Ollama LLM 生成带引用的答案。

引用元数据由 rerank hits 确定性生成；【参考文献】由程序追加。
"""

from __future__ import annotations

import re

from generation_config import describe_generation_config, resolve_generation_config
from pipeline.evidence_select import select_evidence_hits
from pipeline.llm_client import invoke_generation
from pipeline.llm_prompts import build_generation_prompt
from rag_types import AnswerGenerateResult, EvidenceCheckResult, build_citations

_config_logged = False


def _rating_line_from_hits(hits: list[dict]) -> str:
    for hit in hits:
        rating = str(hit.get("rating", "")).strip()
        if not rating:
            continue
        if any(word in rating for word in ("买入", "增持", "推荐", "优于大市", "中性", "卖出")):
            company = str(hit.get("company_name", "")).strip() or "该公司"
            return f"{company}研报投资评级为{rating}。"
    return ""


def _sanitize_llm_body(text: str) -> str:
    """去掉 LLM 可能误输出的参考文献段。"""
    cleaned = text.strip()
    for marker in ("【参考文献】", "参考文献：", "参考文献:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def generate_answer(
    query: str,
    evidence: EvidenceCheckResult,
    *,
    rerank_hits: list[dict] | None = None,
    query_type: str = "factual",
    compare_entities: list[str] | None = None,
) -> AnswerGenerateResult:
    """
    输入：原始问题、已通过 evidence_check 的结果、重排 hits
    输出：AnswerGenerateResult（答案正文 + 引用列表）

    调用方须保证 evidence.passed 为 True。
    """
    global _config_logged

    cfg = resolve_generation_config()
    if not _config_logged:
        print(f"[生成] {describe_generation_config(cfg)}")
        _config_logged = True

    hits = rerank_hits if rerank_hits is not None else evidence.evidence_hits
    citation_count = evidence.citation_count or 3
    used_hits = select_evidence_hits(
        hits,
        query,
        query_type=query_type,
        compare_entities=compare_entities or [],
        top_k=citation_count,
    )
    citations = build_citations(used_hits)

    system_prompt, user_prompt = build_generation_prompt(
        query,
        used_hits,
        citations,
        query_type=query_type,
        compare_entities=compare_entities or [],
        cfg=cfg,
    )
    body = _sanitize_llm_body(invoke_generation(system_prompt, user_prompt, cfg))

    if any(keyword in query for keyword in ("评级", "投资评级", "买入", "增持")):
        rating_line = _rating_line_from_hits(used_hits)
        if rating_line and rating_line not in body:
            body = f"{rating_line} {body}"

    ref_lines = [c.format_line() for c in citations]
    ref_block = "\n".join(["", "【参考文献】", *ref_lines])
    answer = f"根据检索到的研报资料：{body}{ref_block}"

    return AnswerGenerateResult(
        query=query,
        answer=answer,
        citations=citations,
        evidence_hits=used_hits,
        top_rerank_score=evidence.top_rerank_score,
    )
