"""
answer_generate：基于通过校验的证据片段生成带引用的抽取式答案。

答案正文与【参考文献】均包含来源文档与页码，供 Citation Accuracy 评测。
"""

from __future__ import annotations

import re

from rag_types import AnswerGenerateResult, EvidenceCheckResult, build_citations
from reranker import hit_passage_text


def _rating_line_from_hits(hits: list[dict]) -> str:
    for hit in hits:
        rating = str(hit.get("rating", "")).strip()
        if not rating:
            continue
        if any(word in rating for word in ("买入", "增持", "推荐", "优于大市", "中性", "卖出")):
            company = str(hit.get("company_name", "")).strip() or "该公司"
            return f"{company}研报投资评级为{rating}。"
    return ""


def _extractive_snippet(text: str, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    break_at = text.rfind("。", 0, max_chars)
    if break_at > 80:
        return text[: break_at + 1]
    return text[:max_chars] + "…"


def _citation_source_prefix(citation) -> str:
    doc = citation.source_document()
    page = citation.page_label()
    if doc and page != "页码未知":
        return f"据《{doc}》({page})："
    if doc:
        return f"据《{doc}》："
    return ""


def generate_answer(
    query: str,
    evidence: EvidenceCheckResult,
    *,
    rerank_hits: list[dict] | None = None,
) -> AnswerGenerateResult:
    """
    输入：原始问题、已通过 evidence_check 的结果、重排 hits
    输出：AnswerGenerateResult（答案正文 + 引用列表）

    调用方须保证 evidence.passed 为 True。
    """
    hits = rerank_hits if rerank_hits is not None else evidence.evidence_hits
    citations = build_citations(hits[: evidence.citation_count or 3])
    snippets: list[str] = []

    for citation, hit in zip(citations, hits[: len(citations)]):
        snippet = _extractive_snippet(hit_passage_text(hit))
        if not snippet:
            continue
        prefix = _citation_source_prefix(citation)
        snippets.append(f"{prefix}{snippet} [{citation.index}]")

    if snippets:
        body = " ".join(snippets)
    else:
        citation = citations[0] if citations else None
        prefix = _citation_source_prefix(citation) if citation else ""
        body = f"{prefix}{_extractive_snippet(hit_passage_text(hits[0]))} [1]"

    if any(keyword in query for keyword in ("评级", "投资评级", "买入", "增持")):
        rating_line = _rating_line_from_hits(hits[:3])
        if rating_line:
            body = f"{rating_line} {body}"

    ref_block = "\n".join(["", "【参考文献】", *[c.format_line() for c in citations]])
    answer = f"根据检索到的研报资料：{body}{ref_block}"

    return AnswerGenerateResult(
        query=query,
        answer=answer,
        citations=citations,
        evidence_hits=hits[: len(citations)],
        top_rerank_score=evidence.top_rerank_score,
    )
