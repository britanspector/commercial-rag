"""
RAG 答案生成：引用溯源 + 低分拒答（不依赖 Milvus / Embedding）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_constants import DEFAULT_RERANK_REFUSAL_THRESHOLD, REFUSAL_MESSAGE
from rag_tokens import must_tokens_match
from reranker import hit_passage_text


@dataclass
class Citation:
    index: int
    chunk_id: str
    company_name: str
    section_title: str
    page_start: int
    page_end: int
    display_name: str
    score_rerank: float

    def format_line(self) -> str:
        page = ""
        if self.page_start:
            page = f", 第{self.page_start}页" if self.page_start == self.page_end else (
                f", 第{self.page_start}-{self.page_end}页"
            )
        return (
            f"[{self.index}] {self.company_name} — {self.section_title}"
            f"{page} (chunk: {self.chunk_id}, rerank={self.score_rerank:.3f})"
        )


@dataclass
class RAGAnswer:
    query: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    top_rerank_score: float = 0.0
    citations: list[Citation] = field(default_factory=list)
    evidence_hits: list[dict] = field(default_factory=list)


def build_citations(hits: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        citations.append(
            Citation(
                index=index,
                chunk_id=str(hit.get("chunk_id", "")),
                company_name=str(hit.get("company_name", "")),
                section_title=str(hit.get("section_title", "")),
                page_start=int(hit.get("page_start") or 0),
                page_end=int(hit.get("page_end") or 0),
                display_name=str(hit.get("display_name", "")),
                score_rerank=float(hit.get("score_rerank") or hit.get("score") or 0.0),
            )
        )
    return citations


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


def generate_answer_with_citations(
    query: str,
    hits: list[dict],
    refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
) -> RAGAnswer:
    if not hits:
        return RAGAnswer(
            query=query,
            answer=REFUSAL_MESSAGE,
            refused=True,
            refusal_reason="no_hits",
            top_rerank_score=0.0,
        )

    top_score = float(hits[0].get("score_rerank") or hits[0].get("score") or 0.0)
    if top_score < refusal_threshold:
        return RAGAnswer(
            query=query,
            answer=REFUSAL_MESSAGE,
            refused=True,
            refusal_reason="low_rerank_score",
            top_rerank_score=top_score,
            evidence_hits=hits[:3],
        )

    citations = build_citations(hits[:3])
    snippets: list[str] = []
    for citation, hit in zip(citations, hits[:3]):
        snippet = _extractive_snippet(hit_passage_text(hit))
        if snippet:
            snippets.append(f"{snippet} [{citation.index}]")

    body = " ".join(snippets) if snippets else _extractive_snippet(hit_passage_text(hits[0])) + " [1]"

    if any(keyword in query for keyword in ("评级", "投资评级", "买入", "增持")):
        rating_line = _rating_line_from_hits(hits[:3])
        if rating_line:
            body = f"{rating_line} {body}"
    ref_block = "\n".join(["", "【参考文献】", *[c.format_line() for c in citations]])
    answer = f"根据检索到的研报资料：{body}{ref_block}"

    return RAGAnswer(
        query=query,
        answer=answer,
        refused=False,
        top_rerank_score=top_score,
        citations=citations,
        evidence_hits=hits[:3],
    )


def is_answer_factually_supported(
    answer: str,
    must_contain_any: list[str],
    gold_answer: str = "",
) -> bool:
    if not answer or answer == REFUSAL_MESSAGE:
        return False
    if must_contain_any:
        return must_tokens_match(answer, must_contain_any)
    if gold_answer:
        gold_tokens = re.findall(r"[\u4e00-\u9fff]{2,}|\d+\.?\d*", gold_answer)
        if gold_tokens:
            matched = sum(1 for token in gold_tokens if token in answer)
            return matched >= min(2, len(gold_tokens))
    return len(answer) > 40
