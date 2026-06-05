"""
evidence_check：基于 rerank 分数与证据可引用性判断是否允许生成答案。

规则（按顺序，先失败先返回）：
1. no_hits — 无重排结果
2. low_rerank_score — Top-1 rerank < 阈值
3. insufficient_passage — Top-1 正文过短
4. missing_source_page — Top-1 无文档名且无页码（无法Citation Accuracy溯源）
5. stock_mismatch — 指定 stock_code 时 Top-3 均无匹配
6. comparative_insufficient — 对比题 Top-5 主体公司 < 2
7. weak_evidence_intent — 最优证据与问题意图不匹配（章节/意图规则）
"""

from __future__ import annotations

import re

from rag_constants import (
    DEFAULT_RERANK_REFUSAL_THRESHOLD,
    MIN_EVIDENCE_PASSAGE_CHARS,
    REFUSAL_REASON_COMPARATIVE_INSUFFICIENT,
    REFUSAL_REASON_INSUFFICIENT_PASSAGE,
    REFUSAL_REASON_LOW_RERANK,
    REFUSAL_REASON_MISSING_SOURCE_PAGE,
    REFUSAL_REASON_NO_HITS,
    REFUSAL_REASON_STOCK_MISMATCH,
    REFUSAL_REASON_WEAK_EVIDENCE_INTENT,
    format_refusal_message,
)
from pipeline.evidence_select import top_evidence_intent_aligned
from rag_types import EvidenceCheckResult, RerankStepResult
from reranker import hit_passage_text

_COMPARE_HINTS = ("谁更高", "谁更低", "对比", "比较", "vs", "VS", "孰高", "孰低")


def _hit_rerank_score(hit: dict) -> float:
    raw = hit.get("score_rerank")
    if raw is not None:
        return float(raw)
    return float(hit.get("score") or 0.0)


def _passage_length(hit: dict) -> int:
    return len(hit_passage_text(hit).strip())


def _has_citation_anchor(hit: dict) -> bool:
    """是否具备引用溯源要素：页码或研报显示名/文件名。"""
    page = int(hit.get("page_start") or 0)
    if page > 0:
        return True
    display = str(hit.get("display_name") or "").strip()
    if display:
        return True
    filename = str(hit.get("filename") or "").strip()
    if filename:
        return True
    doc_id = str(hit.get("doc_id") or "").strip()
    return bool(doc_id)


def _source_label(hit: dict) -> str:
    return (
        str(hit.get("display_name") or "").strip()
        or str(hit.get("filename") or "").strip()
        or str(hit.get("company_name") or "").strip()
        or str(hit.get("doc_id") or "").strip()
    )


def _distinct_companies(hits: list[dict]) -> set[str]:
    names: set[str] = set()
    for hit in hits:
        name = str(hit.get("company_name") or "").strip()
        if name:
            names.add(name)
    return names


def _stock_code_in_hit(hit: dict, stock_code: str) -> bool:
    code = stock_code.strip()
    if not code:
        return True
    hit_code = str(hit.get("stock_code") or "").strip()
    if hit_code and (hit_code == code or code in hit_code or hit_code in code):
        return True
    blob = " ".join(
        [
            str(hit.get("company_name") or ""),
            str(hit.get("display_name") or ""),
            hit_passage_text(hit),
        ]
    )
    return code in blob


def _is_comparative_query(query: str, query_type: str, compare_entities: list[str]) -> bool:
    if query_type == "comparative":
        return True
    if len(compare_entities) >= 2:
        return True
    q = query or ""
    return any(hint in q for hint in _COMPARE_HINTS) or bool(
        re.search(r"和.+?[,，、].+?谁", q)
    )


def _record_check(name: str, passed: bool, detail: str = "") -> dict:
    return {"name": name, "passed": passed, "detail": detail}


def check_evidence(
    rerank_result: RerankStepResult | list[dict],
    *,
    refusal_threshold: float = DEFAULT_RERANK_REFUSAL_THRESHOLD,
    query: str = "",
    stock_code: str = "",
    query_type: str = "factual",
    compare_entities: list[str] | None = None,
) -> EvidenceCheckResult:
    """
    输入：RerankStepResult 或 rerank 后的 hits 列表，及可选问题上下文。
    输出：EvidenceCheckResult（是否通过、拒答码、用户可读拒答说明、证据片段）。
    """
    if isinstance(rerank_result, RerankStepResult):
        hits = rerank_result.hits
        top_score = rerank_result.top_rerank_score
    else:
        hits = rerank_result
        top_score = _hit_rerank_score(hits[0]) if hits else 0.0

    compare_entities = compare_entities or []
    checks: list[dict] = []
    detail: dict[str, float | int | str | bool] = {
        "refusal_threshold": refusal_threshold,
        "top_rerank_score": top_score,
        "hit_count": len(hits),
    }

    def _fail(
        reason_code: str,
        *,
        evidence_slice: list[dict] | None = None,
        message_kwargs: dict | None = None,
    ) -> EvidenceCheckResult:
        kwargs = {"top_rerank_score": top_score, "refusal_threshold": refusal_threshold}
        if message_kwargs:
            kwargs.update(message_kwargs)
        return EvidenceCheckResult(
            passed=False,
            top_rerank_score=top_score,
            refusal_reason=reason_code,
            refusal_message=format_refusal_message(reason_code, **kwargs),
            refusal_detail={**detail, **(message_kwargs or {})},
            evidence_hits=(evidence_slice if evidence_slice is not None else hits[:3]),
            citation_count=0,
            checks=checks,
        )

    if not hits:
        checks.append(_record_check("has_hits", False))
        return _fail(REFUSAL_REASON_NO_HITS, evidence_slice=[])

    checks.append(_record_check("has_hits", True, f"count={len(hits)}"))

    if top_score < refusal_threshold:
        checks.append(
            _record_check(
                "rerank_threshold",
                False,
                f"top={top_score:.4f} < {refusal_threshold:.2f}",
            )
        )
        return _fail(REFUSAL_REASON_LOW_RERANK)

    checks.append(
        _record_check("rerank_threshold", True, f"top={top_score:.4f} >= {refusal_threshold:.2f}")
    )

    top_hit = hits[0]
    passage_len = _passage_length(top_hit)
    detail["top_passage_chars"] = passage_len

    if passage_len < MIN_EVIDENCE_PASSAGE_CHARS:
        checks.append(
            _record_check(
                "passage_length",
                False,
                f"chars={passage_len} < {MIN_EVIDENCE_PASSAGE_CHARS}",
            )
        )
        return _fail(REFUSAL_REASON_INSUFFICIENT_PASSAGE)

    checks.append(_record_check("passage_length", True, f"chars={passage_len}"))

    if not _has_citation_anchor(top_hit):
        checks.append(_record_check("citation_anchor", False, "no page or document label"))
        return _fail(REFUSAL_REASON_MISSING_SOURCE_PAGE)

    checks.append(
        _record_check(
            "citation_anchor",
            True,
            f"source={_source_label(top_hit)}, page={top_hit.get('page_start')}",
        )
    )

    if stock_code.strip():
        top_slice = hits[:3]
        stock_ok = any(_stock_code_in_hit(hit, stock_code) for hit in top_slice)
        detail["stock_code"] = stock_code.strip()
        if not stock_ok:
            checks.append(_record_check("stock_code_match", False, stock_code))
            return _fail(
                REFUSAL_REASON_STOCK_MISMATCH,
                message_kwargs={"stock_code": stock_code.strip()},
            )
        checks.append(_record_check("stock_code_match", True, stock_code))

    if _is_comparative_query(query, query_type, compare_entities):
        companies = _distinct_companies(hits[:5])
        detail["distinct_companies"] = len(companies)
        if len(companies) < 2:
            checks.append(
                _record_check(
                    "comparative_entities",
                    False,
                    f"found={len(companies)}",
                )
            )
            return _fail(
                REFUSAL_REASON_COMPARATIVE_INSUFFICIENT,
                message_kwargs={
                    "required_entities": 2,
                    "found_entities": len(companies),
                },
            )
        checks.append(
            _record_check("comparative_entities", True, f"found={len(companies)}")
        )

    intent_ok, intent_detail = top_evidence_intent_aligned(
        query, query_type, hits[:10]
    )
    detail["evidence_intent_detail"] = intent_detail
    if not intent_ok:
        checks.append(
            _record_check("evidence_intent", False, intent_detail),
        )
        return _fail(
            REFUSAL_REASON_WEAK_EVIDENCE_INTENT,
            message_kwargs={"intent_detail": intent_detail},
        )
    checks.append(_record_check("evidence_intent", True, intent_detail))

    usable = hits[:3]
    return EvidenceCheckResult(
        passed=True,
        top_rerank_score=top_score,
        refusal_reason="",
        refusal_message="",
        refusal_detail=detail,
        evidence_hits=usable,
        citation_count=min(len(usable), 3),
        checks=checks,
    )
