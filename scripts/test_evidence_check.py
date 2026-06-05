"""evidence_check 单元冒烟（无需加载向量模型）。"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from pipeline.evidence_check import check_evidence
from rag_types import RerankStepResult


def _hit(**kwargs) -> dict:
    base = {
        "chunk_id": "c1",
        "doc_id": "DOC1",
        "display_name": "测试研报",
        "company_name": "测试公司",
        "stock_code": "000001",
        "section_title": "财务摘要",
        "page_start": 5,
        "page_end": 5,
        "text": "2025年营业收入同比增长12.3%，归母净利润10.5亿元。",
        "score_rerank": 0.55,
    }
    base.update(kwargs)
    return base


def test_no_hits() -> None:
    result = check_evidence([])
    assert not result.passed
    assert result.refusal_reason == "no_hits"


def test_low_score() -> None:
    result = check_evidence(
        RerankStepResult(hits=[_hit(score_rerank=0.2)], query="q", top_rerank_score=0.2, rerank_top_k=5)
    )
    assert not result.passed
    assert result.refusal_reason == "low_rerank_score"
    assert "0.20" in result.refusal_message or "0.2" in result.refusal_message


def test_pass_with_citation() -> None:
    result = check_evidence(
        RerankStepResult(hits=[_hit()], query="营收多少", top_rerank_score=0.55, rerank_top_k=5)
    )
    assert result.passed
    assert result.citation_count >= 1


def test_missing_page() -> None:
    result = check_evidence(
        RerankStepResult(
            hits=[_hit(page_start=0, display_name="", doc_id="", filename="")],
            query="q",
            top_rerank_score=0.6,
            rerank_top_k=5,
        )
    )
    assert not result.passed
    assert result.refusal_reason == "missing_source_page"


def main() -> None:
    test_no_hits()
    test_low_score()
    test_pass_with_citation()
    test_missing_page()
    print("evidence_check: all passed")


if __name__ == "__main__":
    main()
