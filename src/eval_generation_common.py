"""
生成质量评测：Citation Accuracy、Refusal Accuracy，及 RAGAS 数据集构建。

与 eval_retrieval.EvalQuestion、RAGPipelineResult 对齐，供 eval_generation.py 批量调用。
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from eval_retrieval import EvalQuestion, is_hit_relevant
from rag_answer import is_answer_factually_supported
from rag_tokens import must_tokens_match
from rag_types import Citation, RAGPipelineResult, RetrievedChunk
from reranker import hit_passage_text

if TYPE_CHECKING:
    from datasets import Dataset


def _retrieved_chunk_as_hit(chunk: RetrievedChunk) -> dict:
    return chunk.to_dict()


def retrieval_hit_in_rerank(question: EvalQuestion, result: RAGPipelineResult) -> bool:
    """Top-K 重排结果中是否至少有一条与标注相关。"""
    return any(
        is_hit_relevant(_retrieved_chunk_as_hit(chunk), question)
        for chunk in result.rerank_hits
    )


def _citation_chunks_in_answer(answer: str, citations: list[Citation]) -> bool:
    """答案是否包含引用序号或页码线索。"""
    if not citations:
        return False
    if re.search(r"\[\d+\]", answer):
        return True
    if "参考文献" in answer:
        return True
    for citation in citations:
        if citation.page_start and str(citation.page_start) in answer:
            return True
        doc = citation.source_document()
        if doc and doc[:8] in answer:
            return True
    return False


def evaluate_citation_accuracy(
    question: EvalQuestion,
    result: RAGPipelineResult,
) -> dict[str, Any]:
    """
    Citation Accuracy：非拒答时引用是否可溯源且与标注一致。

    子项（均记录在 citation_checks JSON 逻辑中）：
    - has_citations
    - pages_present
    - sources_present
    - cited_chunk_relevant（引用 chunk 在 rerank 中且 is_hit_relevant）
    - must_tokens_in_answer（若标注了 must_contain_any）
    - doc_stock_aligned（若标注 doc_id / stock_code）
    """
    if result.refused:
        ok = len(result.citations) == 0
        return {
            "citation_applicable": False,
            "citation_accuracy": 1.0 if ok else 0.0,
            "has_citations": len(result.citations) > 0,
            "pages_present": True,
            "sources_present": True,
            "cited_chunk_relevant": True,
            "must_tokens_in_answer": True,
            "doc_stock_aligned": True,
            "refs_in_answer": True,
        }

    citations = result.citations
    rerank_by_id = {c.chunk_id: _retrieved_chunk_as_hit(c) for c in result.rerank_hits}

    has_citations = len(citations) > 0
    pages_present = all(c.page_start > 0 for c in citations) if citations else False
    sources_present = all(
        bool(c.source_document() or c.doc_id) for c in citations
    ) if citations else False

    cited_chunk_relevant = False
    if citations:
        for citation in citations:
            hit = rerank_by_id.get(citation.chunk_id)
            if hit and is_hit_relevant(hit, question):
                cited_chunk_relevant = True
                break
        if not cited_chunk_relevant and question.gold_chunk_ids:
            cited_chunk_relevant = any(
                c.chunk_id in question.gold_chunk_ids for c in citations
            )

    must_tokens_in_answer = True
    if question.must_contain_any:
        must_tokens_in_answer = must_tokens_match(result.answer, question.must_contain_any)

    doc_stock_aligned = True
    if citations:
        for citation in citations:
            hit = rerank_by_id.get(citation.chunk_id, {})
            stock = str(hit.get("stock_code", "")).strip()
            doc = str(hit.get("doc_id", "")).strip()
            if question.stock_code and stock and stock != question.stock_code:
                doc_stock_aligned = False
                break
            if question.doc_id and doc and doc != question.doc_id:
                doc_stock_aligned = False
                break
            if question.negative_stock_codes and stock in question.negative_stock_codes:
                doc_stock_aligned = False
                break

    refs_in_answer = _citation_chunks_in_answer(result.answer, citations)

    checks = [
        has_citations,
        pages_present,
        sources_present,
        cited_chunk_relevant,
        must_tokens_in_answer,
        doc_stock_aligned,
        refs_in_answer,
    ]
    score = sum(checks) / len(checks) if checks else 0.0

    return {
        "citation_applicable": True,
        "citation_accuracy": score,
        "has_citations": has_citations,
        "pages_present": pages_present,
        "sources_present": sources_present,
        "cited_chunk_relevant": cited_chunk_relevant,
        "must_tokens_in_answer": must_tokens_in_answer,
        "doc_stock_aligned": doc_stock_aligned,
        "refs_in_answer": refs_in_answer,
    }


def evaluate_refusal_accuracy(
    question: EvalQuestion,
    result: RAGPipelineResult,
    *,
    retrieval_hit: bool | None = None,
) -> dict[str, Any]:
    """
    Refusal Accuracy：是否应在「检索未命中关键证据」时拒答。

    should_refuse = not retrieval_hit_in_rerank
    refusal_correct = (refused == should_refuse)
    """
    if retrieval_hit is None:
        retrieval_hit = retrieval_hit_in_rerank(question, result)

    should_refuse = not retrieval_hit
    refused = result.refused
    refusal_correct = refused == should_refuse

    evidence_passed = (
        result.evidence_check.passed if result.evidence_check else retrieval_hit
    )
    evidence_should_refuse = not evidence_passed

    return {
        "retrieval_hit": retrieval_hit,
        "should_refuse": should_refuse,
        "evidence_passed": evidence_passed,
        "evidence_should_refuse": evidence_should_refuse,
        "refusal_correct_evidence": refused == evidence_should_refuse,
        "refused": refused,
        "refusal_correct": refusal_correct,
        "refusal_reason": result.refusal_reason,
        "refusal_message": (
            result.evidence_check.refusal_message
            if result.evidence_check
            else ""
        ),
        "top_rerank_score": result.top_rerank_score,
    }


def evaluate_answer_support(
    question: EvalQuestion,
    result: RAGPipelineResult,
) -> dict[str, Any]:
    """规则型答案支持度（不依赖 LLM），辅助对比 RAGAS。"""
    if result.refused:
        return {
            "answer_factually_supported": False,
            "answer_supported_rule": True,
        }

    fact_ok = is_answer_factually_supported(
        result.answer,
        question.must_contain_any,
        question.gold_answer,
    )
    contexts = contexts_from_result(result)
    context_blob = "\n".join(contexts)
    must_in_ctx = (
        must_tokens_match(context_blob, question.must_contain_any)
        if question.must_contain_any
        else True
    )
    must_in_ans = (
        must_tokens_match(result.answer, question.must_contain_any)
        if question.must_contain_any
        else True
    )
    supported = must_in_ctx and must_in_ans and fact_ok

    return {
        "answer_factually_supported": fact_ok,
        "answer_supported_rule": supported,
        "must_tokens_in_context": must_in_ctx,
        "must_tokens_in_answer": must_in_ans,
    }


def contexts_from_result(result: RAGPipelineResult, max_hits: int = 3) -> list[str]:
    """供 RAGAS 使用的检索上下文列表。"""
    contexts: list[str] = []
    for chunk in result.rerank_hits[:max_hits]:
        text = hit_passage_text(_retrieved_chunk_as_hit(chunk)).strip()
        if text:
            contexts.append(text[:4000])
    if not contexts and result.evidence_check:
        for hit in result.evidence_check.evidence_hits[:max_hits]:
            text = hit_passage_text(hit).strip()
            if text:
                contexts.append(text[:4000])
    return contexts


def build_ragas_dataset(rows: list[dict]) -> Dataset:
    """由评测行构建 RAGAS HuggingFace Dataset。"""
    from datasets import Dataset

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for row in rows:
        if row.get("refused"):
            continue
        ctx = row.get("ragas_contexts") or []
        if not ctx:
            continue
        questions.append(row["query"])
        answers.append(row["answer"])
        contexts.append(ctx)
        ground_truths.append(row.get("gold_answer") or "")

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def _should_use_local_ragas(backend: str | None = None) -> bool:
    import os

    from eval_ragas_config import resolve_ragas_config

    chosen = (backend or os.environ.get("RAGAS_BACKEND", "auto")).lower()
    if chosen in ("ollama", "local"):
        return True
    if chosen == "openai":
        return False
    api_key = os.environ.get("RAGAS_API_KEY") or os.environ.get("OPENAI_API_KEY", "").strip()
    return not bool(api_key)


def run_ragas_evaluate(
    rows: list[dict],
    *,
    llm_model: str | None = None,
    ragas_backend: str | None = None,
    raise_on_error: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    """
    对非拒答样本运行 RAGAS faithfulness + answer_relevancy（逐题按 question_id 写回）。

    默认本地：Ollama qwen3:8b + bge-large-zh（RAGAS_EMBED_BACKEND=bge_local）。
    云端：设置 OPENAI_API_KEY 且 RAGAS_BACKEND=openai。
    """
    if _should_use_local_ragas(ragas_backend):
        from eval_ragas_runner import run_ragas_on_rows

        try:
            summary, updated = run_ragas_on_rows(
                rows,
                backend=ragas_backend or "ollama",
                llm_model=llm_model,
            )
            return summary, updated
        except Exception:
            if raise_on_error:
                raise
            for row in rows:
                row.setdefault("ragas_faithfulness", None)
                row.setdefault("ragas_answer_relevancy", None)
            return {"faithfulness": float("nan"), "answer_relevancy": float("nan")}, rows

    import os

    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        raise ImportError(
            "请安装 RAGAS：pip install ragas datasets langchain-openai langchain-community"
        ) from exc

    dataset = build_ragas_dataset(rows)
    if len(dataset) == 0:
        return {"faithfulness": float("nan"), "answer_relevancy": float("nan")}, []

    llm = _build_ragas_llm(llm_model)
    embeddings = _build_ragas_embeddings()
    if llm is None:
        raise ValueError("RAGAS openai 后端需要 OPENAI_API_KEY 或 RAGAS_API_KEY")

    kwargs: dict[str, Any] = {
        "raise_exceptions": raise_on_error,
        "llm": llm,
    }
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        **kwargs,
    )

    scores: dict[str, float] = {}
    for key in ("faithfulness", "answer_relevancy"):
        val = result.get(key) if hasattr(result, "get") else getattr(result, key, None)
        if val is not None:
            try:
                scores[key] = float(val)
            except (TypeError, ValueError):
                scores[key] = float("nan")

    non_refused = [r for r in rows if not r.get("refused") and r.get("ragas_contexts")]
    if hasattr(result, "to_pandas"):
        pdf = result.to_pandas()
        for row, (_, series) in zip(non_refused, pdf.iterrows()):
            if "faithfulness" in series:
                row["ragas_faithfulness"] = series.get("faithfulness")
            if "answer_relevancy" in series:
                row["ragas_answer_relevancy"] = series.get("answer_relevancy")

    return scores, rows


def _build_ragas_llm(model_name: str | None = None):
    import os

    api_key = os.environ.get("RAGAS_API_KEY") or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        return None

    model = model_name or os.environ.get("RAGAS_LLM_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("RAGAS_OPENAI_BASE") or os.environ.get("OPENAI_API_BASE")
    chat_kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "temperature": 0}
    if base_url:
        chat_kwargs["base_url"] = base_url
    return LangchainLLMWrapper(ChatOpenAI(**chat_kwargs))


def _build_ragas_embeddings():
    import os

    api_key = os.environ.get("RAGAS_API_KEY") or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        return None

    model = os.environ.get("RAGAS_EMBED_MODEL", "text-embedding-3-small")
    base_url = os.environ.get("RAGAS_OPENAI_BASE") or os.environ.get("OPENAI_API_BASE")
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(**kwargs))


def row_from_pipeline(
    question: EvalQuestion,
    result: RAGPipelineResult,
    *,
    strategy: str = "pipeline_chat",
) -> dict[str, Any]:
    """合并单题 Pipeline 结果与各项生成指标。"""
    retrieval_hit = retrieval_hit_in_rerank(question, result)
    citation = evaluate_citation_accuracy(question, result)
    refusal = evaluate_refusal_accuracy(question, result, retrieval_hit=retrieval_hit)
    support = evaluate_answer_support(question, result)
    contexts = contexts_from_result(result)

    citations_payload = [asdict(c) for c in result.citations]
    evidence_json = (
        result.evidence_check.to_dict() if result.evidence_check else {}
    )

    return {
        "question_id": question.id,
        "query_type": question.query_type,
        "category": question.category,
        "strategy": strategy,
        "query": question.query,
        "gold_answer": question.gold_answer,
        "stock_code": question.stock_code,
        "doc_id": question.doc_id,
        "answer": result.answer,
        "answer_preview": result.answer[:300],
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "refusal_message": (
            result.evidence_check.refusal_message if result.evidence_check else ""
        ),
        "top_rerank_score": result.top_rerank_score,
        "citation_count": len(result.citations),
        "citations_json": citations_payload,
        "evidence_check_json": evidence_json,
        "ragas_contexts": contexts,
        "retrieval_hit": retrieval_hit,
        **citation,
        **refusal,
        **support,
        "top1_chunk_id": result.rerank_hits[0].chunk_id if result.rerank_hits else "",
    }


def aggregate_generation_metrics(rows: list[dict]) -> dict[str, float | int]:
    n = len(rows)
    if n == 0:
        return {"question_count": 0}

    applicable = [r for r in rows if r.get("citation_applicable")]
    ragas_rows = [r for r in rows if not r.get("refused") and r.get("ragas_faithfulness") is not None]

    def _mean(key: str, subset: list[dict] | None = None) -> float:
        data = subset if subset is not None else rows
        vals = [r[key] for r in data if r.get(key) is not None and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "question_count": n,
        "refusal_rate": sum(1 for r in rows if r.get("refused")) / n,
        "refusal_accuracy": _mean("refusal_correct"),
        "refusal_accuracy_evidence": _mean("refusal_correct_evidence"),
        "citation_accuracy": _mean("citation_accuracy", applicable) if applicable else float("nan"),
        "citation_accuracy_applicable_n": len(applicable),
        "answer_factually_supported_rate": _mean("answer_factually_supported"),
        "answer_supported_rule_rate": _mean("answer_supported_rule"),
        "retrieval_hit_rate": _mean("retrieval_hit"),
        "faithfulness_ragas": _mean("ragas_faithfulness", ragas_rows) if ragas_rows else float("nan"),
        "answer_relevancy_ragas": _mean("ragas_answer_relevancy", ragas_rows) if ragas_rows else float("nan"),
        "ragas_scored_n": len(ragas_rows),
    }


def aggregate_by_query_type(rows: list[dict]) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row.get("query_type", "unknown"), []).append(row)

    output: list[dict] = []
    for query_type, group in sorted(by_type.items()):
        metrics = aggregate_generation_metrics(group)
        metrics["query_type"] = query_type
        output.append(metrics)
    return output
