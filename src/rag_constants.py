"""RAG 生成相关常量（轻量模块，避免循环依赖）。"""

REFUSAL_MESSAGE = "我不确定，当前检索到的证据不足以可靠回答该问题。"

# bge-reranker-v2-m3 normalize=True 时的默认拒答阈值（0~1，越高越严格）
DEFAULT_RERANK_REFUSAL_THRESHOLD = 0.35

DEFAULT_RECALL_TOP_K = 30
DEFAULT_RERANK_TOP_K = 5

# evidence_check：可用证据的最小正文长度（字符）
MIN_EVIDENCE_PASSAGE_CHARS = 40

# 拒答原因码（机器可读，用于评测 Refusal Accuracy）
REFUSAL_REASON_NO_HITS = "no_hits"
REFUSAL_REASON_LOW_RERANK = "low_rerank_score"
REFUSAL_REASON_INSUFFICIENT_PASSAGE = "insufficient_passage"
REFUSAL_REASON_MISSING_SOURCE_PAGE = "missing_source_page"
REFUSAL_REASON_STOCK_MISMATCH = "stock_mismatch"
REFUSAL_REASON_COMPARATIVE_INSUFFICIENT = "comparative_insufficient"
REFUSAL_REASON_WEAK_EVIDENCE_INTENT = "weak_evidence_intent"

REFUSAL_REASON_LABELS: dict[str, str] = {
    REFUSAL_REASON_NO_HITS: "未检索到相关研报片段",
    REFUSAL_REASON_LOW_RERANK: "检索证据置信度过低",
    REFUSAL_REASON_INSUFFICIENT_PASSAGE: "证据正文过短或为空",
    REFUSAL_REASON_MISSING_SOURCE_PAGE: "证据缺少可溯源的文档或页码",
    REFUSAL_REASON_STOCK_MISMATCH: "证据与指定股票代码不匹配",
    REFUSAL_REASON_COMPARATIVE_INSUFFICIENT: "对比问题缺少多主体证据",
    REFUSAL_REASON_WEAK_EVIDENCE_INTENT: "检索片段与问题意图不匹配",
}


def format_refusal_message(reason_code: str, **kwargs: float | str) -> str:
    """根据拒答原因码生成面向用户的说明（含关键数值）。"""
    label = REFUSAL_REASON_LABELS.get(reason_code, "证据不足")
    score = kwargs.get("top_rerank_score")
    threshold = kwargs.get("refusal_threshold")

    if reason_code == REFUSAL_REASON_LOW_RERANK and score is not None and threshold is not None:
        return (
            f"我不确定：{label}（Top-1 rerank={float(score):.3f}，阈值={float(threshold):.2f}），"
            "无法基于当前检索结果可靠回答。"
        )
    if reason_code == REFUSAL_REASON_NO_HITS:
        return f"我不确定：{label}，无法回答该问题。"
    if reason_code == REFUSAL_REASON_MISSING_SOURCE_PAGE:
        return f"我不确定：{label}，无法生成带页码引用的答案。"
    if reason_code == REFUSAL_REASON_COMPARATIVE_INSUFFICIENT:
        need = kwargs.get("required_entities", 2)
        got = kwargs.get("found_entities", 0)
        return f"我不确定：{label}（需至少 {need} 个主体，当前仅 {got} 个），无法完成对比回答。"
    if reason_code == REFUSAL_REASON_STOCK_MISMATCH:
        code = kwargs.get("stock_code", "")
        return f"我不确定：检索证据未覆盖指定股票（{code}），无法可靠回答。"
    if reason_code == REFUSAL_REASON_WEAK_EVIDENCE_INTENT:
        detail = kwargs.get("intent_detail", "")
        suffix = f"（{detail}）" if detail else ""
        return f"我不确定：{label}{suffix}，无法基于当前检索结果可靠回答。"
    return f"我不确定：{label}，无法基于当前检索结果可靠回答。"
