"""
Evidence 选择：在 rerank Top-N 池内按问题意图与章节元数据重排，选出喂给 LLM 的 Top-K。

参考开源实践（规则化落地，避免额外 LLM 调用）：
- ChunkRAG：生成前过滤无关 chunk
- SetRAG / OverSearchGuard：集合式选取、多样性、冗余惩罚
- LlamaIndex / RAGFlow：metadata / section 过滤与 boost
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reranker import hit_passage_text

# 意图 → 偏好章节（子串匹配）与应降权章节
_INTENT_RULES: dict[str, dict] = {
    "risk": {
        "keywords": ("风险", "风险提示", "风险因素"),
        "boost_sections": ("风险因素", "风险提示", "核心风险提示", "主要风险"),
        "penalize_sections": ("评级说明", "投资评级摘要", "可比公司", "分析师介绍", "免责声明"),
        "boost": 0.18,
        "penalty": 0.22,
    },
    "rating": {
        "keywords": ("评级", "投资评级", "买入", "增持", "推荐", "优于大市", "中性", "卖出"),
        "boost_sections": ("投资评级摘要", "投资评级", "评级"),
        "penalize_sections": ("评级说明", "可比公司", "免责声明"),
        "boost": 0.20,
        "penalty": 0.18,
    },
    "financial_metric": {
        "keywords": (
            "EPS",
            "PE",
            "市盈率",
            "每股收益",
            "净利润",
            "归母",
            "营收",
            "收入",
            "毛利率",
            "预测",
            "估值",
            "盈利",
        ),
        "boost_sections": (
            "盈利预测",
            "利润表",
            "三大报表",
            "附录",
            "财务",
            "估值",
            "PE",
            "EPS",
        ),
        "penalize_sections": ("评级说明", "可比公司", "投资要点", "事件点评"),
        "boost": 0.12,
        "penalty": 0.10,
    },
    "core_view": {
        "keywords": ("核心观点", "投资要点", "看好", "逻辑", "催化剂", "推荐理由", "观点", "摘要"),
        "boost_sections": ("投资要点", "核心观点", "报告导读", "事件点评", "摘要", "投资评级摘要"),
        "penalize_sections": ("评级说明", "可比公司", "附录", "三大报表"),
        "boost": 0.15,
        "penalty": 0.12,
    },
    "actual_period": {
        "keywords": ("上半年", "H1", "全年", "实际", "2024年", "2025年", "已实现", "业绩"),
        "boost_sections": ("核心观点", "事件点评", "业绩", "经营", "财务分析"),
        "penalize_sections": ("2026E", "2027E", "2028E", "盈利预测", "三大报表预测"),
        "boost": 0.10,
        "penalty": 0.08,
    },
}

_GENERIC_PENALIZE_SECTIONS = (
    "评级说明",
    "分析师介绍",
    "免责声明",
    "研究所",
    "可比公司",
)

_MAX_PER_SECTION = 2
_DEFAULT_POOL_SIZE = 10


@dataclass(frozen=True)
class EvidenceSelectConfig:
    top_k: int = 3
    pool_size: int = _DEFAULT_POOL_SIZE


def _hit_rerank_score(hit: dict) -> float:
    raw = hit.get("score_rerank")
    if raw is not None:
        return float(raw)
    return float(hit.get("score") or 0.0)


def detect_query_intents(query: str, query_type: str) -> set[str]:
    """从问题与题型推断证据偏好意图标签。"""
    intents: set[str] = set()
    q = query or ""

    for name, rule in _INTENT_RULES.items():
        if any(kw in q for kw in rule["keywords"]):
            intents.add(name)

    if query_type == "summary":
        intents.add("core_view")
    if query_type == "comparative":
        intents.add("financial_metric")

    if not intents:
        intents.add("core_view")
    return intents


def _section_matches(section: str, patterns: tuple[str, ...]) -> bool:
    section = section or ""
    return any(p in section for p in patterns)


def _entity_in_hit(hit: dict, entity: str) -> bool:
    entity = entity.strip()
    if len(entity) < 2:
        return False
    blob = " ".join(
        [
            str(hit.get("company_name") or ""),
            str(hit.get("display_name") or ""),
            hit_passage_text(hit),
        ]
    )
    return entity in blob


def _normalize_section_key(section: str) -> str:
    s = (section or "").strip()
    s = re.sub(r"^表\d+\s*", "", s)
    return s[:40] if s else "unknown"


def score_hit_for_selection(
    hit: dict,
    *,
    intents: set[str],
    query: str,
    query_type: str,
) -> float:
    """在 rerank 分基础上叠加章节/内容类型调整分。"""
    score = _hit_rerank_score(hit)
    section = str(hit.get("section_title") or "")
    content_type = str(hit.get("content_type") or "")
    passage = hit_passage_text(hit)

    for intent in intents:
        rule = _INTENT_RULES.get(intent)
        if not rule:
            continue
        if _section_matches(section, rule["boost_sections"]) or _section_matches(
            passage[:200], rule["boost_sections"]
        ):
            score += float(rule["boost"])
        if _section_matches(section, rule["penalize_sections"]):
            score -= float(rule["penalty"])

    if content_type == "rating_headline" and "rating" in intents:
        score += 0.25
    if content_type == "comparable_table" and query_type != "comparative":
        score -= 0.15
    if content_type == "noise":
        score -= 0.30

    if "risk" in intents and _section_matches(section, ("评级说明", "投资评级摘要")):
        score -= 0.25

    if "rating" not in intents and _section_matches(section, ("评级说明",)):
        score -= 0.12

    if not intents.intersection({"financial_metric", "comparative"}) and _section_matches(
        section, _GENERIC_PENALIZE_SECTIONS
    ):
        score -= 0.08

    if "actual_period" in intents and re.search(r"202[6-9]E", section + passage[:120]):
        score -= 0.06

    return score


def _greedy_select(
    ranked: list[tuple[dict, float]],
    *,
    top_k: int,
    query_type: str,
    compare_entities: list[str],
) -> list[dict]:
    """带多样性与对比题实体覆盖的贪心选取。"""
    selected: list[dict] = []
    section_counts: dict[str, int] = {}
    covered_entities: set[str] = set()

    # 对比题：先为每个主体各选一条最高分
    if query_type == "comparative" and len(compare_entities) >= 2:
        for entity in compare_entities[:2]:
            for hit, _ in ranked:
                if hit in selected:
                    continue
                if _entity_in_hit(hit, entity):
                    selected.append(hit)
                    covered_entities.add(entity)
                    key = _normalize_section_key(str(hit.get("section_title") or ""))
                    section_counts[key] = section_counts.get(key, 0) + 1
                    break

    for hit, _ in ranked:
        if len(selected) >= top_k:
            break
        if hit in selected:
            continue
        key = _normalize_section_key(str(hit.get("section_title") or ""))
        if section_counts.get(key, 0) >= _MAX_PER_SECTION:
            continue
        selected.append(hit)
        section_counts[key] = section_counts.get(key, 0) + 1

    if len(selected) < top_k:
        for hit, _ in ranked:
            if len(selected) >= top_k:
                break
            if hit not in selected:
                selected.append(hit)

    return selected[:top_k]


_INTENT_MISMATCH_SECTIONS: dict[str, tuple[str, ...]] = {
    "risk": ("评级说明", "投资评级摘要", "可比公司", "分析师介绍"),
    "financial_metric": ("评级说明", "投资要点", "事件点评"),
    "rating": (),
    "core_view": ("评级说明", "可比公司"),
}


def top_evidence_intent_aligned(
    query: str,
    query_type: str,
    hits: list[dict],
    *,
    pool_size: int = 10,
) -> tuple[bool, str]:
    """
    判断重排池内最优证据是否与问题意图匹配（供 evidence_check 硬约束）。
    返回 (是否通过, 说明)。
    """
    if not hits:
        return False, "no hits"

    intents = detect_query_intents(query, query_type)
    selected = select_evidence_hits(
        hits,
        query,
        query_type=query_type,
        top_k=1,
        pool_size=pool_size,
    )
    if not selected:
        return False, "no selectable evidence"

    top = selected[0]
    section = str(top.get("section_title") or "")
    raw = _hit_rerank_score(top)
    adjusted = score_hit_for_selection(
        top, intents=intents, query=query, query_type=query_type
    )

    for intent in intents:
        blocked = _INTENT_MISMATCH_SECTIONS.get(intent, ())
        if blocked and _section_matches(section, blocked):
            return False, f"intent={intent}, section={section}"

    if adjusted < raw - 0.18:
        return False, f"intent_penalty raw={raw:.3f} adj={adjusted:.3f}"

    return True, section or "ok"


def select_evidence_hits(
    hits: list[dict],
    query: str,
    *,
    query_type: str = "factual",
    compare_entities: list[str] | None = None,
    top_k: int = 3,
    pool_size: int = _DEFAULT_POOL_SIZE,
) -> list[dict]:
    """
    从 rerank 结果中选取生成用证据。

    在 Top pool_size 候选内按意图重打分，再贪心选取 top_k 条（章节去重 + 对比题实体覆盖）。
    """
    if not hits:
        return []

    compare_entities = compare_entities or []
    k = min(top_k, len(hits))
    pool = hits[: min(pool_size, len(hits))]
    intents = detect_query_intents(query, query_type)

    ranked = [
        (hit, score_hit_for_selection(hit, intents=intents, query=query, query_type=query_type))
        for hit in pool
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)

    return _greedy_select(
        ranked,
        top_k=k,
        query_type=query_type,
        compare_entities=compare_entities,
    )
