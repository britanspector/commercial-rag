"""
检索查询增强：BM25 扩展、对比型多实体抽取。
"""

from __future__ import annotations

import re

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})E?")
_COMPARE_NOISE_RE = re.compile(
    r"(?:对比|比较|哪家|谁更高|谁更低|孰高|孰低|哪个更高|哪个更|差异|是否|怎样|如何|多少|是什么)"
)
_PUNCT_RE = re.compile(r"[？?。．.!！,，;；:：、的]+")
_METRIC_KEYWORDS = (
    "每股收益",
    "净利润",
    "归母净利润",
    "营业收入",
    "营收",
    "毛利率",
    "市盈率",
    "EPS",
    "PE",
    "发电量",
    "装机容量",
    "投资评级",
    "评级",
    "预测",
)


def enhance_bm25_query(query: str, stock_code: str = "") -> str:
    """为 BM25 补充股票代码与常见研报章节词。"""
    parts = [query.strip()]
    if stock_code:
        parts.append(stock_code)

    if any(keyword in query for keyword in ("EPS", "PE", "市盈率", "每股收益", "净利润", "营收", "预测", "毛利率")):
        parts.append("盈利预测 财务 投资要点")
    if any(keyword in query for keyword in ("评级", "买入", "增持", "推荐")):
        parts.append("投资评级 买入 增持 推荐")
    if "风险" in query:
        parts.append("风险提示 风险因素")
    if any(keyword in query for keyword in ("对比", "哪家", "哪个更高", "哪个更", "差异", "比较")):
        parts.append("盈利预测 业绩")

    return " ".join(parts)


def extract_compare_entities(query: str) -> list[str]:
    """从对比型问题中抽取两个公司/主题短语（启发式）。"""
    patterns = [
        r"^(.+?)(?:和|与)(.+?)(?:\d|哪家|谁|哪个|哪|对比|比较|更高|更低|更积极|更|规模|增速|差异|是否|布局)",
        r"^(.+?)(?:和|与)(.+?)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, query.strip())
        if not match:
            continue
        left = match.group(1).strip()
        right = re.sub(r"[\d０-９].*$", "", match.group(2)).strip()
        if len(left) >= 2 and len(right) >= 2:
            return [left, right]
    return []


def _extract_metric_tail(query: str, *, entities: list[str]) -> str:
    """从对比问句提取指标/主题尾（去掉公司与对比用语）。"""
    text = query.strip()
    for ent in entities:
        if ent:
            text = text.replace(ent, " ")
    text = _YEAR_RE.sub(" ", text)
    text = _COMPARE_NOISE_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    kept: list[str] = []
    upper = text.upper()
    for kw in _METRIC_KEYWORDS:
        if kw.upper() in upper or kw in text:
            kept.append(kw)
    if kept:
        return " ".join(dict.fromkeys(kept))
    return text[:40].strip() if text else "盈利预测"


def build_entity_sub_query(
    entity: str,
    query: str,
    *,
    other_entities: list[str] | None = None,
) -> str:
    """
    为对比题单主体生成检索子查询：「主体 + 年份 + 指标」，不含另一主体与对比词。
    """
    entity = (entity or "").strip()
    if not entity:
        return query.strip()

    others = [e.strip() for e in (other_entities or []) if e.strip() and e.strip() != entity]
    year_match = _YEAR_RE.search(query)
    year_part = f"{year_match.group(1)}年" if year_match else ""
    metric_part = _extract_metric_tail(query, entities=[entity, *others])
    core = " ".join(part for part in (entity, year_part, metric_part) if part).strip()
    return enhance_bm25_query(core, stock_code="")


def build_comparative_sub_queries(query: str, entities: list[str]) -> list[tuple[str, str]]:
    """返回 [(entity, sub_query), ...]。"""
    unique: list[str] = []
    for ent in entities:
        ent = ent.strip()
        if ent and ent not in unique:
            unique.append(ent)
    return [
        (ent, build_entity_sub_query(ent, query, other_entities=[e for e in unique if e != ent]))
        for ent in unique[:3]
    ]


def hybrid_vector_weight(query_type: str, query: str, default: float = 0.4) -> float:
    """事实型数字题略偏 BM25（向量权重更低）。"""
    if query_type == "comparative":
        return 0.45
    if query_type == "summary":
        return 0.5
    if any(
        keyword in query
        for keyword in ("EPS", "PE", "市盈率", "净利润", "归母", "营收", "预测", "毛利率", "%")
    ):
        return default
    return 0.5
