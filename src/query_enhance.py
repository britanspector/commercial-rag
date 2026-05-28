"""
检索查询增强：BM25 扩展、对比型多实体抽取。
"""

from __future__ import annotations

import re


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
