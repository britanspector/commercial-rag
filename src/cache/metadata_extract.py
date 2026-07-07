"""从用户问题中提取 metadata 线索（年份、公司等）。"""

from __future__ import annotations

import re

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
# 常见 A 股公司名后缀
_COMPANY_SUFFIX_RE = re.compile(
    r"([\u4e00-\u9fff]{2,12}(?:科技|股份|集团|国际|装备|新能|微|技术|核电|易佰|传媒|信息|材料|电子|软件|网络|数据|智能|光电|半导体)?)"
)
# 「请告诉我2024年澜起科技的每股收益」→ 年份后的公司名
_YEAR_COMPANY_RE = re.compile(
    r"(?:20\d{2})年\s*([\u4e00-\u9fff]{2,12}(?:科技|股份|集团|国际|装备|新能|微|技术|核电|易佰|传媒|信息|材料|电子|软件|网络|数据|智能|光电|半导体)?)"
)
# 「澜起科技2024年」→ 年份前的公司名
_COMPANY_YEAR_RE = re.compile(
    r"^([\u4e00-\u9fff]{2,12}(?:科技|股份|集团|国际|装备|新能|微|技术|核电|易佰|传媒|信息|材料|电子|软件|网络|数据|智能|光电|半导体)?)\s*(?:20\d{2})"
)

_NON_COMPANY_WORDS = frozenset(
    {
        "请问",
        "请告诉",
        "想了解",
        "帮忙",
        "多少",
        "是什么",
        "如何",
        "预测",
        "盈利",
        "财务",
        "投资",
        "评级",
        "风险",
        "营收",
        "净利润",
        "毛利率",
        "可比",
        "公司",
        "主要",
        "核心",
        "业务",
    }
)


def extract_report_year(text: str) -> str:
    """提取问题中的报告年份（取最后一次出现的 20xx）。"""
    matches = _YEAR_RE.findall(text or "")
    return matches[-1] if matches else ""


def _is_plausible_company(name: str) -> bool:
    name = (name or "").strip()
    if len(name) < 2 or len(name) > 14:
        return False
    if name in _NON_COMPANY_WORDS:
        return False
    if any(word in name for word in ("多少", "什么", "如何", "请问")):
        return False
    return True


def extract_company_hint(text: str, *, stock_code: str = "") -> str:
    """
    从问题文本提取公司名 hint（不填 stock_code 时用于 L2 跨公司防护）。

    支持语序：
    - 澜起科技2024年EPS
    - 请告诉我2024年澜起科技的每股收益
    """
    _ = stock_code
    from cache.policy import normalize_query

    raw = normalize_query(text or "")
    if not raw:
        return ""

    match = _COMPANY_YEAR_RE.search(raw)
    if match and _is_plausible_company(match.group(1)):
        return match.group(1).strip()

    match = _YEAR_COMPANY_RE.search(raw)
    if match and _is_plausible_company(match.group(1)):
        return match.group(1).strip()

    for match in _COMPANY_SUFFIX_RE.finditer(raw):
        candidate = match.group(1).strip()
        if _is_plausible_company(candidate):
            return candidate

    return ""
