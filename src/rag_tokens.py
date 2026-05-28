"""
评测 relevance 与答案事实性校验共用的关键词 / 数字匹配。
"""

from __future__ import annotations

import re

MUST_TOKEN_ALIASES: dict[str, list[str]] = {
    "EPS": ["EPS", "每股收益", "摊薄每股收益", "每股盈利"],
    "PE": ["PE", "市盈率", "P/E"],
    "YoY": ["YoY", "同比"],
    "归母净利润": ["归母净利润", "归母净利", "净利润"],
    "净利润": ["净利润", "归母净利润"],
    "营收": ["营收", "营业收入", "营业总收入"],
    "营业收入": ["营业收入", "营收", "营业总收入"],
    "评级": ["评级", "投资评级"],
    "买入": ["买入", "增持", "推荐", "优于大市"],
    "增持": ["增持", "买入", "推荐"],
    "分红": ["分红", "派息", "股利", "股利支付率"],
    "回购": ["回购"],
    "毛利率": ["毛利率", "销售毛利率"],
    "外销": ["外销", "出口", "海外"],
    "出口": ["出口", "外销", "海外"],
}


def expand_must_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for candidate in MUST_TOKEN_ALIASES.get(token, [token]):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _digits_only(value: str) -> str:
    return re.sub(r"[^\d.]", "", value)


def must_tokens_match(text: str, tokens: list[str]) -> bool:
    """任一 must token 命中即 True；数字 token 允许去单位后匹配。"""
    if not tokens:
        return True
    if not text:
        return False

    candidates = expand_must_tokens(tokens)
    text_digits = _digits_only(text)

    for token in candidates:
        if token in text:
            return True
        if re.search(r"\d", token):
            norm = _digits_only(token)
            if norm and norm in text_digits:
                return True
    return False
