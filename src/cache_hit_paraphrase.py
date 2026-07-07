"""缓存命中测试：paraphrase 模板（与 eval_cache_common 共用逻辑）。"""

from __future__ import annotations

import re

from cache.metadata_extract import extract_company_hint, extract_report_year
from cache.policy import normalize_query

_METRIC_TOKENS = ("EPS", "PE", "营收", "净利润", "毛利率", "市盈率", "每股收益", "营业收入")


def paraphrase_polite_prefix(query: str) -> str:
    base = query.rstrip("？?").strip()
    if base.startswith("请问"):
        return base + "？"
    return f"请问{base}？"


def paraphrase_polite_tell(query: str) -> str:
    """「请告诉我{年}{公司}的{指标}」语序改写。"""
    year = extract_report_year(query)
    company = extract_company_hint(query)
    if not year or not company:
        return paraphrase_polite_prefix(query)

    norm = normalize_query(query)
    metric = ""
    for token in _METRIC_TOKENS:
        if token in norm.upper() if token == "EPS" or token == "PE" else token in norm:
            metric = "每股收益" if token in ("EPS", "每股收益") else token
            if token == "市盈率":
                metric = "市盈率"
            break
    if not metric:
        metric = "相关信息"
    if metric == "EPS":
        metric = "每股收益"
    return f"请告诉我{year}年{company}的{metric}？"


def paraphrase_synonym_metric(query: str) -> str:
    q = query.rstrip("？?").strip()
    upper = q.upper()
    if "EPS" in upper and "每股收益" not in q:
        return re.sub(r"EPS", "每股收益", q, flags=re.IGNORECASE) + "？"
    if "每股收益" in q:
        return q.replace("每股收益", "EPS") + "？"
    if "市盈率" in q and "PE" not in upper:
        return q.replace("市盈率", "PE") + "？"
    if "PE" in upper and "市盈率" not in q:
        return re.sub(r"\bPE\b", "市盈率", q, flags=re.IGNORECASE) + "？"
    if "营业收入" in q:
        return q.replace("营业收入", "营收") + "？"
    if "营收" in q and "营业收入" not in q:
        return q.replace("营收", "营业收入") + "？"
    return paraphrase_polite_prefix(q)


def paraphrase_word_order(query: str) -> str:
    year = extract_report_year(query)
    company = extract_company_hint(query)
    if not year or not company:
        return paraphrase_polite_prefix(query)
    norm = normalize_query(query)
    prefix = f"{company}{year}年"
    if norm.startswith(prefix):
        rest = norm[len(prefix) :]
        return f"{year}年{company}{rest}"
    return paraphrase_polite_prefix(query)


def paraphrase_punctuation(query: str) -> str:
    q = query.rstrip("？?").strip()
    if q.endswith(("是多少", "是什么", "有多少")):
        q = q[:-3] + "是"
    elif q.endswith("吗"):
        q = q[:-1]
    else:
        q = q + "是"
    return q + "？"


def paraphrase_filler_suffix(query: str) -> str:
    return query.rstrip("？?").strip() + "（帮忙查一下）？"


VARIANT_GENERATORS = {
    "polite_prefix": paraphrase_polite_prefix,
    "polite_tell": paraphrase_polite_tell,
    "synonym_metric": paraphrase_synonym_metric,
    "word_order": paraphrase_word_order,
    "punctuation": paraphrase_punctuation,
    "filler_suffix": paraphrase_filler_suffix,
}


def expected_layer(original: str, paraphrase: str) -> str:
    if normalize_query(original) == normalize_query(paraphrase):
        return "l1_exact"
    return "l2_semantic"


def make_paraphrase(query: str, variant_type: str) -> str:
    gen = VARIANT_GENERATORS.get(variant_type)
    if gen is None:
        raise ValueError(f"unknown variant_type: {variant_type}")
    return gen(query)
