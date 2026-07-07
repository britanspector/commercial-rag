"""缓存写入 / 读取策略（纯函数，无 I/O）。"""

from __future__ import annotations

import re
import unicodedata

from cache.config import CacheSettings, cache_settings
from cache.types import CacheEntry, CacheQueryContext, CacheScope

_WHITESPACE_RE = re.compile(r"\s+")
_POLITE_PREFIX_RE = re.compile(
    r"^(?:请问|请告诉我|请介绍|请说明|请概括|想了解一下|想了解|帮我查一下|帮我|帮忙|能否|能不能|可以|麻烦)"
)
_FILLER_SUFFIX_RE = re.compile(
    r"(?:（帮忙查一下）|（请帮忙）|\(帮忙查一下\)|\(请帮忙\)|帮忙查一下|谢谢|多谢)$"
)
_QUESTION_TAIL_RE = re.compile(
    r"(?:是多少|是什么|有多少|怎么样|如何|吗|呢|呀|啊|么)([？?]*)$"
)
_WEAK_IS_TAIL_RE = re.compile(r"是[？?]*$")
_METRIC_DE_RE = re.compile(
    r"([\u4e00-\u9fff]{2,12})的(EPS|PE|营收|净利润|毛利率)"
)
_CN_NUM_SPACE_RE = re.compile(r"([\u4e00-\u9fff])\s+(\d)|(\d)\s+([\u4e00-\u9fff])")
_TRAILING_PUNCT_RE = re.compile(r"[？?。．.!！,，;；:：]+$")
_YEAR_DE_RE = re.compile(r"(20\d{2})年的")
_COMPANY_YEAR_DE_RE = re.compile(r"年的(?=20\d{2})")
_COMPANY_BEFORE_YEAR_RE = re.compile(
    r"^([\u4e00-\u9fff]{2,12}(?:科技|股份|集团|国际|装备|新能|微|技术|核电|易佰|传媒|信息|材料|电子|软件|网络|数据|智能|光电|半导体)?)"
    r"(20\d{2})年?(.*)$"
)
_YEAR_BEFORE_COMPANY_RE = re.compile(
    r"^(20\d{2})年([\u4e00-\u9fff]{2,12}(?:科技|股份|集团|国际|装备|新能|微|技术|核电|易佰|传媒|信息|材料|电子|软件|网络|数据|智能|光电|半导体)?)(.*)$"
)

# 金融指标同义词 → canonical（L1 key 归一）
_METRIC_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"每股收益"), "EPS"),
    (re.compile(r"市盈率(?!PE)"), "PE"),
    (re.compile(r"营业收入"), "营收"),
    (re.compile(r"归母净利润"), "净利润"),
]


def normalize_query(text: str) -> str:
    """
    L1 精确缓存用的查询规范化。

    去除礼貌用语、统一标点与问句尾、金融同义词归一、弱化「的」位置差异。
    """
    text = unicodedata.normalize("NFKC", (text or "").strip())
    text = _WHITESPACE_RE.sub(" ", text)
    if not text:
        return ""

    text = _POLITE_PREFIX_RE.sub("", text).strip()
    text = _FILLER_SUFFIX_RE.sub("", text).strip()
    text = _QUESTION_TAIL_RE.sub(r"\1", text)
    text = _WEAK_IS_TAIL_RE.sub("", text)
    text = _TRAILING_PUNCT_RE.sub("", text).strip()

    for pattern, canonical in _METRIC_SYNONYMS:
        text = pattern.sub(canonical, text)

    text = _METRIC_DE_RE.sub(r"\1\2", text)
    text = _CN_NUM_SPACE_RE.sub(
        lambda m: (m.group(1) or m.group(3)) + (m.group(2) or m.group(4)),
        text,
    )
    text = _YEAR_DE_RE.sub(r"\1年", text)
    text = _COMPANY_YEAR_DE_RE.sub("年", text)
    text = re.sub(r"的+", "的", text)
    text = text.replace("年的", "年").replace("年 的", "年")
    text = _canonicalize_entity_order(text)

    return _WHITESPACE_RE.sub(" ", text).strip()


def _canonicalize_entity_order(text: str) -> str:
    """统一为「公司+年份+指标」语序，减少 word_order / polite_tell 导致的 L1 miss。"""
    m = _COMPANY_BEFORE_YEAR_RE.match(text)
    if m:
        company, year, rest = m.group(1), m.group(2), m.group(3)
        return f"{company}{year}年{rest}".rstrip("年") if not rest else f"{company}{year}年{rest}"
    m = _YEAR_BEFORE_COMPANY_RE.match(text)
    if m:
        year, company, rest = m.group(1), m.group(2), m.group(3)
        return f"{company}{year}年{rest}"
    return text


def build_config_fingerprint(
    *,
    recall_route: str,
    recall_top_k: int,
    rerank_top_k: int,
    refusal_threshold: float,
    hybrid_vector_weight: float,
    hybrid_pool_size: int,
) -> str:
    return (
        f"route={recall_route}|rtk={recall_top_k}|rrk={rerank_top_k}"
        f"|ref={refusal_threshold:.4f}|hw={hybrid_vector_weight:.4f}|pool={hybrid_pool_size}"
    )


def build_generation_fingerprint(
    *,
    llm_model: str,
    prompt_version: str = "v1",
    num_ctx: int = 8192,
    num_predict: int = 2048,
) -> str:
    return (
        f"model={llm_model}|prompt={prompt_version}"
        f"|ctx={num_ctx}|pred={num_predict}"
    )


def should_cache_result(
    *,
    scope: CacheScope,
    refused: bool,
    query_type: str,
    settings: CacheSettings | None = None,
) -> bool:
    """判断是否允许写入缓存。"""
    cfg = settings or cache_settings
    if not cfg.active:
        return False
    if query_type == "comparative":
        return False
    if refused:
        return cfg.ttl_refused_s > 0
    return True


def should_serve_cached(
    entry: CacheEntry,
    *,
    index_fingerprint: str,
    config_fingerprint: str,
    generation_fingerprint: str = "",
    similarity: float = 1.0,
    settings: CacheSettings | None = None,
) -> tuple[bool, str]:
    """
    命中后安全校验（不含 chunk 存在性 / evidence 复检，由 service 层补充）。

    Returns:
        (ok, reject_reason)
    """
    cfg = settings or cache_settings
    if entry.key.index_fingerprint != index_fingerprint:
        return False, "index_fingerprint_mismatch"
    if entry.key.config_fingerprint != config_fingerprint:
        return False, "config_fingerprint_mismatch"
    if entry.key.scope == CacheScope.CHAT:
        if entry.key.generation_fingerprint != generation_fingerprint:
            return False, "generation_fingerprint_mismatch"
    if not entry.exact_match and similarity < cfg.similarity_threshold:
        return False, "similarity_below_threshold"
    if entry.refused and cfg.ttl_refused_s <= 0:
        return False, "refused_not_served"
    return True, ""


def ttl_for_entry(
    *,
    scope: CacheScope,
    refused: bool,
    query_type: str,
    settings: CacheSettings | None = None,
) -> int:
    cfg = settings or cache_settings
    if refused:
        return cfg.ttl_refused_s
    if scope == CacheScope.SEARCH:
        return cfg.ttl_search_s
    if query_type == "summary":
        return min(cfg.ttl_chat_s, 4 * 3600)
    return cfg.ttl_chat_s


def is_entry_expired(entry: CacheEntry, *, now_ts: float | None = None) -> bool:
    """根据 created_at_iso + ttl_s 判断是否过期。"""
    from datetime import datetime, timezone

    if entry.ttl_s <= 0:
        return True
    try:
        created = datetime.fromisoformat(entry.created_at_iso)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    if now_ts is None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    age_s = (now - created).total_seconds()
    return age_s >= entry.ttl_s


def _company_names_match(req_name: str, ent_name: str) -> bool:
    """公司名模糊匹配：包含关系或归一化后相等。"""
    req_norm = normalize_query(req_name)
    ent_norm = normalize_query(ent_name)
    if not req_norm or not ent_norm:
        return False
    if req_norm == ent_norm:
        return True
    return req_norm in ent_norm or ent_norm in req_norm


def validate_semantic_metadata(
    entry: CacheEntry,
    context: CacheQueryContext,
) -> tuple[bool, str]:
    """
    L2 向量召回后的 metadata 安全校验（公司 / 年份 / 文档 / filter 维度）。

    请求侧指定了 filter 时，entry 侧必须一致，避免金融研报场景错误命中。
    """
    if context.key.stock_code and entry.key.stock_code != context.key.stock_code:
        return False, "stock_code_mismatch"
    if entry.key.query_type != context.key.query_type:
        return False, "query_type_mismatch"

    req = context.metadata_filters
    ent = entry.metadata_filters
    if req is None:
        return True, ""
    if ent is None:
        return False, "metadata_missing"

    req_company = req.company_name.strip()
    ent_company = ent.company_name.strip()
    if req_company and ent_company:
        if not _company_names_match(req_company, ent_company):
            return False, "company_mismatch"

    checks = [
        (req.report_year, ent.report_year, "report_year"),
        (req.doc_id, ent.doc_id, "doc_id"),
        (req.doc_version, ent.doc_version, "doc_version"),
    ]
    for req_val, ent_val, field in checks:
        req_norm = req_val.strip()
        if not req_norm:
            continue
        ent_norm = ent_val.strip()
        if not ent_norm or ent_norm != req_norm:
            return False, f"{field}_mismatch"
    return True, ""
