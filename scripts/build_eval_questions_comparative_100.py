"""
构建 comparative 专项 100 题测试集。

设计原则：
1. 保留现有主评测集中的 26 题 comparative
2. 基于已验证的 factual / summary 题拼装新增 74 题，复用 gold_chunk_ids
3. 避免题目集中在同一行业/同一指标，补充数值、趋势、结构、逻辑、边界类 comparative

用法：
    python scripts/build_eval_questions_comparative_100.py
    python scripts/build_eval_questions_comparative_100.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "data" / "eval" / "eval_questions.jsonl"
OUTPUT_PATH = ROOT / "data" / "eval" / "eval_questions_comparative_100.jsonl"

TARGET_COUNT = 100
KEEP_EXISTING_COUNT = 26


def dedupe_keep_order(values: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return result


def load_source_questions() -> list[dict]:
    questions: list[dict] = []
    with open(SOURCE_PATH, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if len(questions) < 150:
        raise ValueError(f"期望主评测集至少 150 题，实际 {len(questions)}")
    return questions


EXTRA_SPECS: list[dict] = [
    # 半导体 19 题
    {"id": "cq001", "industry": "半导体", "tag": "numeric_revenue", "query": "澜起科技和华峰测控2025年营收规模对比", "gold_answer": "对比两家公司营收规模", "source_ids": ["q58", "q41"]},
    {"id": "cq002", "industry": "半导体", "tag": "numeric_eps", "query": "澜起科技和恒烁股份2026E EPS弹性对比", "gold_answer": "对比两家公司2026E EPS/盈利预测", "source_ids": ["q01", "q59"]},
    {"id": "cq003", "industry": "半导体", "tag": "structure_positioning", "query": "京仪装备和拓荆科技在半导体设备链中的业务定位差异", "gold_answer": "温控设备与薄膜沉积设备定位差异", "source_ids": ["q05", "q145"]},
    {"id": "cq004", "industry": "半导体", "tag": "risk_compare", "query": "国民技术和恒烁股份当前风险暴露点对比", "gold_answer": "对比两家公司风险因素与业务脆弱点", "source_ids": ["q09", "q08"]},
    {"id": "cq005", "industry": "半导体", "tag": "valuation_rating", "query": "华峰测控和芯朋微的投资评级与估值表述差异", "gold_answer": "对比两家公司评级与估值表达", "source_ids": ["q45", "q07"]},
    {"id": "cq006", "industry": "半导体", "tag": "logic_compare", "query": "京仪装备和中芯国际对半导体景气的受益方向有何不同", "gold_answer": "对比设备链和制造链的受益逻辑", "source_ids": ["q05", "q10"]},
    {"id": "cq007", "industry": "半导体", "tag": "numeric_quality", "query": "拓荆科技和卓胜微2025年收入与盈利表现对比", "gold_answer": "对比收入体量与盈利表现", "source_ids": ["q144", "q147"]},
    {"id": "cq008", "industry": "半导体", "tag": "same_company_multi_report", "query": "澜起科技主表与附录对2026E EPS表述是否一致", "gold_answer": "核对主表与附录的EPS预测是否一致", "source_ids": ["q01", "q26"]},
    {"id": "cq009", "industry": "半导体", "tag": "numeric_margin", "query": "华峰测控和京仪装备2026E盈利能力对比", "gold_answer": "对比两家公司毛利率/盈利能力", "source_ids": ["q43", "q06"]},
    {"id": "cq010", "industry": "半导体", "tag": "logic_compare", "query": "国民技术和中芯国际的核心投资逻辑差异", "gold_answer": "对比两家公司投资逻辑与赛道侧重点", "source_ids": ["q09", "q10"]},
    {"id": "cq011", "industry": "半导体", "tag": "industry_view", "query": "芯朋微与恒烁股份赛道景气来源差异", "gold_answer": "对比电源管理与NOR Flash赛道景气来源", "source_ids": ["q07", "q39"]},
    {"id": "cq012", "industry": "半导体", "tag": "logic_compare", "query": "拓荆科技和京仪装备在资本开支扩张中的受益链条对比", "gold_answer": "对比两家公司在设备扩产中的受益逻辑", "source_ids": ["q145", "q05"]},
    {"id": "cq013", "industry": "半导体", "tag": "structure_compare", "query": "澜起科技和中芯国际2025-2026年的经营重心差异", "gold_answer": "对比IC设计与晶圆制造的经营重心", "source_ids": ["q58", "q10"]},
    {"id": "cq014", "industry": "半导体", "tag": "numeric_profit", "query": "华峰测控和卓胜微2025年盈利质量对比", "gold_answer": "对比两家公司利润质量与经营表现", "source_ids": ["q42", "q147"]},
    {"id": "cq015", "industry": "半导体", "tag": "structure_compare", "query": "恒烁股份和澜起科技在存储链条中的定位差异", "gold_answer": "对比NOR Flash与DDR5接口芯片定位", "source_ids": ["q39", "q01"]},
    {"id": "cq016", "industry": "半导体", "tag": "logic_compare", "query": "京仪装备和华峰测控利润率来源差异", "gold_answer": "对比两家公司利润率的形成来源", "source_ids": ["q06", "q43"]},
    {"id": "cq017", "industry": "半导体", "tag": "structure_positioning", "query": "中芯国际和拓荆科技谁更偏制造、谁更偏设备", "gold_answer": "对比两家公司在产业链中的位置", "source_ids": ["q10", "q145"]},
    {"id": "cq018", "industry": "半导体", "tag": "growth_driver", "query": "芯朋微和国民技术业务成长驱动差异", "gold_answer": "对比两家公司成长驱动与业务重点", "source_ids": ["q07", "q09"]},
    {"id": "cq019", "industry": "半导体", "tag": "risk_compare", "query": "卓胜微和恒烁股份2025年业绩压力来源对比", "gold_answer": "对比两家公司业绩压力与风险来源", "source_ids": ["q147", "q08"]},
    # 电力 18 题
    {"id": "cq020", "industry": "电力", "tag": "mixed_financial", "query": "中国核电和长江电力2026E EPS与股息特征对比", "gold_answer": "对比两家公司EPS与股息特征", "source_ids": ["q27", "q17", "q37"]},
    {"id": "cq021", "industry": "电力", "tag": "numeric_generation", "query": "中国核电和三峡能源2025年发电/装机指标对比", "gold_answer": "对比发电量与装机相关指标", "source_ids": ["q38", "q18"]},
    {"id": "cq022", "industry": "电力", "tag": "structure_compare", "query": "嘉泽新能和协鑫能科新能源布局路径差异", "gold_answer": "对比绿电、绿醇和装机布局路径", "source_ids": ["q29", "q34", "q21"]},
    {"id": "cq023", "industry": "电力", "tag": "rating_compare", "query": "通宝能源和福能股份投资评级表达差异", "gold_answer": "对比两家公司评级表述", "source_ids": ["q19", "q33"]},
    {"id": "cq024", "industry": "电力", "tag": "numeric_profit", "query": "中国核电和中国广核2025-2026年盈利对比", "gold_answer": "对比核电龙头的盈利表现", "source_ids": ["q38", "q127", "q128"]},
    {"id": "cq025", "industry": "电力", "tag": "numeric_profit", "query": "华能国际和中国广核2025年盈利规模与经营重点对比", "gold_answer": "对比盈利规模与经营重心", "source_ids": ["q129", "q127", "q130"]},
    {"id": "cq026", "industry": "电力", "tag": "growth_driver", "query": "南网储能和嘉泽新能增长驱动差异", "gold_answer": "对比储能与绿醇/绿电驱动差异", "source_ids": ["q131", "q29"]},
    {"id": "cq027", "industry": "电力", "tag": "structure_compare", "query": "三峡能源和协鑫能科装机扩张路径对比", "gold_answer": "对比两家公司扩张路径与装机逻辑", "source_ids": ["q18", "q34"]},
    {"id": "cq028", "industry": "电力", "tag": "mixed_financial", "query": "长江电力和通宝能源股息与评级特征对比", "gold_answer": "对比高股息属性与评级表达", "source_ids": ["q37", "q19"]},
    {"id": "cq029", "industry": "电力", "tag": "numeric_profit", "query": "福能股份和嘉泽新能2026年利润弹性对比", "gold_answer": "对比两家公司利润预测与弹性", "source_ids": ["q20", "q12"]},
    {"id": "cq030", "industry": "电力", "tag": "same_company_multi_report", "query": "中国核电不同研报对2026E EPS口径是否一致", "gold_answer": "核对中国核电不同研报的EPS口径", "source_ids": ["q11", "q27"]},
    {"id": "cq031", "industry": "电力", "tag": "logic_compare", "query": "华能国际和长江电力高股息与火电逻辑差异", "gold_answer": "对比高股息水电与火电盈利逻辑", "source_ids": ["q130", "q37"]},
    {"id": "cq032", "industry": "电力", "tag": "numeric_profit", "query": "中国广核和南网储能盈利增速对比", "gold_answer": "对比两家公司盈利增速与质量", "source_ids": ["q127", "q131"]},
    {"id": "cq033", "industry": "电力", "tag": "structure_compare", "query": "三峡能源和中国核电新能源/核电经营模式差异", "gold_answer": "对比新能源与核电经营模式", "source_ids": ["q18", "q38"]},
    {"id": "cq034", "industry": "电力", "tag": "logic_compare", "query": "嘉泽新能和福能股份盈利预测口径差异", "gold_answer": "对比两家公司盈利预测侧重点", "source_ids": ["q12", "q20"]},
    {"id": "cq035", "industry": "电力", "tag": "logic_compare", "query": "通宝能源和华能国际投资属性对比", "gold_answer": "对比区域火电资产与央企火电逻辑", "source_ids": ["q19", "q130"]},
    {"id": "cq036", "industry": "电力", "tag": "growth_driver", "query": "南网储能和中国核电业务稳定性与成长性对比", "gold_answer": "对比两家公司稳定性与成长来源", "source_ids": ["q131", "q38"]},
    {"id": "cq037", "industry": "电力", "tag": "structure_compare", "query": "中国广核和中国核电发电与盈利结构差异", "gold_answer": "对比两家核电公司发电/盈利结构", "source_ids": ["q127", "q38"]},
    # 互联网电商 18 题
    {"id": "cq038", "industry": "互联网电商", "tag": "numeric_profit", "query": "华鼎股份和吉宏股份2025年上半年利润规模对比", "gold_answer": "对比两家公司利润规模", "source_ids": ["q135", "q136"]},
    {"id": "cq039", "industry": "互联网电商", "tag": "numeric_revenue", "query": "华鼎股份和华凯易佰收入体量与盈利质量对比", "gold_answer": "对比两家公司收入与利润质量", "source_ids": ["q134", "q28"]},
    {"id": "cq040", "industry": "互联网电商", "tag": "growth_driver", "query": "吉宏股份和若羽臣增长驱动差异", "gold_answer": "对比AI跨境营销与品牌孵化驱动", "source_ids": ["q137", "q40"]},
    {"id": "cq041", "industry": "互联网电商", "tag": "structure_compare", "query": "焦点科技和青木科技盈利预测与业务模式差异", "gold_answer": "对比B2B平台与代运营模式", "source_ids": ["q14", "q35", "q22"]},
    {"id": "cq042", "industry": "互联网电商", "tag": "numeric_growth", "query": "赛维时代和华凯易佰2025年营收增速对比", "gold_answer": "对比两家公司营收增速", "source_ids": ["q16", "q13"]},
    {"id": "cq043", "industry": "互联网电商", "tag": "structure_compare", "query": "若羽臣和焦点科技品牌孵化与B2B平台模式差异", "gold_answer": "对比两家公司业务模式差异", "source_ids": ["q15", "q31"]},
    {"id": "cq044", "industry": "互联网电商", "tag": "logic_compare", "query": "吉宏股份和焦点科技AI/数字化能力对比", "gold_answer": "对比两家公司数字化和AI能力", "source_ids": ["q137", "q31"]},
    {"id": "cq045", "industry": "互联网电商", "tag": "structure_compare", "query": "青木科技和若羽臣代运营与品牌孵化差异", "gold_answer": "对比代运营与品牌孵化路径", "source_ids": ["q22", "q15"]},
    {"id": "cq046", "industry": "互联网电商", "tag": "numeric_quality", "query": "华鼎股份和吉宏股份盈利增长质量对比", "gold_answer": "对比利润规模与收入增速", "source_ids": ["q135", "q138"]},
    {"id": "cq047", "industry": "互联网电商", "tag": "numeric_growth", "query": "华凯易佰和赛维时代业绩弹性对比", "gold_answer": "对比两家公司收入/利润弹性", "source_ids": ["q28", "q16"]},
    {"id": "cq048", "industry": "互联网电商", "tag": "numeric_profit", "query": "焦点科技和吉宏股份盈利预测口径对比", "gold_answer": "对比两家公司盈利预测/利润表现", "source_ids": ["q14", "q136"]},
    {"id": "cq049", "industry": "互联网电商", "tag": "structure_compare", "query": "若羽臣和青木科技谁更偏品牌、谁更偏代运营", "gold_answer": "对比两家公司业务定位", "source_ids": ["q15", "q22"]},
    {"id": "cq050", "industry": "互联网电商", "tag": "growth_driver", "query": "华鼎股份和吉宏股份增长逻辑差异", "gold_answer": "对比跨境业务与AI驱动差异", "source_ids": ["q134", "q137"]},
    {"id": "cq051", "industry": "互联网电商", "tag": "logic_compare", "query": "焦点科技和若羽臣业绩确定性来源对比", "gold_answer": "对比会员平台与品牌孵化的确定性来源", "source_ids": ["q31", "q40"]},
    {"id": "cq052", "industry": "互联网电商", "tag": "growth_driver", "query": "赛维时代和吉宏股份跨境增长驱动差异", "gold_answer": "对比两家公司跨境增长驱动", "source_ids": ["q16", "q137"]},
    {"id": "cq053", "industry": "互联网电商", "tag": "structure_compare", "query": "华凯易佰和焦点科技商业模式差异", "gold_answer": "对比跨境卖家与B2B平台模式", "source_ids": ["q13", "q31"]},
    {"id": "cq054", "industry": "互联网电商", "tag": "numeric_profit", "query": "青木科技和吉宏股份2026E利润弹性对比", "gold_answer": "对比两家公司利润弹性", "source_ids": ["q35", "q136"]},
    {"id": "cq055", "industry": "互联网电商", "tag": "numeric_profit", "query": "若羽臣和华鼎股份2025-2026年盈利表现对比", "gold_answer": "对比两家公司盈利表现", "source_ids": ["q40", "q135"]},
    # 白色家电 19 题
    {"id": "cq056", "industry": "白色家电", "tag": "numeric_scale", "query": "美的集团和海尔智家2025年营收与利润体量对比", "gold_answer": "对比营收规模与利润规模", "source_ids": ["q91", "q105", "q106"]},
    {"id": "cq057", "industry": "白色家电", "tag": "numeric_growth", "query": "美的集团和格力电器2025年上半年营收增速与利润规模对比", "gold_answer": "对比两家公司营收增速和利润规模", "source_ids": ["q91", "q99", "q100"]},
    {"id": "cq058", "industry": "白色家电", "tag": "mixed_financial", "query": "海信家电和TCL智家2025年利润规模与股东回报对比", "gold_answer": "对比利润规模与股东回报", "source_ids": ["q112", "q116"]},
    {"id": "cq059", "industry": "白色家电", "tag": "structure_compare", "query": "海尔智家和美的集团高端品牌升级路径对比", "gold_answer": "对比卡萨帝与COLMO/东芝的高端化路径", "source_ids": ["q108", "q96"]},
    {"id": "cq060", "industry": "白色家电", "tag": "logic_compare", "query": "格力电器和海尔智家盈利能力与全球化对比", "gold_answer": "对比盈利能力与全球化布局", "source_ids": ["q100", "q109"]},
    {"id": "cq061", "industry": "白色家电", "tag": "logic_compare", "query": "美的集团和海信家电2025年经营韧性对比", "gold_answer": "对比两家公司经营韧性和业绩表现", "source_ids": ["q94", "q112"]},
    {"id": "cq062", "industry": "白色家电", "tag": "numeric_profit", "query": "TCL智家和长虹美菱2025年利润变化对比", "gold_answer": "对比两家公司利润变化和同比趋势", "source_ids": ["q115", "q118"]},
    {"id": "cq063", "industry": "白色家电", "tag": "numeric_forecast", "query": "海尔智家和格力电器2025-2027年盈利预测对比", "gold_answer": "对比两家公司盈利预测", "source_ids": ["q110", "q101"]},
    {"id": "cq064", "industry": "白色家电", "tag": "growth_driver", "query": "美的集团和海尔智家高端化与海外扩张路径对比", "gold_answer": "对比高端品牌与海外扩张逻辑", "source_ids": ["q96", "q109"]},
    {"id": "cq065", "industry": "白色家电", "tag": "numeric_profit", "query": "海信家电和格力电器盈利趋势差异", "gold_answer": "对比两家公司盈利趋势", "source_ids": ["q112", "q100"]},
    {"id": "cq066", "industry": "白色家电", "tag": "mixed_financial", "query": "美的集团与格力电器投资回报与分红策略对比", "gold_answer": "对比分红和股东回报策略", "source_ids": ["q95", "q102"]},
    {"id": "cq067", "industry": "白色家电", "tag": "structure_compare", "query": "海尔智家和海信家电盈利规模与品牌侧重点对比", "gold_answer": "对比利润规模与品牌/品类侧重点", "source_ids": ["q106", "q111"]},
    {"id": "cq068", "industry": "白色家电", "tag": "numeric_scale", "query": "TCL智家和美的集团营收体量与利润水平差距", "gold_answer": "对比两家公司规模差距", "source_ids": ["q94", "q115"]},
    {"id": "cq069", "industry": "白色家电", "tag": "logic_compare", "query": "格力电器和长虹美菱2025年业绩变化原因对比", "gold_answer": "对比两家公司业绩变化原因", "source_ids": ["q103", "q118"]},
    {"id": "cq070", "industry": "白色家电", "tag": "structure_compare", "query": "海尔智家和格力电器高端化与外销表现对比", "gold_answer": "对比高端化和外销表现", "source_ids": ["q108", "q103"]},
    {"id": "cq071", "industry": "白色家电", "tag": "same_company_multi_report", "query": "美的集团中报与年报对盈利表现表述是否一致", "gold_answer": "核对美的集团中报与年报核心盈利表述", "source_ids": ["q91", "q94"]},
    {"id": "cq072", "industry": "白色家电", "tag": "structure_compare", "query": "海信家电和海尔智家高端化与海外发展路径对比", "gold_answer": "对比两家公司高端化与海外扩张路径", "source_ids": ["q111", "q109"]},
    {"id": "cq073", "industry": "白色家电", "tag": "mixed_financial", "query": "TCL智家和美的集团分红回报对比", "gold_answer": "对比两家公司股东回报策略", "source_ids": ["q116", "q95"]},
    {"id": "cq074", "industry": "白色家电", "tag": "numeric_forecast", "query": "海尔智家和美的集团2025-2027年盈利确定性对比", "gold_answer": "对比两家公司盈利预测确定性", "source_ids": ["q110", "q92"]},
]


def build_item(spec: dict, source_map: dict[str, dict]) -> dict:
    sources = [source_map[qid] for qid in spec["source_ids"]]
    first = sources[0]

    section_keywords = dedupe_keep_order(
        [kw for src in sources for kw in src.get("section_keywords") or []],
        limit=8,
    )
    must_contain_any = dedupe_keep_order(
        [kw for src in sources for kw in src.get("must_contain_any") or []],
        limit=8,
    )
    gold_chunk_ids = dedupe_keep_order(
        [cid for src in sources for cid in src.get("gold_chunk_ids") or []],
        limit=8,
    )
    negative_stock_codes = dedupe_keep_order(
        [code for src in sources for code in src.get("negative_stock_codes") or []],
        limit=6,
    )

    return {
        "id": spec["id"],
        "query_type": "comparative",
        "category": "compare",
        "query": spec["query"],
        "gold_answer": spec["gold_answer"],
        "stock_code": first.get("stock_code", ""),
        "doc_id": first.get("doc_id", ""),
        "industry_label": spec["industry"],
        "section_keywords": section_keywords,
        "must_contain_any": must_contain_any,
        "gold_chunk_ids": gold_chunk_ids,
        "negative_stock_codes": negative_stock_codes,
        "compare_tag": spec["tag"],
        "source_question_ids": list(spec["source_ids"]),
    }


def validate_questions(questions: list[dict]) -> None:
    if len(questions) != TARGET_COUNT:
        raise ValueError(f"期望 {TARGET_COUNT} 题，实际 {len(questions)}")

    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("存在重复题目 id")

    for item in questions:
        if item.get("query_type") != "comparative":
            raise ValueError(f"{item['id']} query_type 不是 comparative")
        if not item.get("industry_label"):
            raise ValueError(f"{item['id']} 缺少 industry_label")
        if not item.get("must_contain_any"):
            raise ValueError(f"{item['id']} 缺少 must_contain_any")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_questions = load_source_questions()
    source_map = {item["id"]: item for item in source_questions}

    base_comparative = [item for item in source_questions if item.get("query_type") == "comparative"]
    if len(base_comparative) != KEEP_EXISTING_COUNT:
        raise ValueError(f"期望保留 {KEEP_EXISTING_COUNT} 题现有 comparative，实际 {len(base_comparative)}")

    missing_source_ids = sorted(
        {
            qid
            for spec in EXTRA_SPECS
            for qid in spec["source_ids"]
            if qid not in source_map
        }
    )
    if missing_source_ids:
        raise ValueError(f"以下 source ids 不存在：{missing_source_ids}")

    extra_questions = [build_item(spec, source_map) for spec in EXTRA_SPECS]
    merged = base_comparative + extra_questions
    validate_questions(merged)

    industry_counts = Counter(item["industry_label"] for item in merged)
    tag_counts = Counter(item.get("compare_tag", "legacy_existing") for item in merged)
    source_size_counts = Counter(len(item.get("source_question_ids") or []) for item in merged if item["id"].startswith("cq"))

    if args.dry_run:
        print(f"[dry-run] comparative 题数：{len(merged)}")
        print("行业分布：", dict(sorted(industry_counts.items())))
        print("新增题 tag 分布：", dict(sorted(tag_counts.items())))
        print("新增题 source 数分布：", dict(sorted(source_size_counts.items())))
        print(f"输出路径：{OUTPUT_PATH}")
        return

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for item in merged:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已写入 {OUTPUT_PATH}，共 {len(merged)} 题")
    print("行业分布：", dict(sorted(industry_counts.items())))
    print("新增题 tag 分布：", dict(sorted(tag_counts.items())))


if __name__ == "__main__":
    main()
