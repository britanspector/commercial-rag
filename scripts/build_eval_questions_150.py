"""
将评测集扩展至 150 题，覆盖四行业（半导体/电力/互联网电商/白色家电）研报内容。

设计原则见 docs/eval-scheme.md「150 题评测集设计」。

用法：
    python scripts/build_eval_questions_150.py
    python scripts/build_eval_questions_150.py --dry-run   # 仅校验，不写文件
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "eval" / "eval_questions.jsonl"
BACKUP_PATH = ROOT / "data" / "eval" / "eval_questions_90.jsonl"
OUTPUT_PATH = ROOT / "data" / "eval" / "eval_questions.jsonl"
CHUNKS_PATH = ROOT / "data" / "parsed" / "chunks.jsonl"

TARGET_COUNT = 150

# 历史题中指向不可检索 chunk 的修正（200 报告重新分块后）
LEGACY_PATCHES: dict[str, dict] = {
    "q39": {
        "gold_chunk_ids": [
            "H3_AP202605231822827683_1_0011",
            "H3_AP202605231822827683_1_0010",
        ],
    },
}


def load_chunk_index() -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if not chunk.get("is_retrievable", True):
                continue
            by_doc.setdefault(chunk["doc_id"], []).append(chunk)
    return by_doc


def find_gold_chunks(
    doc_id: str,
    keywords: list[str],
    *,
    by_doc: dict[str, list[dict]],
    limit: int = 3,
) -> list[str]:
    if not doc_id or not keywords:
        return []
    scored: list[tuple[int, str]] = []
    for chunk in by_doc.get(doc_id, []):
        text = chunk.get("text", "")
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            scored.append((score, chunk["chunk_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _, chunk_id in scored[:limit]]


def q(
    qid: str,
    query: str,
    gold_answer: str,
    *,
    query_type: str = "factual",
    category: str = "financial",
    stock_code: str = "",
    doc_id: str = "",
    industry_label: str = "",
    section_keywords: list[str] | None = None,
    must_contain_any: list[str] | None = None,
    gold_chunk_ids: list[str] | None = None,
    negative_stock_codes: list[str] | None = None,
) -> dict:
    return {
        "id": qid,
        "query_type": query_type,
        "category": category,
        "query": query,
        "gold_answer": gold_answer,
        "stock_code": stock_code,
        "doc_id": doc_id,
        "industry_label": industry_label,
        "section_keywords": section_keywords or [],
        "must_contain_any": must_contain_any or [],
        "gold_chunk_ids": gold_chunk_ids or [],
        "negative_stock_codes": negative_stock_codes or [],
    }


# 新增 60 题（q91–q150）：补全白色家电，并加深电力/电商/半导体覆盖
NEW_QUESTIONS: list[dict] = [
    # ── 白色家电 factual q91–q118 ──
    q(
        "q91",
        "美的集团2025年上半年归母净利润是多少？",
        "2025H1归母净利润约260.14亿元，同比+25%",
        stock_code="000333",
        doc_id="H3_AP202509031738338909_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "半年报", "2025H1"],
        must_contain_any=["260", "归母净利润"],
        gold_chunk_ids=["H3_AP202509031738338909_1_0009"],
    ),
    q(
        "q92",
        "美的集团2025-2027年EPS盈利预测",
        "2025-2027年EPS约6.09/6.86/7.73元",
        stock_code="000333",
        doc_id="H3_AP202509031738338909_1",
        industry_label="白色家电",
        section_keywords=["EPS", "盈利预测", "财务"],
        must_contain_any=["EPS", "6.09"],
        gold_chunk_ids=["H3_AP202509031738338909_1_0015"],
    ),
    q(
        "q93",
        "美的集团2025年新能源与工业技术业务收入增速",
        "2025H1新能源与工业技术收入约219.6亿元，同比+28.6%",
        stock_code="000333",
        doc_id="H3_AP202509051739268593_1",
        industry_label="白色家电",
        category="text",
        section_keywords=["新能源", "工业技术", "收入"],
        must_contain_any=["新能源", "28.6", "工业技术"],
        gold_chunk_ids=["H3_AP202509051739268593_1_0002"],
    ),
    q(
        "q94",
        "美的集团2025年全年营业收入",
        "2025年营收约4585亿元，同比+12.1%",
        stock_code="000333",
        doc_id="H3_AP202604081821065709_1",
        industry_label="白色家电",
        section_keywords=["营收", "2025年", "年报"],
        must_contain_any=["4585", "营收"],
        gold_chunk_ids=["H3_AP202604081821065709_1_0001"],
    ),
    q(
        "q95",
        "美的集团2025年合计股利支付率",
        "中期+年度分红，合计股利支付率约73.6%",
        stock_code="000333",
        doc_id="H3_AP202604081821065709_1",
        industry_label="白色家电",
        category="financial",
        section_keywords=["股利支付率", "分红", "回购"],
        must_contain_any=["73.6", "股利", "分红"],
        gold_chunk_ids=["H3_AP202604081821065709_1_0001"],
    ),
    q(
        "q96",
        "美的集团COLMO与东芝高端品牌零售额增速",
        "COLMO+东芝双高端品牌零售额同比增超60%",
        stock_code="000333",
        doc_id="H3_AP202509051739268593_1",
        industry_label="白色家电",
        category="text",
        section_keywords=["COLMO", "东芝", "高端"],
        must_contain_any=["COLMO", "东芝", "60%"],
    ),
    q(
        "q97",
        "美的集团2025年三季报归母净利润",
        "25Q1-Q3归母净利润约378.8亿元，同比+19.5%",
        stock_code="000333",
        doc_id="H3_AP202510311772362212_1",
        industry_label="白色家电",
        section_keywords=["三季报", "归母净利润"],
        must_contain_any=["378", "归母净利润"],
    ),
    q(
        "q98",
        "美的集团智能建筑科技业务收入规模",
        "2025H1智能建筑科技收入约195.1亿元",
        stock_code="000333",
        doc_id="H3_AP202509051739268593_1",
        industry_label="白色家电",
        category="text",
        section_keywords=["智能建筑", "收入"],
        must_contain_any=["智能建筑", "195"],
    ),
    q(
        "q99",
        "格力电器2025年上半年营业收入",
        "2025H1营收约976.2亿元，同比-2.7%",
        stock_code="000651",
        doc_id="H3_AP202509031738366644_1",
        industry_label="白色家电",
        section_keywords=["营收", "2025H1", "半年报"],
        must_contain_any=["976", "营收"],
        gold_chunk_ids=["H3_AP202509031738366644_1_0002"],
    ),
    q(
        "q100",
        "格力电器2025年上半年归母净利润",
        "2025H1归母净利润约144.1亿元，同比+2.0%",
        stock_code="000651",
        doc_id="H3_AP202509031738366644_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "半年报"],
        must_contain_any=["144.1", "归母净利润"],
        gold_chunk_ids=["H3_AP202509031738366644_1_0002"],
    ),
    q(
        "q101",
        "格力电器2025-2027年归母净利润盈利预测",
        "预计2025-2027年归母净利润约335/352/368亿元",
        stock_code="000651",
        doc_id="H3_AP202509031738366644_1",
        industry_label="白色家电",
        section_keywords=["盈利预测", "归母净利润"],
        must_contain_any=["335", "归母净利润", "预测"],
        gold_chunk_ids=["H3_AP202509031738366644_1_0004"],
    ),
    q(
        "q102",
        "格力电器投资评级结论",
        "维持优于大市（买入）评级",
        stock_code="000651",
        doc_id="H3_AP202509031738366644_1",
        industry_label="白色家电",
        category="rating",
        section_keywords=["优于大市", "评级", "投资建议"],
        must_contain_any=["优于大市", "评级", "买入"],
        gold_chunk_ids=["H3_AP202509031738366644_1_0004"],
    ),
    q(
        "q103",
        "格力电器2025年上半年空调外销收入增速",
        "H1外销收入同比约+10.2%",
        stock_code="000651",
        doc_id="H3_AP202509031738366644_1",
        industry_label="白色家电",
        category="text",
        section_keywords=["外销", "空调", "出口"],
        must_contain_any=["外销", "10.2", "出口"],
        gold_chunk_ids=["H3_AP202509031738366644_1_0002"],
    ),
    q(
        "q104",
        "格力电器2025年三季报归母净利润",
        "25Q3归母净利润约53.4亿元",
        stock_code="000651",
        doc_id="H3_AP202510311772394150_1",
        industry_label="白色家电",
        section_keywords=["三季报", "归母净利润"],
        must_contain_any=["53.4", "归母净利润"],
    ),
    q(
        "q105",
        "海尔智家2025年上半年营业收入",
        "2025H1营收约1564.9亿元，同比+10.2%",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        section_keywords=["营收", "半年报"],
        must_contain_any=["1564", "营收"],
        gold_chunk_ids=["H3_AP202509031738372559_1_0001"],
    ),
    q(
        "q106",
        "海尔智家2025年上半年归母净利润",
        "2025H1归母净利润约120.3亿元，同比+15.6%",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "半年报"],
        must_contain_any=["120.3", "归母净利润"],
        gold_chunk_ids=["H3_AP202509031738372559_1_0001"],
    ),
    q(
        "q107",
        "海尔智家2025年中期现金分红方案",
        "拟每10股派发现金2.69元，现金分红率约20.8%",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        category="financial",
        section_keywords=["分红", "派息", "中期"],
        must_contain_any=["2.69", "分红"],
        gold_chunk_ids=["H3_AP202509031738372559_1_0001"],
    ),
    q(
        "q108",
        "海尔智家卡萨帝品牌2025年上半年收入增速",
        "H1卡萨帝品牌收入增幅超20%",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        category="text",
        section_keywords=["卡萨帝", "高端", "品牌"],
        must_contain_any=["卡萨帝", "20%"],
        gold_chunk_ids=["H3_AP202509031738372559_1_0002"],
    ),
    q(
        "q109",
        "海尔智家2025年欧洲市场收入增速",
        "H1欧洲收入同比约+24.1%",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        category="text",
        section_keywords=["欧洲", "海外", "收入"],
        must_contain_any=["欧洲", "24.1"],
        gold_chunk_ids=["H3_AP202509031738372559_1_0002"],
    ),
    q(
        "q110",
        "海尔智家2025-2027年归母净利润预测",
        "预计2025-2027年归母利润约212/234/257亿元",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        section_keywords=["盈利预测", "归母"],
        must_contain_any=["212", "归母", "预测"],
        gold_chunk_ids=["H3_AP202509031738372559_1_0004"],
    ),
    q(
        "q111",
        "海信家电2025年全年营业收入",
        "2025年营收约879.3亿元，同比-5.2%",
        stock_code="000921",
        doc_id="H3_AP202604011820946598_1",
        industry_label="白色家电",
        section_keywords=["营收", "年报", "2025"],
        must_contain_any=["879", "营收"],
        gold_chunk_ids=["H3_AP202604011820946598_1_0011"],
    ),
    q(
        "q112",
        "海信家电2025年归母净利润",
        "2025年归母净利润约31.9亿元",
        stock_code="000921",
        doc_id="H3_AP202604011820946598_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "2025"],
        must_contain_any=["31.9", "归母净利润"],
        gold_chunk_ids=["H3_AP202604011820946598_1_0011"],
    ),
    q(
        "q113",
        "海信家电厨电业务扩张情况",
        "厨电业务规模高速扩张（2025Q3点评）",
        stock_code="000921",
        doc_id="H3_AP202510311772354118_1",
        industry_label="白色家电",
        category="text",
        section_keywords=["厨电", "扩张", "业务"],
        must_contain_any=["厨电"],
    ),
    q(
        "q114",
        "TCL智家2025年全年营业收入",
        "2025年营收约185.3亿元，同比+0.9%",
        stock_code="002668",
        doc_id="H3_AP202603121820519602_1",
        industry_label="白色家电",
        section_keywords=["营收", "2025", "年报"],
        must_contain_any=["185.3", "营收"],
        gold_chunk_ids=["H3_AP202603121820519602_1_0001"],
    ),
    q(
        "q115",
        "TCL智家2025年归母净利润",
        "2025年归母净利润约11.2亿元，同比+10.2%",
        stock_code="002668",
        doc_id="H3_AP202603121820519602_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "2025"],
        must_contain_any=["11.2", "归母净利润"],
        gold_chunk_ids=["H3_AP202603121820519602_1_0001"],
    ),
    q(
        "q116",
        "TCL智家2025年是否启动分红回报股东",
        "2025年报开启分红回报股东",
        stock_code="002668",
        doc_id="H3_AP202603121820519602_1",
        industry_label="白色家电",
        category="financial",
        section_keywords=["分红", "股东回报"],
        must_contain_any=["分红"],
        gold_chunk_ids=["H3_AP202603121820519602_1_0001"],
    ),
    q(
        "q117",
        "长虹美菱2025年营业收入",
        "2025年营收约304.1亿元，同比+6.3%",
        stock_code="000521",
        doc_id="H3_AP202604091821080729_1",
        industry_label="白色家电",
        section_keywords=["营收", "2025"],
        must_contain_any=["304", "营收"],
        gold_chunk_ids=["H3_AP202604091821080729_1_0012"],
    ),
    q(
        "q118",
        "长虹美菱2025年归母净利润及同比",
        "2025年归母净利润约4.1亿元，同比-41.3%",
        stock_code="000521",
        doc_id="H3_AP202604091821080729_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "同比"],
        must_contain_any=["4.1", "归母净利润"],
        gold_chunk_ids=["H3_AP202604091821080729_1_0012"],
    ),
    # ── 白色家电 comparative + summary q119–q126 ──
    q(
        "q119",
        query_type="comparative",
        category="compare",
        query="美的集团和格力电器2025年上半年归母净利润谁更高？",
        gold_answer="美的约260亿 vs 格力约144亿",
        stock_code="000333",
        doc_id="H3_AP202509031738338909_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "半年报"],
        must_contain_any=["归母净利润"],
        gold_chunk_ids=["H3_AP202509031738338909_1_0009"],
    ),
    q(
        "q120",
        query_type="comparative",
        category="compare",
        query="海尔智家和美的集团2025年上半年营收增速对比",
        gold_answer="海尔+10.2% vs 美的+15.7%",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        section_keywords=["营收", "同比", "增速"],
        must_contain_any=["营收", "同比"],
        gold_chunk_ids=["H3_AP202509031738372559_1_0001"],
    ),
    q(
        "q121",
        query_type="comparative",
        category="compare",
        query="海信家电和TCL智家2025年归母净利润规模对比",
        gold_answer="海信约31.9亿 vs TCL智家约11.2亿",
        stock_code="000921",
        doc_id="H3_AP202604011820946598_1",
        industry_label="白色家电",
        section_keywords=["归母净利润"],
        must_contain_any=["归母净利润"],
        gold_chunk_ids=["H3_AP202604011820946598_1_0011"],
    ),
    q(
        "q122",
        query_type="comparative",
        category="compare",
        query="美的集团与海尔智家海外收入增速哪个更快？",
        gold_answer="对比两公司海外/欧洲等区域增速",
        stock_code="000333",
        doc_id="H3_AP202509051739268593_1",
        industry_label="白色家电",
        section_keywords=["海外", "收入", "同比"],
        must_contain_any=["海外", "收入"],
    ),
    q(
        "q123",
        query_type="comparative",
        category="compare",
        query="格力电器和长虹美菱2025年盈利能力变化对比",
        gold_answer="格力盈利稳健 vs 美菱利润下滑",
        stock_code="000651",
        doc_id="H3_AP202509031738366644_1",
        industry_label="白色家电",
        section_keywords=["归母净利润", "同比", "净利率"],
        must_contain_any=["归母净利润", "同比"],
    ),
    q(
        "q124",
        query_type="summary",
        category="summary",
        query="白色家电龙头2025年高端化与品牌升级策略",
        gold_answer="卡萨帝/COLMO/东芝等高端品牌论述",
        stock_code="600690",
        doc_id="H3_AP202509031738372559_1",
        industry_label="白色家电",
        section_keywords=["高端", "品牌", "卡萨帝", "COLMO"],
        must_contain_any=["高端", "品牌"],
    ),
    q(
        "q125",
        query_type="summary",
        category="summary",
        query="白色家电行业2025年出口与全球化布局概况",
        gold_answer="美的/海尔/格力等海外收入与全球化",
        stock_code="000333",
        doc_id="H3_AP202509051739268593_1",
        industry_label="白色家电",
        section_keywords=["海外", "全球化", "出口"],
        must_contain_any=["海外", "全球化"],
    ),
    q(
        "q126",
        query_type="summary",
        category="summary",
        query="白色家电板块高分红与股东回报政策梳理",
        gold_answer="美的/格力/海尔分红回购论述",
        stock_code="000333",
        doc_id="H3_AP202604081821065709_1",
        industry_label="白色家电",
        section_keywords=["分红", "回购", "股利"],
        must_contain_any=["分红", "回购"],
        gold_chunk_ids=["H3_AP202604081821065709_1_0001"],
    ),
    # ── 电力 q127–q133 ──
    q(
        "q127",
        "中国广核2025年归母净利润",
        "2025年归母净利润约97.65亿元",
        stock_code="003816",
        doc_id="H3_AP202604011820946602_1",
        industry_label="电力",
        section_keywords=["归母净利润", "2025"],
        must_contain_any=["97.65", "归母净利润"],
        gold_chunk_ids=["H3_AP202604011820946602_1_0003"],
    ),
    q(
        "q128",
        "中国广核2026E EPS预测",
        "盈利预测表2026E EPS",
        stock_code="003816",
        doc_id="H3_AP202604011820946602_1",
        industry_label="电力",
        section_keywords=["EPS", "盈利预测", "2026E"],
        must_contain_any=["EPS", "每股收益"],
        gold_chunk_ids=["H3_AP202604011820946602_1_0004"],
    ),
    q(
        "q129",
        "华能国际2025年营业收入与归母净利润",
        "2025年营收约2292.9亿元，归母净利润约144.1亿元",
        stock_code="600011",
        doc_id="H3_AP202604021820978919_1",
        industry_label="电力",
        section_keywords=["营收", "归母净利润", "年报"],
        must_contain_any=["2292", "144.1"],
        gold_chunk_ids=["H3_AP202604021820978919_1_0002"],
    ),
    q(
        "q130",
        "华能国际火电盈利与股息配置价值",
        "火电盈利提升，重视高股息配置",
        stock_code="600011",
        doc_id="H3_AP202604021820978919_1",
        industry_label="电力",
        category="text",
        section_keywords=["火电", "股息", "盈利"],
        must_contain_any=["股息", "火电"],
        gold_chunk_ids=["H3_AP202604021820978919_1_0002"],
    ),
    q(
        "q131",
        "南网储能2025年归母净利润及同比增速",
        "2025年归母净利润约16.89亿元，同比+49.89%",
        stock_code="600995",
        doc_id="H3_AP202604011820956516_1",
        industry_label="电力",
        section_keywords=["归母净利润", "同比"],
        must_contain_any=["16.89", "归母净利润"],
        gold_chunk_ids=["H3_AP202604011820956516_1_0005"],
    ),
    q(
        "q132",
        query_type="comparative",
        category="compare",
        query="中国广核和华能国际2025年归母净利润规模对比",
        gold_answer="广核约97.7亿 vs 华能约144.1亿",
        stock_code="003816",
        doc_id="H3_AP202604011820946602_1",
        industry_label="电力",
        section_keywords=["归母净利润"],
        must_contain_any=["归母净利润"],
        gold_chunk_ids=["H3_AP202604011820946602_1_0003"],
    ),
    q(
        "q133",
        query_type="summary",
        category="summary",
        query="电力行业抽水蓄能与新型储能协同发展",
        gold_answer="南网储能等抽蓄与储能业务",
        stock_code="600995",
        doc_id="H3_AP202604011820956516_1",
        industry_label="电力",
        section_keywords=["抽水蓄能", "储能", "调峰"],
        must_contain_any=["抽水蓄能", "储能"],
        gold_chunk_ids=["H3_AP202604011820956516_1_0006"],
    ),
    # ── 互联网电商 q134–q143 ──
    q(
        "q134",
        "华鼎股份2025年上半年营业收入",
        "2025H1营收约24.11亿元",
        stock_code="601113",
        doc_id="H3_AP202508201730938917_1",
        industry_label="互联网电商",
        section_keywords=["营收", "半年报"],
        must_contain_any=["24.11", "营收"],
        gold_chunk_ids=["H3_AP202508201730938917_1_0001"],
    ),
    q(
        "q135",
        "华鼎股份2025年上半年归母净利润",
        "2025H1归母净利润约1.53亿元",
        stock_code="601113",
        doc_id="H3_AP202508201730938917_1",
        industry_label="互联网电商",
        section_keywords=["归母净利润", "半年报"],
        must_contain_any=["1.53", "归母净利润"],
        gold_chunk_ids=["H3_AP202508201730938917_1_0001"],
    ),
    q(
        "q136",
        "吉宏股份2025年上半年归母净利润及增速",
        "2025H1归母净利润约1.18亿元，同比+63.3%",
        stock_code="002803",
        doc_id="H3_AP202508271735306180_1",
        industry_label="互联网电商",
        section_keywords=["归母净利润", "半年报", "同比"],
        must_contain_any=["1.18", "归母净利润"],
        gold_chunk_ids=["H3_AP202508271735306180_1_0011"],
    ),
    q(
        "q137",
        "吉宏股份AI赋能与全球化布局",
        "AI赋能+全球化布局驱动成长",
        stock_code="002803",
        doc_id="H3_AP202508271735306180_1",
        industry_label="互联网电商",
        category="text",
        section_keywords=["AI", "全球化", "跨境"],
        must_contain_any=["AI", "全球"],
        gold_chunk_ids=["H3_AP202508271735306180_1_0005"],
    ),
    q(
        "q138",
        "吉宏股份2025Q2营业收入同比增速",
        "25Q2营收同比约+55.5%",
        stock_code="002803",
        doc_id="H3_AP202508271735306180_1",
        industry_label="互联网电商",
        section_keywords=["营收", "同比", "Q2"],
        must_contain_any=["55.5", "同比", "营收"],
        gold_chunk_ids=["H3_AP202508271735306180_1_0011"],
    ),
    q(
        "q139",
        query_type="comparative",
        category="compare",
        query="华鼎股份和吉宏股份2025年上半年归母净利润对比",
        gold_answer="华鼎约1.53亿 vs 吉宏约1.18亿",
        stock_code="601113",
        doc_id="H3_AP202508201730938917_1",
        industry_label="互联网电商",
        section_keywords=["归母净利润"],
        must_contain_any=["归母净利润"],
        gold_chunk_ids=["H3_AP202508201730938917_1_0001"],
    ),
    q(
        "q140",
        query_type="comparative",
        category="compare",
        query="若羽臣和吉宏股份跨境电商增长驱动因素对比",
        gold_answer="品牌孵化 vs AI+跨境营销",
        stock_code="002803",
        doc_id="H3_AP202508271735306180_1",
        industry_label="互联网电商",
        section_keywords=["跨境", "品牌", "AI"],
        must_contain_any=["跨境", "品牌"],
    ),
    q(
        "q141",
        query_type="summary",
        category="summary",
        query="跨境电商企业自有品牌与代运营业务模式差异",
        gold_answer="若羽臣/青木/华凯等模式对比",
        stock_code="003010",
        doc_id="H3_AP202605061821991586_1",
        industry_label="互联网电商",
        section_keywords=["自有品牌", "代运营", "品牌"],
        must_contain_any=["品牌", "代运营"],
    ),
    q(
        "q142",
        query_type="summary",
        category="summary",
        query="互联网电商板块AI营销与数字化趋势",
        gold_answer="吉宏股份等AI赋能论述",
        stock_code="002803",
        doc_id="H3_AP202508271735306180_1",
        industry_label="互联网电商",
        section_keywords=["AI", "数字化", "营销"],
        must_contain_any=["AI", "数字化"],
    ),
    q(
        "q143",
        query_type="trap",
        category="trap",
        query="吉宏股份2025年上半年营收是多少？",
        gold_answer="应命中吉宏002803，不应命中华凯易佰",
        stock_code="002803",
        doc_id="H3_AP202508271735306180_1",
        industry_label="互联网电商",
        section_keywords=["营收", "2025H1"],
        must_contain_any=["32.34", "营收"],
        gold_chunk_ids=["H3_AP202508271735306180_1_0011"],
        negative_stock_codes=["300592"],
    ),
    # ── 半导体 q144–q150 ──
    q(
        "q144",
        "拓荆科技2025年营业收入",
        "2025年营收约65.19亿元，同比+58.9%",
        stock_code="688072",
        doc_id="H3_AP202604301821862002_1",
        industry_label="半导体",
        section_keywords=["营收", "2025", "PECVD"],
        must_contain_any=["65.19", "营收"],
        gold_chunk_ids=["H3_AP202604301821862002_1_0005"],
    ),
    q(
        "q145",
        "拓荆科技薄膜沉积设备业务进展",
        "PECVD/ALD薄膜沉积设备收入高增",
        stock_code="688072",
        doc_id="H3_AP202604301821862002_1",
        industry_label="半导体",
        category="text",
        section_keywords=["薄膜沉积", "PECVD", "ALD"],
        must_contain_any=["薄膜", "PECVD"],
        gold_chunk_ids=["H3_AP202604301821862002_1_0005"],
    ),
    q(
        "q146",
        "中微公司2026年一季度营业收入",
        "1Q26营收约29.15亿元，同比+34.13%",
        stock_code="688012",
        doc_id="H3_AP202605061821990814_1",
        industry_label="半导体",
        section_keywords=["营收", "一季报", "2026"],
        must_contain_any=["29.15", "营收"],
        gold_chunk_ids=["H3_AP202605061821990814_1_0001"],
    ),
    q(
        "q147",
        "卓胜微2025年营业收入与净利润",
        "2025年收入约37.26亿元，归母净利润约-2.93亿元",
        stock_code="300782",
        doc_id="H3_AP202604301821812915_1",
        industry_label="半导体",
        section_keywords=["营收", "归母净利润", "2025"],
        must_contain_any=["37.26", "2.93"],
        gold_chunk_ids=["H3_AP202604301821812915_1_0001"],
    ),
    q(
        "q148",
        query_type="comparative",
        category="compare",
        query="拓荆科技和中微公司半导体设备业务定位差异",
        gold_answer="薄膜沉积 vs 刻蚀设备",
        stock_code="688072",
        doc_id="H3_AP202604301821862002_1",
        industry_label="半导体",
        section_keywords=["薄膜", "刻蚀", "设备"],
        must_contain_any=["薄膜", "设备"],
        gold_chunk_ids=["H3_AP202604301821862002_1_0005"],
    ),
    q(
        "q149",
        query_type="summary",
        category="summary",
        query="半导体设备行业2025年扩产景气与资本开支",
        gold_answer="拓荆/中微/北方华创等设备股行业段",
        stock_code="688072",
        doc_id="H3_AP202604301821862002_1",
        industry_label="半导体",
        section_keywords=["扩产", "资本开支", "景气"],
        must_contain_any=["扩产", "资本开支", "景气"],
    ),
    q(
        "q150",
        query_type="summary",
        category="summary",
        query="研报库四行业（半导体/电力/电商/白色家电）2025年业绩概览",
        gold_answer="跨行业2025年营收利润综述",
        stock_code="",
        doc_id="",
        industry_label="",
        section_keywords=["2025年", "营收", "归母净利润", "业绩"],
        must_contain_any=["2025", "营收", "净利润"],
    ),
]


def enrich_gold_chunks(questions: list[dict], by_doc: dict[str, list[dict]]) -> None:
    for item in questions:
        if item.get("gold_chunk_ids"):
            continue
        keywords = list(item.get("section_keywords") or []) + list(
            item.get("must_contain_any") or []
        )
        keywords = [keyword for keyword in keywords if len(keyword) >= 2][:6]
        item["gold_chunk_ids"] = find_gold_chunks(
            item.get("doc_id", ""),
            keywords,
            by_doc=by_doc,
            limit=3,
        )


def validate_questions(questions: list[dict], by_doc: dict[str, list[dict]]) -> None:
    all_chunk_ids = {
        chunk["chunk_id"]
        for chunks in by_doc.values()
        for chunk in chunks
    }
    missing_gold: list[str] = []
    for item in questions:
        for chunk_id in item.get("gold_chunk_ids") or []:
            if chunk_id not in all_chunk_ids:
                missing_gold.append(f"{item['id']}:{chunk_id}")

    if missing_gold:
        preview = ", ".join(missing_gold[:10])
        raise ValueError(f"gold_chunk_id 不存在（{len(missing_gold)} 个）：{preview}")

    industry_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for item in questions:
        industry = item.get("industry_label") or "(跨行业)"
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        query_type = item.get("query_type", "factual")
        type_counts[query_type] = type_counts.get(query_type, 0) + 1

    print(f"行业分布：{dict(sorted(industry_counts.items()))}")
    print(f"query_type 分布：{type_counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    legacy: list[dict] = []
    with open(INPUT_PATH, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item["id"] in LEGACY_PATCHES:
                    item.update(LEGACY_PATCHES[item["id"]])
                legacy.append(item)

    if len(legacy) != 90:
        raise ValueError(f"当前评测集应为 90 题，实际 {len(legacy)} 题")

    by_doc = load_chunk_index()
    enrich_gold_chunks(NEW_QUESTIONS, by_doc)

    merged = legacy + NEW_QUESTIONS
    if len(merged) != TARGET_COUNT:
        raise ValueError(f"期望 {TARGET_COUNT} 题，实际 {len(merged)} 题")

    existing_ids = {item["id"] for item in merged}
    if len(existing_ids) != len(merged):
        raise ValueError("存在重复题目 id")

    validate_questions(merged, by_doc)

    if args.dry_run:
        print(f"[dry-run] 校验通过，将写入 {TARGET_COUNT} 题")
        return

    if not BACKUP_PATH.exists():
        shutil.copy2(INPUT_PATH, BACKUP_PATH)
        print(f"已备份 90 题至 {BACKUP_PATH}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for item in merged:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已写入 {OUTPUT_PATH}，共 {len(merged)} 题")


if __name__ == "__main__":
    main()
