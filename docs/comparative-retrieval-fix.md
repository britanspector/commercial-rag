# 对比题多主体检索优化结果

> **评测时间**：2026-06-10  
> **场景**：`query_type=comparative`，`stock_code` 为空（模拟 UI /chat 不填代码）  
> **开关**：`RAG_COMPARATIVE_ENTITY_RERANK=1`（默认开启）

## 问题根因

对比题虽有多路 Recall（`entity + 整句`），但存在两点缺陷：

1. **子查询仍含另一主体**，BM25/向量偏向单公司；
2. **共用整句 `query_vector`**，分主体召回未真正独立；
3. **Rerank 用整句对合并池打分**，Top-5 易被一家公司占满 → `evidence_check` 触发 `comparative_insufficient`。

## 方案摘要

| 阶段 | 改动 | 文件 |
|------|------|------|
| 子查询 | `{主体} {年份} {指标}`，去掉对比词与另一主体 | `query_enhance.py` |
| 分主体 embed | comparative 时每主体独立 `encode_query` | `query_rewrite.py`, `rag_types.py` |
| 分主体 Recall | 独立 `(sub_query, sub_vector)` 多路 RRF | `retrieval.py`, `hybrid_retrieve.py` |
| 分主体 Rerank | 每主体子查询 rerank，`slots_per_entity=2` 配额合并 | `pipeline/comparative_rerank.py` |
| Pipeline 接入 | 对比题走 `rerank_step_for_rewrite` | `rag_pipeline.py` |

**未修改** `evidence_check.py` 规则；通过保证 Rerank Top-5 覆盖 ≥2 家公司来满足现有检查。

## 用户 badcase 回归（3 条）

| 问题 | 优化前 Top-5 | 优化后 `/search` Top-5 | `comparative_insufficient` |
|------|-------------|------------------------|----------------------------|
| 澜起科技和华能国际2026年的EPS对比？ | 仅华能国际（5/5） | 澜起×2 + 华能×3（2 家） | **已消除** |
| 中国广核和华能国际2026年的EPS对比？ | 仅中国广核（5/5） | 广核×3 + 华能×2（2 家） | **已消除** |
| 中国广核和华能国际2025年归母净利润规模对比 | 两家均有 | 广核×4 + 华能×1（2 家） | **未退化** |

### `/chat` 补充说明

| 问题 | `/chat` 结果 | 说明 |
|------|-------------|------|
| 澜起 vs 华能 2026 EPS | `refused=false` | 正常生成 |
| 广核 vs 华能 2026 EPS | `refused=true`，`weak_evidence_intent` | 多主体证据已通过（`comparative_entities found=2`），但 Top-1 章节为「投资要点」与财务指标意图不匹配 |
| 广核 vs 华能 2025 净利润 | `refused=false` | 正常生成 |

## 26 题 comparative 子集评测

**脚本**：Pipeline 直连 `_run_search_core` + `evidence_check`（不调 LLM）  
**明细**：`data/eval/comparative_retrieval_eval.json`

| 指标 | 优化后 |
|------|--------|
| 题数 | 26 |
| Top-5 含 ≥2 家公司 | **23/26（88.5%）** |
| `comparative_insufficient` | **3/26（11.5%）** |
| 证据检查通过率 | 19/26（73.1%） |
| 总拒答率（含其他原因） | 7/26（26.9%） |

### 拒答原因分布

| 原因 | 数量 | 说明 |
|------|------|------|
| （通过） | 19 | — |
| `comparative_insufficient` | 3 | 见下表「预期残留」 |
| `insufficient_passage` | 3 | 片段过短，与本次检索优化无关 |
| `weak_evidence_intent` | 1 | 章节/意图不匹配 |

### 仍 `comparative_insufficient` 的 3 题（预期残留）

| ID | 问题 | 原因 |
|----|------|------|
| q23 | 半导体行业DDR5内存接口芯片相关公司 | 非典型双主体对比（行业泛问） |
| q74 | 中国核电两份研报对2026E EPS预测是否一致 | 同一公司两份研报，`distinct_companies=1`（本 PR 非目标） |
| q75 | 京仪装备温控设备与半导体专用设备营收结构对比 | 产品维度 vs 公司维度，Recall 池仅单公司 |

优化前同类题（如 q61 澜起 vs 芯朋微 PE、q72 华峰测控 vs 华凯易佰）在旧 Pipeline 中频繁 `comparative_insufficient`；优化后 **q61、q72 均已通过**。

## 单元测试

```bash
cd commercial-rag
PYTHONPATH=src python -m test_comparative_retrieval
# Ran 5 tests — OK
```

## 回退

设 `RAG_COMPARATIVE_ENTITY_RERANK=0` 可关闭分主体 Rerank，回退为整句 rerank（comparative Recall 仍保留分主体子查询路径）。

## 运行 API

```bash
cd commercial-rag
RAG_COMPARATIVE_ENTITY_RERANK=1 uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
```
