# 缓存命中效果评测结果

> **评测时间**：2026-06-10  
> **脚本**：`src/eval_cache_hit.py`  
> **测试集**：`data/eval/cache_hit_pairs.jsonl`（125 组，含用户回归用例）  
> **场景**：`stock_code` 为空（模拟前端 /chat 不填代码）

## 改进摘要

| 项 | 改动 |
|----|------|
| L1 规范化 | 礼貌词剥离、金融同义词（EPS↔每股收益）、语序归一为「公司+年份+指标」 |
| L2 默认启用 | `l2_backend=milvus`，`sim_threshold=0.88` |
| 公司 metadata | 从问题文本提取 `company_hint`，修复「请问」前缀导致 metadata 不一致 |
| 评测基建 | `eval_cache_hit.py` + 125 组 paraphrase 测试对 + 结构化日志 |

## 用户回归用例

| 用例 | 改进前 | 改进后（run 20260610_092656 + 语序 fix） |
|------|--------|------------------------------------------|
| 澜起科技2026年EPS ↔ 请问…EPS是？ | L1 miss | **L1 命中**（`l1_exact`, sim=1.0） |
| 澜起科技2024年EPS ↔ 请告诉我…每股收益 | L1/L2 miss，检索漂移到焦点科技 | 语序+同义词归一后 **应 L1 命中**（见单测） |

## 全量评测（125 对 × 2 模式）

**run_id**: `20260610_092656`  
**日志**: `data/eval/logs/cache_hit_20260610_092656.log`  
**明细**: `data/eval/cache_hit_results_20260610_092656.csv`  
**报告**: `data/eval/cache_hit_report_20260610_092656.md`

| 模式 | 命中率 | L1 | L2 | punctuation 类 |
|------|--------|----|----|----------------|
| l1_only | 28.0% (35/125) | 35 | 0 | **83.3%** |
| l1_l2 | **56.8%** (71/125) | 34 | 37 | **83.3%** |

### 按 variant 分析（l1_l2）

| variant_type | 命中率 | 说明 |
|--------------|--------|------|
| punctuation | 83.3% | 达标，L1 主力 |
| filler_suffix | 67.5% | L2 语义命中为主 |
| word_order | 32.1% | 语序改写 + 后续 canonical 改进可提升 |
| polite_tell | 0% | paraphrase 与原文差异过大，需 L2 阈值或模板优化 |
| synonym_metric | 样本少 | 仅 1 条 |

### 主要 badcase 原因

1. **warmup 本身未写入缓存**（`refused=true` 或 comparative 策略）→ measure 必然 miss  
2. **polite_tell / word_order** 改写幅度大，向量相似度 < 0.88  
3. **summary 类长问题**无明确公司实体，metadata 防护与 embedding 均不稳定  

## 复现命令

```bash
cd commercial-rag
PYTHONPATH=src python src/scripts/generate_cache_hit_pairs.py
PYTHONPATH=src python src/eval_cache_hit.py --modes l1_only,l1_l2 --stock-code-mode empty
PYTHONPATH=src python -m cache.test_normalize_query
```

## 生产启用

见 [`.env.example`](../.env.example)：开启 `RAG_SEMANTIC_CACHE_ENABLED=1` 与 L2 Milvus 配置后重启 uvicorn。
