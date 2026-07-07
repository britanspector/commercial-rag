# 语义缓存三模式评测结果

> **评测日期**：2026-06-08  
> **评测脚本**：`src/eval_cache.py --modes off,l1,l1l2 --skip-ragas`  
> **评测集**：`data/eval/eval_questions.jsonl`（150 题：factual 100 / comparative 26 / summary 23 / trap 1）  
> **详细报告**：[`data/eval/eval_cache_report.md`](../data/eval/eval_cache_report.md)  
> **原始数据**：[`eval_cache_results.csv`](../data/eval/eval_cache_results.csv)、[`eval_cache_comparison.csv`](../data/eval/eval_cache_comparison.csv)

---

## 复现命令

```bash
cd commercial-rag
PYTHONPATH=src python src/eval_cache.py --modes off,l1,l1l2 --skip-ragas
```

---

## 1. 延迟与命中率（全量 150 题）

| 模式 | 命中率 | L1 | L2 | 平均延迟 | P50 | P95 | 向量检索 | LLM 调用 |
|------|--------|-----|-----|----------|-----|-----|----------|----------|
| cache_off / cold | 0% | — | — | **1171 ms** | 1004 ms | 2069 ms | 150 | 150 |
| l1_only / measure_l1 | 82.7%* | 82.7% | — | **343 ms** | 136 ms | 1693 ms | 26 | 26 |
| l1_l2 / measure_l1 | 82.7%* | 73.3% | 9.3% | 485 ms | 138 ms | 2498 ms | 26 | 26 |
| l1_l2 / measure_l2 | 76.7%* | — | 76.7% | **800 ms** | 521 ms | 2326 ms | 35 | 35 |

\* 全量含 26 道 **comparative** 题；该题型默认**不写入缓存**（`cache/policy.py`），故全量命中率低于可缓存子集。

### 可缓存题（124 题，排除 comparative）

| 指标 | 数值 |
|------|------|
| L1 精确命中（measure_l1） | **100%**（124/124） |
| L2 语义命中（measure_l2，paraphrase） | **92.7%**（115/124） |
| L1 延迟降幅（相对 cold） | **70.7%**（1171 ms → 343 ms） |
| L1 跳过向量检索 | 124 次 |
| L1 跳过 LLM 调用 | 124 次 |

---

## 2. 质量指标

| 模式 | Citation | Refusal | Recall@10 | MRR |
|------|----------|---------|-----------|-----|
| cache_off | 86.3% | 100% | 70.0% | 0.567 |
| l1_only / measure_l1 | 86.3% | 100% | 70.0% | 0.567 |
| l1_l2 / measure_l2 | 85.8% | 100% | 67.3% | 0.549 |

Citation / Refusal 与 cache_off 差异 ≤ 0.05（标准 #7 通过）。Recall@10 在本轮 eval 中与第一阶段基线存在差异，主因是评测实现路径与 gold 标注口径；缓存命中返回 warmup 相同 payload，**l1_only measure 与 cold 质量指标完全一致**。

---

## 3. 第二阶段完成标准

| # | 标准 | 结果 |
|---|------|------|
| 1 | 缓存开/关可切换 | 通过 |
| 2 | 相同问题 L1 命中（可缓存题） | 通过（124/124） |
| 3 | 语义相近 L2 命中（可缓存题） | 通过（115/124，92.7%） |
| 4 | 版本/metadata 变化不误命中 | 通过 |
| 5 | 请求级遥测字段齐全 | 通过（900 条） |
| 6 | 三模式对比 | 通过 |
| 7 | 质量不明显下降 | 通过 |

**结论：7 项标准全部通过。**

---

## 4. 生产配置与前端

- **Redis L1**：见项目根 [`.env.example`](../.env.example) 与 [semantic-cache-scheme.md §15](semantic-cache-scheme.md#151-生产启用-redis-l1)
- **自测**：`PYTHONPATH=src python -m cache.self_test --redis-url redis://127.0.0.1:6379/0`
- **前端**：`/cache` 页对接 `GET /cache/stats`；`/search`、`/chat` 展示响应 `cache` 字段

---

## 相关文档

- 技术方案：[semantic-cache-scheme.md](semantic-cache-scheme.md)
- 第一阶段生成评测：[eval-results.md](eval-results.md)
