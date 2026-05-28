# 评测结果快照（200 份研报 × 150 题）

> **评测日期**：2026-05-28  
> **索引版本**：P2 分块（`chunk_mineru` v3 + 评级块 / 可比表标记 / 附录合并 / 表语义化）  
> **默认混合权重**：向量 **0.35** / BM25 **0.65**（`retrieval.DEFAULT_HYBRID_VECTOR_WEIGHT`）  
> **Rerank 配置**：初召回 pool **200** → Top **30** → Rerank → Top **5**；拒答阈值 **0.35**

---

## 1. 数据与索引规模

| 项目 | 数值 |
|------|------|
| 研报 PDF | **200**（半导体 / 电力 / 互联网电商 / 白色家电 各 50） |
| 总 chunk | 10,263 |
| 可检索 chunk | **7,382** |
| Milvus 向量条数 | 7,382 |
| BM25 文档数 | 7,382 |
| `rating_headline` 块 | 114 |
| `comparable_table` 块 | 46 |
| 评测题 | **150**（factual 100 / comparative 26 / summary 23 / trap 1） |
| 评测集路径 | `data/eval/eval_questions.jsonl` |
| `gold_chunk_id` 校验 | 150/150 全部存在 |

---

## 2. 三路召回（Top-10，混合权重 0.35）

| 路线 | Recall@3 | Recall@5 | Recall@10 | MRR | 命中数 @10 |
|------|----------|----------|-----------|-----|------------|
| A 纯向量 | 80.0% | 83.3% | 86.0% | 0.748 | 129/150 |
| B 纯 BM25 | 78.7% | 84.7% | **92.0%** | 0.750 | 138/150 |
| **C 混合** | **86.7%** | **90.7%** | **92.0%** | **0.836** | **138/150** |

**按 query_type（混合 Recall@10）**

| query_type | 题量 | Recall@3 | Recall@5 | Recall@10 | MRR |
|------------|------|----------|----------|-----------|-----|
| factual | 100 | 90.0% | 93.0% | **95.0%** | 0.888 |
| comparative | 26 | 80.8% | 80.8% | **80.8%** | 0.712 |
| summary | 23 | 78.3% | 91.3% | 91.3% | 0.743 |

**解读**：混合在 **Recall@3/5 与 MRR** 上优于单路 BM25；Recall@10 与 BM25 持平（92%），但浅层排序更利于 Rerank 前段。

**产出文件**：`data/eval/eval_route_comparison.csv`、`eval_route_comparison_by_query_type.csv`

---

## 3. 混合权重分组实验（150 题，Recall@10）

脚本：`scripts/eval_hybrid_weight_sweep.py`  
产出：`data/eval/eval_hybrid_weight_sweep.csv`

| 向量权重 | BM25 权重 | Recall@10 | MRR | 命中 @10 |
|----------|-----------|-----------|-----|----------|
| 0.50 | 0.50 | 91.3% | 0.834 | 137/150 |
| **0.40** | **0.60** | **92.0%** | **0.839** | **138/150** |
| **0.35** | **0.65** | **92.0%** | 0.836 | **138/150** |
| 0.30 | 0.70 | 91.3% | 0.833 | 137/150 |
| 0.60 | 0.40 | 90.0% | 0.822 | 135/150 |
| 0.70 | 0.30 | 90.7% | 0.831 | 136/150 |

**结论**：相对 0.5/0.5，**0.35~0.40 向量权重** 各多命中 1 题；生产默认采用 **0.35/0.65**（与 0.4/0.6 Recall 相同，实现已落地）。

---

## 4. Rerank 与答案评测（混合 0.35 初召回）

| 策略 | Recall@5 | Top-1 准确率 | MRR | 检索命中率 |
|------|----------|--------------|-----|------------|
| 混合直接 Top5 | **92.7%** | 80.7% | 0.849 | 92.7% |
| 混合 Top30 → Rerank → Top5 | 90.7% | 78.0% | 0.832 | 90.7% |

| 策略 | 答案事实准确率 | 拒答恰当率 | 拒答率 |
|------|----------------|-----------|--------|
| 混合直接 Top5 | 84.0% | 82.0% | 2.0% |
| **混合 + Rerank** | **88.0%** | 82.7% | 0.67% |

**解读**：Rerank 提升 **答案事实准确率 +4.0pp**，略牺牲 Recall@5（-2.0pp）；拒答率更低、幻觉相对可控。

**产出文件**：`data/eval/eval_rerank_comparison.csv`、`eval_rerank_answer_comparison.csv`、`eval_rerank_answer_results.csv`

---

## 5. 优化历程（150 题，混合 Recall@10 / Rerank 答案准确率）

| 阶段 | 主要改动 | Recall@10 | 答案准确率 |
|------|----------|-----------|------------|
| 基线（200 份索引，优化前） | 原分块 + 混合 0.5 | 84.7% | 82.7% |
| **P0** | relevance 放宽、token 别名、embedding_text 摘录 | — | — |
| **P1** | stock boost、查询增强、对比 RRF、pool=200、权重 0.35 | 87.3%* | 86.7%* |
| **P2** | 评级块、表语义、可比表降权、附录合并 + 重索引 | **92.0%** | **88.0%** |

\*P0+P1 数值为 P2 重切分前索引上的评测，见 `logs/eval_retrieval_150_p01.log`。

**相对 150 题基线（84.7% / 82.7%）累计提升**：Recall@10 **+7.3pp**，答案准确率 **+5.3pp**。

---

## 6. 当前主要失败面（Badcase 摘要）

| 类别 | 题数 | 说明 |
|------|------|------|
| 混合 Recall@10 未命中 | **12** | `eval_misses_hybrid.jsonl` |
| 答案事实错误（Rerank 管线） | **18** | 其中检索已命中仍错 **12** |
| 拒答不当 | **26** | 应拒未拒 **25** |

**检索未命中主因**：对比题错股/缺实体（6）、对股但 gold 段未进 Top10（6）。  
**答案错误主因**：EPS/精确数字未写入摘录（7）、报告期错配（3）、多关键词摘要（2）。

完整分析与 P3 方案见 **[eval-badcase-analysis.md](eval-badcase-analysis.md)**。

---

## 7. 复现命令

```bash
cd commercial-rag
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub

# 索引流水线（P2 后需全量重跑）
python src/chunk_mineru.py
python src/embed_chunks.py
python src/build_bm25_index.py

# 评测
python src/eval_retrieval.py --compare-routes --top-k 10
python scripts/eval_hybrid_weight_sweep.py --weights 0.5 0.4 0.35 0.3 0.6 0.7
python src/eval_rerank.py
```

**日志**：`logs/eval_retrieval_150_p2.log`、`logs/eval_rerank_150_p2.log`、`logs/eval_hybrid_weight_sweep.log`

---

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| [eval-scheme.md](eval-scheme.md) | 评测集设计、指标定义、命令 |
| [eval-badcase-analysis.md](eval-badcase-analysis.md) | P2 后 Badcase 与 P3 优化建议 |
| [midterm-summary.md](midterm-summary.md) | 项目全历程实验与 POC 对照 |
| [chunk-scheme.md](chunk-scheme.md) | 分块策略（含 P2） |
