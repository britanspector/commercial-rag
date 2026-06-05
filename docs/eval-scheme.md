# 检索评测说明

## 评测集

路径：`data/eval/eval_questions.jsonl`（**150 题**）

历史版本备份：`data/eval/eval_questions_90.jsonl`

| query_type | 说明 | 目标占比 | 示例 |
|------------|------|----------|------|
| `factual` | 事实型：单一指标/评级 | ~65% | 美的集团2025H1归母净利润 |
| `comparative` | 对比型：多公司/多指标比较 | ~19% | 美的和格力2025H1净利润谁更高 |
| `summary` | 汇总型：行业/主题归纳 | ~16% | 白色家电出口与全球化布局 |

题型字段 `category` 仍保留（financial / table / rating / text / compare / summary / trap 等）。

### 150 题评测集设计（四行业 × 200 份研报）

| 行业 | 题量 | 覆盖重点 |
|------|------|----------|
| 半导体 | ~47 | 澜起/芯朋微/设备（拓荆、中微）/存储等 |
| 电力 | ~37 | 核电/水电/火电/储能（中国广核、华能、南网储能等） |
| 互联网电商 | ~30 | 跨境/品牌/代运营/AI 营销（华鼎、吉宏等） |
| 白色家电 | ~36 | 美的/格力/海尔/海信/TCL智家/长虹美菱 |
| 跨行业 | 1 | 四行业 2025 业绩概览（q150） |

新增 60 题（q91–q150）以**白色家电**为主，补全第 4 行业；并加深电力、电商、半导体在新研报上的覆盖。`gold_chunk_ids` 人工标注 + 脚本按 `section_keywords` 自动补全。

每题字段：

| 字段 | 说明 |
|------|------|
| `query` | 用户自然语言问题 |
| `query_type` | factual / comparative / summary |
| `gold_answer` | 人工参考答案 |
| `stock_code` / `doc_id` | 期望命中的文档（汇总/跨行业可留空） |
| `industry_label` | 半导体 / 电力 / 互联网电商 / 白色家电 |
| `section_keywords` | 章节/正文应含关键词 |
| `must_contain_any` | 命中 chunk 正文应含任一关键词 |
| `gold_chunk_ids` | 可选：标准 chunk（用于 Recall 标注） |
| `negative_stock_codes` | 可选：不应出现的股票（陷阱题） |

构建脚本：

```bash
python scripts/build_eval_questions_150.py          # 从当前 90 题扩展至 150
python scripts/build_eval_questions_150.py --dry-run  # 仅校验
```

历史扩展：`scripts/build_eval_questions_90.py`（40→90）

## 三路召回

| 路线 | CLI `--route` | 说明 |
|------|---------------|------|
| A | `vector` | Milvus 纯向量（COSINE） |
| B | `bm25` | BM25Okapi + jieba（`data/vector/bm25_index.pkl`） |
| C | `hybrid` | 向量相似度与 BM25 分数 min-max 归一化后加权融合（**默认向量 0.35 / BM25 0.65**） |

实现：`src/retrieval.py`、`src/bm25_store.py`（`DEFAULT_HYBRID_VECTOR_WEIGHT = 0.35`，`DEFAULT_HYBRID_POOL_SIZE = 200`）

## 运行流程

```bash
conda activate commercial-rag

python src/chunk_mineru.py      # 如需重建分块
python src/embed_chunks.py      # Milvus 向量
python src/build_bm25_index.py  # BM25 索引

python src/eval_retrieval.py --dry-run
python src/eval_retrieval.py --route vector
python src/eval_retrieval.py --compare-routes --top-k 10
```

离线环境可设置：`HF_HUB_OFFLINE=1`

## 输出文件

| 文件 | 说明 |
|------|------|
| `data/eval/eval_route_comparison.csv` | **三路 Recall@3/5/10 对比** |
| `data/eval/eval_route_comparison_by_query_type.csv` | 按 query_type 分组对比 |
| `data/eval/eval_results_{route}.csv` | 每题明细 |
| `data/eval/eval_metrics_{route}.csv` | 单路汇总指标 |
| `data/eval/eval_misses_{route}.jsonl` | 未命中 case |

## 指标含义

- **Recall@3 / @5 / @10**：Top-K 内是否出现至少 1 个「相关」chunk
- **MRR**：第一个相关 chunk 的排名倒数
- **命中率**：Recall@10 等价（K=10 时）

## 当前评测结果（2026-05-28，P2 索引，150 题）

> 完整表格与复现命令见 **[eval-results.md](eval-results.md)**；Badcase 见 **[eval-badcase-analysis.md](eval-badcase-analysis.md)**。

| 实验 | 核心指标 |
|------|----------|
| 三路召回（混合 0.35） | Recall@10 **92.0%**，MRR **0.836** |
| 权重扫描最优 | 0.35~0.40 向量权重 → Recall@10 **92.0%**（基线 0.5/0.5 为 91.3%） |
| 混合 + Rerank | 答案事实准确率 **88.0%**（混合直接 Top5：84.0%） |
| 检索未命中 | **12** 题（`eval_misses_hybrid.jsonl`） |

**混合权重分组脚本**：

```bash
python scripts/eval_hybrid_weight_sweep.py --weights 0.5 0.4 0.35 0.3 0.6 0.7
# 产出：data/eval/eval_hybrid_weight_sweep.csv
```

**优化历程（150 题 Recall@10 / 答案准确率）**：84.7% / 82.7%（基线）→ 87.3% / 86.7%（P0+P1）→ **92.0% / 88.0%**（P2 + 权重 0.35）。

## 生成质量评测（阶段一）

全链路 Pipeline + Citation/Refusal + 可选 RAGAS，见 **[eval-generation-scheme.md](eval-generation-scheme.md)**：

```bash
python src/eval_generation.py --skip-ragas
```

## Milvus 索引类型实验

FAISS Flat/IVF/HNSW 对比在 **Milvus Lite 上不可直接复现**，可行方案见 `docs/milvus-index-comparison.md`。
