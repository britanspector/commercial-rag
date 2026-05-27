# 检索评测说明

## 评测集

路径：`data/eval/eval_questions.jsonl`（**90 题**）

| query_type | 说明 | 示例 |
|------------|------|------|
| `factual` | 事实型：单一指标/评级 | 京仪装备2026E毛利率多少 |
| `comparative` | 对比型：多公司/多指标比较 | 澜起科技和芯朋微哪家PE更高 |
| `summary` | 汇总型：行业/主题归纳 | 半导体DDR5发展趋势 |

题型字段 `category` 仍保留（financial / table / rating 等）。

每题字段：

| 字段 | 说明 |
|------|------|
| `query` | 用户自然语言问题 |
| `query_type` | factual / comparative / summary |
| `gold_answer` | 人工参考答案 |
| `stock_code` / `doc_id` | 期望命中的文档（汇总型可留空） |
| `section_keywords` | 章节/正文应含关键词 |
| `must_contain_any` | 命中 chunk 正文应含任一关键词 |
| `gold_chunk_ids` | 可选：标准 chunk |
| `negative_stock_codes` | 可选：不应出现的股票 |

扩展 40→90 题：`python scripts/build_eval_questions_90.py`

## 三路召回

| 路线 | CLI `--route` | 说明 |
|------|---------------|------|
| A | `vector` | Milvus 纯向量（COSINE） |
| B | `bm25` | BM25Okapi + jieba（`data/vector/bm25_index.pkl`） |
| C | `hybrid` | 向量相似度与 BM25 分数 min-max 归一化后加权融合（默认各 0.5） |

实现：`src/retrieval.py`、`src/bm25_store.py`

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

## Milvus 索引类型实验

FAISS Flat/IVF/HNSW 对比在 **Milvus Lite 上不可直接复现**，可行方案见 `docs/milvus-index-comparison.md`。
