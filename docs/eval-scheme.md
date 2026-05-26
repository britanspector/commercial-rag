# 检索评测说明

## 评测集

路径：`data/eval/eval_questions.jsonl`（28 题）

覆盖三类行业（半导体 / 电力 / 互联网电商），题型包括：

| 类型 | 说明 |
|------|------|
| financial | 营收、EPS、PE、净利润等 |
| table | 可比公司、分产品营收表 |
| rating | 投资评级、目标价 |
| risk | 风险提示 |
| cross | 行业/主题检索 |
| trap | 易混淆公司（如芯朋微 vs 澜起） |

每题字段：

| 字段 | 说明 |
|------|------|
| `query` | 用户自然语言问题 |
| `gold_answer` | 人工参考答案（便于核对） |
| `stock_code` / `doc_id` | 期望命中的文档 |
| `section_keywords` | 章节/正文应含关键词 |
| `must_contain_any` | 命中 chunk 正文应含任一关键词 |
| `gold_chunk_ids` | 可选：明确标注的标准 chunk |
| `negative_stock_codes` | 可选：Top 结果不应为该股票 |

## 运行流程

```bash
conda activate commercial-rag

# 1. 分块（已完成可跳过）
python src/chunk_mineru.py

# 2. 向量化写入 Milvus
python src/embed_chunks.py

# 3a. 仅校验评测题（无需 Milvus）
python src/eval_retrieval.py --dry-run

# 3b. 正式评测（默认 Top-10，输出 Recall@5 / Recall@10 / MRR）
python src/eval_retrieval.py
python src/eval_retrieval.py --top-k 10
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `data/eval/eval_metrics.csv` | 汇总指标 |
| `data/eval/eval_results.csv` | 每题明细 |
| `data/eval/eval_misses.jsonl` | **未命中** case，用于改 chunk 规则 |
| `data/eval/eval_detail.jsonl` | 全量结果 JSON |

## 指标含义

- **Recall@5 / Recall@10**：Top-K 内是否出现至少 1 个「相关」chunk
- **MRR**：第一个相关 chunk 的排名倒数（1/rank），未命中为 0
- **命中率**：与 Recall@10 在本脚本中一致（K=10 时）

## 维护评测集

1. 用 `check_milvus.py "你的问题"` 看 Top-5 实际召回
2. 将正确 chunk 的 `chunk_id` 填入 `gold_chunk_ids`
3. 调整 `section_keywords` / `must_contain_any` 避免过严或过松
4. 重跑 `python src/eval_retrieval.py` 对比指标变化
