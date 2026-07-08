# 评测体系

本文档说明项目如何评测检索、Rerank 和生成质量，以及这些脚本之间的关系。

## 评测目标

项目不是只看“能不能回答”，而是把评测拆成三层：

1. 检索层：能不能把相关 chunk 找回来
2. Rerank 层：能不能把更适合回答的 chunk 排到前面
3. 生成层：能不能在正确证据上生成带引用、可拒答的答案

## 1. 评测集

主评测集位于：

- `data/eval/eval_questions.jsonl`

当前规模：

- `150` 题
- `factual 100`
- `comparative 26`
- `summary 23`
- `trap 1`

每题会包含问题、题型、参考答案，以及命中约束字段，例如：

- `stock_code`
- `doc_id`
- `section_keywords`
- `must_contain_any`
- `gold_chunk_ids`
- `negative_stock_codes`

## 2. 检索评测

入口文件：

- `src/eval_retrieval.py`

目标：

- 对比 `vector`、`bm25`、`hybrid` 三条召回路线
- 评估 Recall@K、Context Precision 和 MRR
- 分析命中失败样本

常用命令：

```bash
python src/eval_retrieval.py --dry-run
python src/eval_retrieval.py --route vector
python src/eval_retrieval.py --compare-routes --top-k 10
python src/eval_retrieval.py --route hybrid --pipeline-stage rerank
```

主要输出：

- `data/eval/eval_route_comparison.csv`
- `data/eval/eval_route_comparison_by_query_type.csv`
- `data/eval/eval_results_{route}.csv`
- `data/eval/eval_metrics_{route}.csv`
- `data/eval/eval_misses_{route}.jsonl`

## 3. Rerank 评测

入口文件：

- `src/eval_rerank.py`

目标：

- 比较“混合直接 Top-K”与“混合召回后再 Rerank”的效果
- 同时评估检索命中和规则答案准确率

常用命令：

```bash
python src/eval_rerank.py
python src/eval_rerank.py --skip-answer
```

主要输出：

- `data/eval/eval_rerank_comparison.csv`
- `data/eval/eval_rerank_answer_comparison.csv`
- `data/eval/eval_rerank_answer_results.csv`

## 4. 生成评测

入口文件：

- `src/eval_generation.py`
- `src/eval_ragas.py`

目标：

- 跑完整 `RAGPipeline`
- 评估回答是否带有效引用
- 评估拒答是否合理
- 在需要时补跑 RAGAS 指标

### 阶段一：规则指标

```bash
python src/eval_generation.py --dry-run
python src/eval_generation.py --skip-ragas --save-detail
python src/eval_generation.py --skip-ragas --save-detail --resume
```

主要关注：

- `citation_accuracy`
- `refusal_accuracy`
- `answer_factually_supported`

### 阶段二：RAGAS

```bash
ollama serve
python src/eval_ragas.py
python src/eval_ragas.py --limit 3
python src/eval_ragas.py --resume
```

RAGAS 常见指标：

- `faithfulness`
- `answer_relevancy`

## 5. 指标分层理解

### 检索层

- `Recall@3 / @5 / @10`
- `Context Precision@K`
- `MRR`

### 生成层规则指标

- `citation_accuracy`：引用是否齐全、定位是否合理
- `refusal_accuracy`：该拒答时是否拒答，不该拒答时是否继续回答
- `answer_factually_supported`：答案是否满足关键事实约束

### RAGAS 指标

- `Faithfulness`：答案是否忠于给定上下文
- `Answer Relevancy`：答案与问题是否相关

## 6. 评测脚本之间的关系

```text
eval_retrieval.py
  -> 检索层指标：Recall / Precision / MRR

eval_rerank.py
  -> 重排后的检索效果 + 规则答案准确率

eval_generation.py
  -> 完整问答链路 + Citation / Refusal / 规则支持度

eval_ragas.py
  -> 基于 detail 文件补跑 RAGAS，不重跑整个 Pipeline
```

## 7. 结果文件

生成评测最常用的结果文件包括：

- `data/eval/eval_generation_results.csv`
- `data/eval/eval_generation_metrics.csv`
- `data/eval/eval_generation_metrics_by_query_type.csv`
- `data/eval/eval_generation_detail.jsonl`
- `data/eval/eval_generation_detail_ragas.jsonl`

完整快照见 [eval-results.md](eval-results.md)。

## 8. 评测运行建议

- 跑生成评测前先确认索引已就绪
- 本地生成评测依赖 Ollama，通常比检索评测慢很多
- RAGAS 依赖额外 LangChain / ragas 版本栈，尽量和普通运行环境分开管理
- 评测过程中不要和占用 `milvus.db` 的其他流程并行运行

## 9. 相关文档

- [eval-results.md](eval-results.md)：当前结果快照
- [eval-badcase-analysis.md](eval-badcase-analysis.md)：Badcase 和后续优化
- [cache.md](cache.md)：缓存与评测隔离策略
- [service-ops.md](service-ops.md)：`POST /eval` 异步任务和服务运行说明
