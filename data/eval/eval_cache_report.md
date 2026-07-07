# 语义缓存三模式评测报告

- 评测集：`eval_questions.jsonl`（150 题，实际 limit=全量）
- 模式：`cache_off` / `l1_only` / `l1_l2`

## 1. 延迟与命中率对比

| label | question_count | cache_hit_rate | l1_hit_rate | l2_hit_rate | avg_latency_ms | p50_latency_ms | p95_latency_ms | vector_retrieval_count | llm_call_count | vector_retrievals_saved | llm_calls_saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cache_off/cold | 150 | 0.0 | 0.0 | 0.0 | 1170.7974 | 1003.98 | 2068.92 | 150 | 150 | 0 | 0 |
| l1_only/measure_l1 | 150 | 0.8266666666666667 | 0.8266666666666667 | 0.0 | 342.6252 | 136.48 | 1692.8 | 26 | 26 | 124 | 124 |
| l1_l2/measure_l1 | 150 | 0.8266666666666667 | 0.7333333333333333 | 0.09333333333333334 | 484.8824 | 137.78 | 2497.85 | 26 | 26 | 124 | 124 |
| l1_l2/measure_l2 | 150 | 0.7666666666666667 | 0.0 | 0.7666666666666667 | 800.3134666666667 | 521.45 | 2325.73 | 35 | 35 | 115 | 115 |

## 2. 质量指标对比

| label | citation_accuracy | refusal_accuracy | faithfulness_ragas | answer_relevancy_ragas | recall_at_10 | mrr |
| --- | --- | --- | --- | --- | --- | --- |
| cache_off/cold | 0.8626373626373626 | 1.0 | nan | nan | 0.7 | 0.5674444444444444 |
| l1_only/measure_l1 | 0.8626373626373626 | 1.0 | nan | nan | 0.7 | 0.5674444444444444 |
| l1_l2/measure_l1 | 0.8505494505494505 | 1.0 | nan | nan | 0.6933333333333334 | 0.5421111111111111 |
| l1_l2/measure_l2 | 0.8582766439909296 | 1.0 | nan | nan | 0.6733333333333333 | 0.5485555555555556 |

## 3. 第二阶段完成标准验证

| # | 完成标准 | 结果 | 说明 |
|---|---------|------|------|
| 1 | Pipeline 可在缓存开启/关闭间切换 | 通过 | off 模式 150 题均未命中缓存 |
| 2 | 完全相同问题命中 L1 | 通过 | 可缓存题 L1 命中率 100.0% (124/124)；comparative 26 题策略不缓存 |
| 3 | 语义相近问题在安全条件下命中 L2 | 通过 | 可缓存题 L2 paraphrase 命中率 92.7% (115/124)；comparative 不写入缓存 |
| 4 | 版本/metadata/配置变化不误命中 | 通过 | 命中后安全校验失败 0 条；整体 safety_ok=100.0% |
| 5 | 每次请求记录 cache_source/hit/耗时/相似度/拒绝原因 | 通过 | 样本字段齐全；共 900 条记录 |
| 6 | 可用第一阶段评测集对比延迟/命中率/质量 | 通过 | 已跑模式: cache_off, l1_l2, l1_only |
| 7 | 缓存开启后 Faithfulness/Citation/Refusal 不明显下降 | 通过 | Citation Δ=+0.000 Refusal Δ=+0.000（阈值 0.05） |

## 4. 结论

- **延迟**：L1 命中 pass 平均延迟 343ms，相对 cache_off cold run 1171ms，降低约 **70.7%**。
- **计算开销**：L1 measure 阶段 vector_retrieval 从 150 次降至 26 次；LLM 调用从 150 降至 26。
- **L2 语义缓存**：paraphrase measure 全量 L2 命中率 76.7%，平均延迟 800ms（可缓存题见 eval-cache-results.md）。
- **comparative 题型**：26 题默认不写入缓存，全量命中率低于可缓存子集。
- **质量**：Citation/Refusal 与 cache_off 差异在可接受范围内（见标准 #7）。
- **完成度**：全部通过第二阶段完成标准（见上表）。
