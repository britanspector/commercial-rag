# 缓存设计与评测

本文档统一说明项目中的语义缓存设计、接入点、评测结果和使用边界。

## 为什么要做缓存

当前 `RAGPipeline` 的主要耗时集中在这些阶段：

- 查询向量生成
- 混合召回
- Rerank
- LLM 生成

对于重复问题或近似问题，如果能在不破坏引用正确性与拒答契约的前提下复用历史结果，就可以显著降低 `/search` 和 `/chat` 的延迟。

## 当前接入位置

缓存围绕统一主链路工作，而不是散落在各个脚本里。

主要入口：

- `src/cache/pipeline_bridge.py`
- `src/cache/manager.py`
- `src/cache/index_fingerprint.py`
- `src/cache/invalidate_hooks.py`

作用方式：

- `run_search_with_cache()` 包裹 `RAGPipeline.run_search()`
- `run_chat_with_cache()` 包裹 `RAGPipeline.run()`

这意味着缓存是对统一问答主链路的增强，而不是一套平行逻辑。

## 缓存层级

项目当前设计把缓存拆成两类主范围：

- `search`：缓存检索 + 重排结果
- `chat`：缓存完整回答结果

在概念上可以理解为：

- L1：`/search` 结果缓存
- L2：`/chat` 结果缓存

此外还有语义近邻命中的设计，用于近似问题复用。

## 设计原则

缓存设计遵循这些原则：

1. 缓存的是“已验证的 Pipeline 结果”，不是裸 LLM 文本
2. 命中缓存后仍要做轻量安全校验
3. 索引、配置或生成参数变化时，旧缓存不能复用
4. 评测默认绕过缓存，避免指标失真
5. 对拒答结果保持谨慎，不默认长期缓存

## Cache Key 与失效

缓存命中不只取决于用户问题本身，还取决于这些信息：

- `scope`：`search` 或 `chat`
- 规范化后的 query
- `stock_code`
- `query_type`
- 检索和重排配置
- 当前索引指纹
- 生成模型与 Prompt 指纹

索引指纹通常和这些产物相关：

- `data/parsed/chunks.jsonl`
- `data/vector/milvus.db`
- `data/vector/bm25_index.pkl`
- manifest / 行数 / hash 等元信息

这样做的目标是：只要知识库或配置有变化，就强制旧缓存失效。

## 命中后的安全校验

缓存命中后仍要校验以下条件：

- 索引指纹是否一致
- 请求配置是否一致
- 近似命中时语义相似度是否达标
- 被引用的 chunk 是否仍存在
- `chat` 场景下 evidence check 是否仍通过
- `stock_code` 等约束是否仍成立
- TTL 是否过期

只有这些校验通过，缓存结果才能返回给用户。

## 与评测的关系

缓存与评测的边界必须非常清楚。

这些路径默认应绕过缓存：

- `src/eval_retrieval.py`
- `src/eval_generation.py`
- `src/eval_ragas.py`
- `POST /eval` 触发的后台任务

原因很简单：评测需要反映真实的检索、重排、证据校验和生成成本，不能把缓存命中算作系统本身的能力。

## 缓存结果与历史评测

项目历史上曾有多份缓存结果文档，主要覆盖两类信息：

- 缓存命中率与延迟收益
- 不同模式、不同数据集下的命中行为对比

相关结果文件集中在：

- `data/eval/eval_cache_results.csv`
- `data/eval/eval_cache_comparison.csv`
- `data/eval/eval_cache_report.md`
- `data/eval/cache_hit_results_*.csv`
- `data/eval/cache_hit_report_*.md`

这些结果说明缓存已经不仅是设计概念，也有配套评测和报告产物。

## 与审计和前端的关系

缓存命中情况需要同时体现在：

- 审计记录
- API 响应字段
- 前端缓存监控页面

当前前端已经预留 `缓存监控` 页面，对应 `GET /cache/stats`。

需要关注的典型指标包括：

- L1 / L2 命中率
- 请求级命中率
- 平均延迟
- 平均节省延迟
- 节省的向量检索次数
- 节省的 LLM 调用次数

## 使用与排查

最常见的排查思路：

1. 看 `/health` 和 `/cache/stats` 是否显示缓存已启用
2. 确认索引是否刚更新，导致旧缓存整体失效
3. 确认请求参数是否和历史缓存配置一致
4. 判断是不是 evidence check 或 `stock_code` 校验阻止了缓存复用

## 相关实现

建议按这个顺序阅读代码：

1. `src/cache/config.py`
2. `src/cache/manager.py`
3. `src/cache/index_fingerprint.py`
4. `src/cache/pipeline_bridge.py`
5. `src/cache/invalidate_hooks.py`
6. `src/cache/backends/`

## 相关文档

缓存原始设计与历史评测已整合到本文档。若需要看旧入口，可参考：

- `semantic-cache-scheme.md`
- `eval-cache-results.md`
- `cache-hit-eval-results.md`
