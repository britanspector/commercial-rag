# 检索与问答主链路

本文档聚焦在线检索与问答流程，也就是 `RAGPipeline` 如何把一个问题变成“可引用的答案”或“合理拒答”。

## 主流程

当前统一入口是 `src/rag_pipeline.py`，默认步骤如下：

```text
query_rewrite
  -> hybrid_retrieve
  -> rerank
  -> evidence_check
  -> answer_generate
```

这条链路被 CLI、FastAPI 和离线评测共同复用。

## 1. 统一编排器

核心入口：

- `src/rag_pipeline.py`

它提供两类能力：

- `run_search()`：检索 + 重排，不生成答案
- `run()`：完整问答链路

对应使用场景：

- `/search`、检索调试、检索评测复用 `run_search()`
- `/chat`、生成评测和 CLI 问答复用 `run()`

## 2. Query Rewrite

相关文件：

- `src/pipeline/query_rewrite.py`
- `src/query_enhance.py`

职责：

- 规范化用户问题
- 提取 `stock_code`、题型等上下文
- 生成查询向量
- 为 BM25 构造更适合召回的扩展查询

它的目标不是改写成自然语言更通顺，而是让后续向量检索和 BM25 召回都能吃到更稳定的信号。

## 3. Hybrid Retrieve

相关文件：

- `src/retrieval.py`
- `src/pipeline/hybrid_retrieve.py`
- `src/bm25_store.py`
- `src/milvus_store.py`

当前支持三条路线：

- `vector`：Milvus 纯向量召回
- `bm25`：BM25 纯词法召回
- `hybrid`：向量分数和 BM25 分数融合

默认生产路线是 `hybrid`，当前默认参数：

- 向量权重：`0.35`
- BM25 权重：`0.65`
- 候选池：`200`

项目里保留三路召回，不只是为了实验，也为了：

- 对比不同召回来源的覆盖面
- 验证混合权重是否合理
- 在 badcase 分析时区分是向量问题、词法问题还是融合问题

## 4. Rerank

相关文件：

- `src/pipeline/rerank.py`
- `src/reranker.py`
- `src/pipeline/comparative_rerank.py`

当前默认重排模型是 `BAAI/bge-reranker-v2-m3`。

Rerank 的作用不是提高 Recall@10，而是把更适合回答的问题片段排到前面，从而提升：

- Top-1 证据质量
- 引用准确率
- 最终回答的事实支持度

当前常见对比是：

- 混合直接 Top-5
- 混合 Top-30 -> Rerank -> Top-5

项目现有结果表明，Rerank 会略微牺牲浅层召回，但能提升最终答案的事实准确率。

### 对比题的特殊处理

对 `query_type=comparative` 的问题，项目已经引入分主体召回和分主体 Rerank 逻辑，避免 Top-5 被单个公司占满。

相关实现：

- `src/pipeline/comparative_rerank.py`
- `src/pipeline/query_rewrite.py`
- `src/retrieval.py`

专项结果见历史文档 `comparative-retrieval-fix.md`。

## 5. Evidence Check

相关文件：

- `src/pipeline/evidence_check.py`

职责：

- 判断当前证据是否足以支持回答
- 校验问题意图、股票约束、重排分数和多主体覆盖
- 决定是继续生成答案还是进入拒答

它是系统“拒答契约”的核心，不允许仅凭“模型能写出一段话”就返回结果。

常见检查方向包括：

- rerank 分数是否低于阈值
- 证据与问题意图是否匹配
- comparative 题是否真的覆盖了多个主体
- `stock_code` 约束是否满足

## 6. Answer Generate

相关文件：

- `src/pipeline/answer_generate.py`
- `src/pipeline/compose.py`
- `src/rag_answer.py`

职责：

- 选择回答所需的上层证据片段
- 组织引用和文案
- 由本地 LLM 生成带引用的中文答案

当前生成默认依赖本地 Ollama，常见模型是 `qwen3:8b`。

回答结果通常包含：

- 回答正文
- 引用编号，如 `[1]`、`[2]`
- 文档、章节、页码、chunk 等参考信息

## 7. 拒答策略

当证据不足、意图不匹配或重排质量过低时，系统不会继续生成答案，而是返回拒答。

当前拒答是主链路的一部分，而不是生成失败后的兜底分支。这样做的目的有两个：

- 降低幻觉
- 让评测和产品行为使用同一套标准

## 8. 对外入口

主链路对外暴露为三类入口：

- `src/rag_chat.py`：命令行问答
- `src/api/main.py` 的 `/search`：检索调试
- `src/api/main.py` 的 `/chat`：完整问答

前端页面对应：

- `/search`：检索调试
- `/chat`：RAG 问答

## 9. 与离线构建和缓存的关系

- 主链路依赖 `chunks.jsonl`、Milvus 和 BM25 索引
- `/upload` 走 `src/pipeline/ingest.py`，负责把新 PDF 变成最新索引
- 语义缓存不是旁路逻辑，而是围绕 `run_search()` 和 `run()` 做的包裹层，详见 [cache.md](cache.md)

## 10. 推荐阅读

如果想继续深入这一层，可以按这个顺序看源码：

1. `src/rag_pipeline.py`
2. `src/retrieval.py`
3. `src/pipeline/query_rewrite.py`
4. `src/pipeline/hybrid_retrieve.py`
5. `src/pipeline/rerank.py`
6. `src/pipeline/comparative_rerank.py`
7. `src/pipeline/evidence_check.py`
8. `src/pipeline/answer_generate.py`
