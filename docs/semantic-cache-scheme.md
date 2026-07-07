# 语义缓存设计（Phase 2）

> **状态**：设计文档 + CacheManager + **Redis L1 精确缓存**（`src/cache/backends/redis.py`）  
> **前置**：现有 RAG Pipeline（`rag_pipeline.py`）、`/chat` `/search`、审计库（`db/tracker.py`）、离线评测（`eval_generation.py` / `eval_retrieval.py`）  
> **目标**：在 **不牺牲引用正确性与拒答契约** 的前提下，为重复/近似问题降低 Embedding、Rerank、LLM 延迟

---

## 1. 当前链路回顾

```
POST /chat
  → RAGPipeline.run()
      ① query_rewrite   （BM25 扩展 + query 向量）
      ② hybrid_retrieve （Milvus + BM25）
      ③ rerank          （bge-reranker-v2-m3）
      ④ evidence_check  （阈值 / 股票 / 意图）
      ⑤ answer_generate （Ollama qwen3:8b + 引用）

POST /search
  → RAGPipeline.run_search()   # ①②③，不含 ④⑤
```

**耗时分布（经验）**：Rerank + LLM 占 `/chat` 大头；Embedding + Milvus/BM25 占 `/search` 与 `/chat` 共同部分。

**已有「缓存」概念（本设计不混用）**：

| 名称 | 位置 | 性质 |
|------|------|------|
| HF 模型权重缓存 | `hf_env.py` | 本地磁盘，非语义缓存 |
| 评测 hybrid pool | `eval_rerank.py` pickle | 离线评测中间结果 |
| MinerU 解析缓存 | `parse_pdf_mineru.py` | 解析幂等，非问答缓存 |
| 审计库 | `data/audit/rag_audit.db` | 请求日志，可扩展 cache 字段 |

---

## 2. 设计原则

1. **缓存的是「已验证的 Pipeline 输出」**，不是裸 LLM 文本。
2. **命中后仍要做轻量安全校验**，不能 blind return。
3. **索引或配置变更即失效**，版本号纳入 cache key。
4. **评测路径默认 bypass**，避免污染 Recall / Citation 指标。
5. **默认关闭**，通过环境变量渐进启用（`RAG_SEMANTIC_CACHE_ENABLED`）。

---

## 3. 适合缓存 vs 不应直接缓存

### 3.1 建议缓存（按优先级）

| 层级 | 范围 | 缓存内容 | 收益 | 风险 |
|------|------|----------|------|------|
| **L1** | `/search` 全链路 | `RAGSearchResult` 序列化（rewrite + recall hits + rerank hits） | 省 Embedding + Milvus + BM25 + Rerank | 索引更新后需失效 |
| **L2** | `/chat` 全链路 | `RAGPipelineResult`（含 evidence_check + citations + answer） | 省 L1 + evidence + LLM | 需严格校验引用 chunk 仍有效 |
| **L3** | 单步 | `query_vector`（rewrite 后） | 省一次 Embedding | 仅精确 key 命中，收益有限 |
| **L4** | 单步 | `(rewrite_key, recall_top_ids)` → rerank hits | 省 Rerank | recall 集合变化则失效 |

**推荐 Phase 2 落地顺序**：L1（search）→ L2（chat，仅 `evidence.passed` 且 `refused=false`）→ 再考虑 L4。

### 3.2 不应直接缓存

| 对象 | 原因 |
|------|------|
| **拒答结果** (`refused=true`) | 阈值、证据规则、索引微调后可能应从拒答变可答；可设极短 TTL 或默认不缓存 |
| **未过 evidence_check 的中间态** | 不满足产品契约 |
| **裸 LLM 生成文本**（无 chunk_id / citation 绑定） | 无法保证 grounded，Corpus 变更后易幻觉 |
| **离线评测** (`eval_generation` / `eval_retrieval` / `POST /eval`) | 指标要求可复现、全链路真实耗时；必须 `cache_bypass` |
| **comparative 题型（默认）** | 实体解析歧义大，语义近似不等于同一对比任务 |
| **upload / ingest 流程** | 写路径，非查询缓存 |
| **动态配置探测请求** | 不同 `recall_route` / `top_k` / `refusal_threshold` 不得共用同一 entry |

### 3.3 条件缓存（需额外标记）

| 场景 | 策略 |
|------|------|
| `query_type=summary` | 可缓存，但 TTL 短于 factual |
| 带 `stock_code` 的问题 | key 必须含 `stock_code`；命中时再校验 stock 约束 |
| 低分 rerank（刚过阈值） | 可不缓存或 TTL 更短，避免边界样本固化 |

---

## 4. Cache Key 与索引版本

### 4.1 逻辑 Key（精确维度）

```
CacheKey =
  scope            # search | chat
  query_normalized # 去空白、统一标点
  stock_code
  query_type
  config_fingerprint
  index_fingerprint
  generation_fingerprint   # 仅 chat scope
```

**config_fingerprint** 应包含：

- `recall_route`, `recall_top_k`, `rerank_top_k`
- `refusal_threshold`, `hybrid_vector_weight`, `hybrid_pool_size`

**index_fingerprint**（Corpus 变更检测）建议：

- `milvus_total_rows` + `bm25_total_chunks`（`UploadResponse` / 索引 manifest 已有）
- `chunks.jsonl` 或 Milvus manifest 的 mtime / hash（见 `data/vector/.../manifest.json`）
- 可选：最近一次 `/upload` 的 `doc_id` 集合 hash

**generation_fingerprint**（仅 chat）：

- `GEN_LLM_MODEL`, `GEN_NUM_CTX`, prompt 模板版本号（后续在 `llm_prompts.py` 增加 `PROMPT_VERSION`）

### 4.2 语义 Key（近似命中）

在逻辑 Key 的 `query_normalized` 之外，额外存储 **query embedding**（与检索同源 `bge-large-zh-v1.5`）：

1. 先用逻辑 Key 做 **精确命中**（O(1) hash）。
2. 未命中时，在 **同 scope + 同 fingerprint** 桶内做 **向量近邻**（cosine ≥ `RAG_SEMANTIC_CACHE_SIM_THRESHOLD`，默认 0.92）。
3. 近邻命中仍须走 **§5 安全校验**；不通过则 miss 并走完整 Pipeline。

**Phase 2 存储**：Redis（KV + 向量索引 HNSW）或 SQLite + 本地向量表；本阶段仅定义接口，不选型落地。

---

## 5. 命中时的安全校验（必做）

缓存命中 **不等于** 跳过所有逻辑。建议按顺序执行：

| # | 检查项 | 说明 | 失败处理 |
|---|--------|------|----------|
| 1 | **index_fingerprint 一致** | Milvus/BM25 行数或 manifest hash 未变 | miss，删除 stale entry |
| 2 | **config / generation fingerprint 一致** | 请求参数与 entry 相同 | miss |
| 3 | **语义相似度 ≥ 阈值** | 仅语义近邻路径 | miss |
| 4 | **chunk 仍存在** | entry 中 `rerank_hits[].chunk_id` / `citations[].chunk_id` 在当前索引可解析 | miss 或降级重跑 retrieve |
| 5 | **evidence 轻量复检**（chat） | 对缓存的 rerank top1 重跑 `check_evidence`（不调 LLM） | 不通过则 miss，禁止返回旧答案 |
| 6 | **stock_code 约束** | 与 `evidence_check` 中 `stock_code_match` 一致 | miss |
| 7 | **拒答标记** | entry 为 `refused=true` 时默认不 serve（或极短 TTL 后重检） | miss |
| 8 | **TTL 未过期** | entry 创建时间 + scope TTL | miss |

**允许跳过**：LLM 生成、完整 hybrid_retrieve、rerank（在校验通过后）。

**/chat 返回契约**：仍通过 `chat_result_to_response()` 序列化；额外 HTTP 头或 audit 字段标记 `cache_hit=true`。

---

## 6. 接入点建议

```
api/main.py
  search() / chat()
       ↓
  cache.lookup(scope, rag_query, pipeline_config)   # 未来
       ↓ miss                    ↓ hit + validate
  pipeline.run_search / run       return cached + audit(cache_hit)
       ↓
  cache.store(...)   # 仅 store 通过校验且 policy 允许的 result
```

**RAGPipeline 内不直接依赖 Redis**：保持 Pipeline 纯函数式；缓存作为 **API 或薄 wrapper**（`src/cache/service.py`，Phase 2 再实现）。

**必须 bypass 的入口**：

- `eval_generation.run_generation_eval`
- `eval_retrieval.run_retrieval_eval`
- `POST /eval` 触发的 job worker
- 单元测试 / 回归对比脚本

实现方式：线程上下文 `cache_bypass=True` 或 env `RAG_SEMANTIC_CACHE_BYPASS=1`。

---

## 7. 与审计、前端、评测的关系

### 7.1 审计库（`db/`）

建议在 `rag_requests` 或 `query_logs` 增加：

- `cache_hit: bool`
- `cache_scope: search | chat | null`
- `cache_similarity: float | null`
- `cache_entry_id: str | null`

便于前端 **缓存监控** 页展示命中率、平均节省延迟。

### 7.2 前端

`/cache` 占位页后续对接：

- 命中率、miss 原因分布（fingerprint 变更 / evidence 失败 / TTL）
- 索引版本与 entry 数量
- 手动 purge（按 doc_id / 全量）

### 7.3 评测

| 评测 | 缓存策略 |
|------|----------|
| `eval_retrieval` | **强制 bypass**；Recall@K / MRR 必须反映真实检索 |
| `eval_generation` | **强制 bypass**；Citation / Refusal 指标依赖完整链路 |
| `eval_ragas` | **强制 bypass**；Faithfulness 需真实 LLM 输出 |

评测 CSV / JSONL 可增加 `cache_hit` 列（恒为 false）以证明路径正确。

---

## 8. TTL 与容量（建议默认值）

| scope | 默认 TTL | 说明 |
|-------|----------|------|
| search | 24h | 检索结果无 LLM 漂移 |
| chat（factual，已回答） | 12h | 研报低频更新 |
| chat（summary） | 4h | 总结题对上下文更敏感 |
| refused | 不缓存或 15min | 避免固化错误拒答 |

容量：LRU + 最大 entry 数（如 10k）；/upload 成功后 **按 doc_id 失效** 相关 entry（包含该 doc chunk 的答案）。

---

## 9. Phase 2 实施路线图（不在本步实现）

| 阶段 | 内容 |
|------|------|
| P2.0 | `src/cache/` 类型 + config + policy（**当前**） |
| P2.1 | 内存 LRU 精确 key 缓存 + `/search` 接入 + audit 字段 |
| P2.2 | Redis + 语义近邻 + `/chat` 接入 + evidence 复检 |
| P2.3 | upload 失效钩子 + 前端缓存监控 + 命中率指标 |

---

## 10. 风险与规避

| 风险 | 规避 |
|------|------|
| 索引更新后返回过期引用 | index_fingerprint + chunk 存在性检查 |
| 近似问法误命中 | 提高阈值；comparative 默认不语义缓存 |
| 评测指标虚高 | eval 路径强制 bypass |
| LLM / Prompt 升级后旧答案 | generation_fingerprint + 版本 bump 清 cache |
| 缓存与 audit 不一致 | store 仅在 Pipeline 成功后；hit 也写 audit |

---

## 11. 代码结构（CacheManager 抽象层）

```
src/cache/
  __init__.py           # 对外导出 CacheManager / 类型 / key_builder
  config.py             # 环境变量、TTL、开关
  types.py              # CacheKey / CacheEntry / CacheStats / InvalidateFilter
  policy.py             # should_cache / should_serve / TTL / 过期判断
  key_builder.py        # 从 RAGQuery + PipelineConfig 构造 CacheQueryContext
  stats.py              # 线程安全统计
  manager.py            # CacheManager：lookup / store / invalidate / stats
  index_fingerprint.py  # 索引版本指纹
  backends/
    base.py             # ExactCacheBackend / SemanticCacheBackend 抽象
    factory.py          # create_exact_backend（Redis 不可用时回退内存）
    memory.py           # L1 内存 LRU
    redis.py            # L1 Redis String + SETEX + TTL
    serialization.py    # CacheEntry JSON 序列化
    semantic.py         # L2 占位 NullSemanticBackend
    milvus_semantic.py  # L2 Milvus 向量语义缓存
  self_test.py          # L1/L2 存储层自测（不依赖 API）
```

### 11.3 Redis L1 精确缓存

**启用：**

```bash
export RAG_SEMANTIC_CACHE_ENABLED=1
export RAG_SEMANTIC_CACHE_L1_BACKEND=redis
export RAG_SEMANTIC_CACHE_REDIS_URL=redis://127.0.0.1:6379/0
# 可选
export RAG_SEMANTIC_CACHE_REDIS_KEY_PREFIX=rag:cache
export RAG_SEMANTIC_CACHE_REDIS_TIMEOUT_S=1.0
export RAG_SEMANTIC_CACHE_TTL_SEARCH_S=86400
```

**Cache Key 组成（`CacheKey.storage_key()`）：**

| 维度 | 字段 | 说明 |
|------|------|------|
| 用户问题 | `query_normalized` | NFKC + 空白归一 |
| metadata 过滤 | `metadata_fingerprint()` | `stock_code` + `query_type` |
| 知识库版本 | `index_fingerprint` | Milvus manifest + chunks/bm25 mtime |
| 检索配置 | `config_fingerprint` | route / top_k / threshold / hybrid 权重 |
| 生成配置 | `generation_fingerprint` | 仅 chat scope（LLM + prompt 版本） |
| 作用域 | `scope` | search / chat |

Redis 物理 key：`{prefix}:l1:{sha256(storage_key)}`，值为 JSON entry，TTL 由 `ttl_for_entry()` 决定。

**降级：** Redis 连接失败或读写异常时，`lookup` 记 miss、`store` 静默跳过，**不阻断 RAG Pipeline**；工厂自动回退 `MemoryExactBackend`。

**日志：** `cache L1 hit` / `cache miss` / `cache store` / `cache L1 reject`（INFO 级）。

**自测：**

```bash
cd commercial-rag
PYTHONPATH=src python -m cache.self_test
# 可选：指定 Redis
PYTHONPATH=src python -m cache.self_test --redis-url redis://127.0.0.1:6379/0
```

**能力清单（L1）：** `ping()` / `lookup` / `store` / `delete` / `invalidate` / `stats_snapshot` / TTL / hit-miss 日志 / Redis 降级。

### 11.4 Milvus L2 语义缓存

**启用：**

```bash
export RAG_SEMANTIC_CACHE_ENABLED=1
export RAG_SEMANTIC_CACHE_L2_BACKEND=milvus
export RAG_SEMANTIC_CACHE_SIM_THRESHOLD=0.92
# 可选
export RAG_SEMANTIC_CACHE_L2_MILVUS_DB=data/vector/semantic_cache/milvus.db
export RAG_SEMANTIC_CACHE_L2_VECTOR_DIM=1024
export RAG_SEMANTIC_CACHE_L2_SEARCH_TOP_K=5
```

**存储字段：** query 向量、原始问题、改写问题、`metadata_filters`（公司/年份/doc/版本）、payload、chunk_ids、`index_fingerprint` 等。

**查询流程：**

1. L1 精确 miss 后，在同 `semantic_bucket_hash` 桶内做向量近邻（Milvus COSINE）；
2. 相似度 ≥ `RAG_SEMANTIC_CACHE_SIM_THRESHOLD`；
3. `validate_semantic_metadata()`：校验 stock_code / company / report_year / doc_id / doc_version；
4. `should_serve_cached()`：index/config/generation fingerprint + TTL。

**降级：** Milvus 不可用时回退 `NullSemanticBackend`（L2 始终 miss，不中断 Pipeline）。

**日志：** `cache L2 hit` / `cache L2 reject` / `cache miss`（INFO 级）。

### 11.1 Pipeline 接入（读写闭环）

`RAGPipeline.run_search()` / `run()` 已内置缓存编排（**不在 API 层重复处理**）：

1. 构建 `CacheQueryContext`（含 config/index/metadata fingerprint）
2. `CacheManager.lookup()` → L1 Redis 精确 → L2 Milvus 语义
3. 两级 miss 时执行原始 Pipeline（L2 尝试时已做 `query_rewrite`，可复用向量）
4. 成功后 `store()` 写入 L1 + L2

返回结果附带 `CacheInfo`：

| 字段 | 说明 |
|------|------|
| `hit` | 是否命中缓存 |
| `source` | `l1_exact` / `l2_semantic` / `pipeline` |
| `similarity` | L2 语义相似度（L1 为 1.0） |
| `reason` | `served` / `not_found` / `cache_disabled` / `cache_bypass` |

评测路径请传 `use_cache=False` 或设置 `RAG_SEMANTIC_CACHE_BYPASS=1`。

### 11.2 CacheManager 用法（API 直接调用时）

```python
from cache import (
    CacheScope,
    build_query_context,
    build_config_fingerprint_from_pipeline,
    build_generation_fingerprint_from_env,
    extract_chunk_ids,
    get_cache_manager,
)

manager = get_cache_manager()
ctx = build_query_context(
    scope=CacheScope.SEARCH,
    rag_query=rag_query,
    config_fingerprint=build_config_fingerprint_from_pipeline(pipeline.config),
    query_embedding=rewrite.query_vector,  # 可选，供未来 L2
)

result = manager.lookup(ctx)
if result.hit:
    return result.entry.payload

pipeline_result = pipeline.run_search(rag_query)
manager.store(
    ctx,
    payload=pipeline_result.to_dict(),
    refused=False,
    top_rerank_score=pipeline_result.top_rerank_score,
    chunk_ids=extract_chunk_ids(pipeline_result.to_dict()),
)
```

### 11.3 接口一览

| 方法 | 职责 |
|------|------|
| `lookup(context)` | L1 精确 → L2 语义；policy + TTL 校验 |
| `store(context, payload, ...)` | policy 允许时写入 L1 + L2 |
| `delete(context)` | 删除单条 L1 精确 entry |
| `invalidate(filter)` | 按 scope / doc_id / index_fp 失效 L1+L2 |
| `invalidate_for_upload(doc_id)` | upload 后失效 |
| `ping()` / `stats_snapshot()` / `describe()` | 连通性、命中率、entry 数量 |

Pipeline **不直接依赖** Redis/Milvus；仅通过 `get_cache_manager()` 调用上述接口。

---

## 12. 安全约束与失效机制

### 12.1 版本 / 配置指纹

| 指纹 | 触发失效场景 | 组成 |
|------|-------------|------|
| `index_fingerprint` | upload、Milvus/BM25/chunks 变更 | milvus 行数 + chunks/docs 行数 + mtime + sha16 |
| `config_fingerprint` | recall/rerank/refusal/hybrid 参数变化 | route / top_k / threshold / pool |
| `generation_fingerprint` | LLM / prompt / ctx 变化（仅 chat） | model + PROMPT_VERSION + num_ctx + num_predict |
| `metadata_filter_fingerprint` | stock / 公司 / 年份 / doc 过滤不一致 | `CacheMetadataFilters.fingerprint()` |

`PROMPT_VERSION` 默认 = system prompt 模板 sha256 前 12 位；可设 `GEN_PROMPT_VERSION` 手动 bump。

### 12.2 命中后安全校验（`cache/safety.py`）

L1 / L2 统一走 `validate_entry_safety()`：

1. metadata filter 一致（防跨公司 / 跨年份 / 跨文档）
2. index / config / generation fingerprint 一致
3. 语义相似度 ≥ 阈值（L2）
4. 引用 `chunk_id` 仍在 `chunks.jsonl` 中（防文档删除后返回过期引用）

校验失败且 entry 不可信 → 自动 purge（`STALE_ENTRY_REASONS`）。

### 12.3 失效钩子（`cache/invalidate_hooks.py`）

| 事件 | 调用 | 行为 |
|------|------|------|
| 进程首次 lookup | `ensure_fingerprint_sync()` | 索引/生成配置变化 → 批量失效 |
| PDF upload 成功 | `on_corpus_updated(doc_id)` | 按 doc_id 失效 + 刷新 chunk 注册表 |
| 手动 purge | `invalidate_all_caches()` | 全量失效 |

`pipeline/ingest.py` 入库成功后自动调用 `on_corpus_updated()`。

### 12.4 自测

```bash
PYTHONPATH=src python -m cache.self_test
```

含 index fingerprint 不一致、stock 隔离、report_year 拒绝、chunk/TTL 等用例。

---

## 13. 可观测性与性能评估

### 13.1 单次请求遥测（`cache/telemetry.py`）

每次 `run_search` / `run` 记录并输出结构化 INFO 日志，字段包括：

| 字段 | 说明 |
|------|------|
| `hit` / `source` | 是否命中、来源（l1_exact / l2_semantic / pipeline） |
| `similarity` | L2 语义相似度 |
| `safety_ok` / `safety_reason` | 安全校验结果 |
| `latency_ms` / `lookup_ms` / `pipeline_ms` | 总耗时、缓存查询、Pipeline 执行 |
| `vector_retrieval` | 本次是否执行向量检索 |
| `llm_called` | 本次是否调用 LLM（chat） |

响应体 `cache` 字段（`CacheInfoResponse`）同步返回上述遥测，供前端展示。

**日志示例：**

```
cache request scope=search hit=True source=l1_exact similarity=- safety_ok=True reason=served latency_ms=15.2 lookup_ms=1.8 pipeline_ms=0.0 vector_retrieval=False llm_called=False stock=688008 query='澜起科技 EPS'
```

### 13.2 累计统计（`CacheStats` / `CacheStatsCollector`）

| 指标 | 说明 |
|------|------|
| `l1_hit_rate` / `l2_hit_rate` / `total_hit_rate` | 请求级命中率 |
| `lookup_hit_rate` | lookup 级命中率（含 L1/L2 两次尝试） |
| `avg_hit_latency_ms` / `avg_miss_latency_ms` | 命中 vs 未命中平均延迟 |
| `avg_latency_saved_ms` | 命中相对未命中节省的平均延迟 |
| `vector_retrievals_saved` | 因缓存跳过的向量检索次数 |
| `llm_calls_saved` / `llm_call_reduction_rate` | 因缓存减少的 LLM 调用 |
| `safety_rejects` | 安全校验拒绝次数 |

### 13.3 API 端点

| 端点 | 说明 |
|------|------|
| `GET /health` | `cache` 字段含累计统计摘要 |
| `GET /cache/stats` | 完整缓存统计 + L1/L2 后端状态 |

```bash
curl -s http://127.0.0.1:8000/cache/stats | jq .
```

### 13.4 评测路径

评测脚本传 `use_cache=False` 或 `RAG_SEMANTIC_CACHE_BYPASS=1`，遥测中 `reason=cache_bypass`，不计入命中率分子（记为 bypass 请求）。

---

## 14. 三模式缓存评测

### 14.1 脚本与模式

独立脚本 `src/eval_cache.py`，对比 **cache_off / l1_only / l1_l2** 三模式：

| 模式 | 说明 |
|------|------|
| `cache_off` | 关闭缓存，cold run 基线 |
| `l1_only` | 仅 L1 精确缓存（memory L1） |
| `l1_l2` | L1 + L2 语义缓存（measure_l2 阶段用 paraphrase 问法测 L2） |

```bash
cd commercial-rag
PYTHONPATH=src python src/eval_cache.py --dry-run
PYTHONPATH=src python src/eval_cache.py --limit 12 --modes off,l1,l1l2 --skip-ragas
PYTHONPATH=src python src/eval_cache.py --modes off,l1,l1l2 --skip-ragas   # 全量 150 题
```

### 14.2 产出文件

| 文件 | 内容 |
|------|------|
| `data/eval/eval_cache_results.csv` | 每题每阶段明细 |
| `data/eval/eval_cache_comparison.csv` | 模式聚合对比 |
| `data/eval/eval_cache_report.md` | 延迟/命中率/质量 + 第二阶段完成标准 |

完整结果摘要见 [eval-cache-results.md](eval-cache-results.md)。

### 14.3 第二阶段完成标准（7 项）

评测脚本 `verify_completion_criteria()` 对标准 #2/#3 仅统计**可缓存题**（排除 `comparative`，该题型默认不写入缓存，见 `cache/policy.py`）。

1. Pipeline 可在缓存开启/关闭间切换  
2. 完全相同问题命中 L1（可缓存题 ≥95%）  
3. 语义相近问题在安全条件下命中 L2（可缓存题 ≥85%）  
4. 版本/metadata/配置变化不误命中  
5. 每次请求记录 cache_source/hit/耗时/相似度/拒绝原因  
6. 可用第一阶段评测集对比延迟/命中率/质量  
7. 缓存开启后 Citation/Refusal 不明显下降（Δ ≤ 0.05）

---

## 15. 生产部署与前端对接

### 15.1 生产启用 Redis L1

推荐环境变量（见项目根 `.env.example`）：

```bash
export RAG_SEMANTIC_CACHE_ENABLED=1
export RAG_SEMANTIC_CACHE_L1_BACKEND=redis
export RAG_SEMANTIC_CACHE_REDIS_URL=redis://127.0.0.1:6379/0
export RAG_SEMANTIC_CACHE_REDIS_KEY_PREFIX=rag:cache
export RAG_SEMANTIC_CACHE_REDIS_TIMEOUT_S=1.0
```

Redis 不可用时自动回退 `MemoryExactBackend`，不阻断 RAG Pipeline。

**验证：**

```bash
PYTHONPATH=src python -m cache.self_test
PYTHONPATH=src python -m cache.self_test --redis-url redis://127.0.0.1:6379/0
```

### 15.2 前端展示

| 位置 | 数据来源 | 展示内容 |
|------|----------|----------|
| `/cache` 缓存监控页 | `GET /cache/stats` | 累计 L1/L2 命中率、平均延迟、资源节省、后端状态 |
| `/search`、`/chat` 结果区 | 响应体 `cache` 字段 | 本次请求命中/来源/延迟/相似度/安全校验 |
| 概览页后端状态卡 | `GET /health` → `cache` | 缓存启用状态、累计命中率与平均延迟摘要 |

启动后端时加载上述环境变量，前端通过 Vite 代理 `/api` 访问即可。

