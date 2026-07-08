# 语义缓存设计（历史入口）

本文件已整合进 [cache.md](cache.md)。

建议统一从下面文档阅读：

- [cache.md](cache.md)：缓存设计、接入点、边界与评测
- [rag-pipeline.md](rag-pipeline.md)：缓存所包裹的主问答链路
- [service-ops.md](service-ops.md)：缓存统计接口和服务观测

当前缓存相关实现仍集中在：

- `src/cache/pipeline_bridge.py`
- `src/cache/manager.py`
- `src/cache/index_fingerprint.py`
- `src/cache/invalidate_hooks.py`

原文中的这些内容现在已经并入新的缓存主文档：

- 缓存原则和适用边界
- cache key 与索引指纹
- 命中后的安全校验
- 与评测、审计、前端的关系
- 代码结构和演进路线

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

