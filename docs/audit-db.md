# RAG 审计数据库

每次 `/upload`（同步）、`/search`、`/chat` 的请求与结果写入审计库。`/eval` 与异步入库走内存 Job，结果路径见 `GET /jobs/{id}`。用于追踪、评测对照、引用校验与拒答分析。

## 配置

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `RAG_DATABASE_URL` | SQLAlchemy 连接串 | `sqlite:///data/audit/rag_audit.db` |
| `RAG_AUDIT_ENABLED` | 设为 `0` / `false` 关闭写入 | `1`（开启） |

**PostgreSQL 示例**（需 `pip install psycopg[binary]`）：

```bash
export RAG_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/rag_audit
```

**初始化表**（首次启动 API 会自动建表，也可手动）：

```bash
python -m db.init_db
# 或
cd src && python -c "from db.engine import init_db; init_db()"
```

## 表结构

| 表名 | 用途 |
|------|------|
| `rag_requests` | 每次 API 请求：类型、状态、耗时、错误 |
| `documents` | 文档元数据（上传 upsert） |
| `chunks` | 分块元数据 + 正文预览（≤800 字） |
| `query_logs` | 用户问题、改写后 query、检索配置 |
| `retrieval_hits` | 召回 / 重排片段（rank、分数、页码） |
| `chat_answers` | 最终答案、引用 JSON、`refusal_reason`（码）、`refusal_message`（用户文案）、`evidence_check_json`、`citation_count` |
| `refusal_records` | 拒答专表：`refusal_reason`、`refusal_message`、`evidence_check_json`（供 Refusal Accuracy） |
| `upload_logs` | 上传阶段状态 JSON |

关系：`rag_requests` 1 — 1 `query_logs` / `upload_logs` / `chat_answers`；1 — N `retrieval_hits`。

## 常用 SQL（SQLite）

```sql
-- 最近 10 次问答
SELECT r.id, r.created_at, q.original_query, c.answer, c.refused, c.top_rerank_score
FROM rag_requests r
JOIN query_logs q ON q.request_id = r.id
JOIN chat_answers c ON c.request_id = r.id
WHERE r.request_type = 'chat'
ORDER BY r.id DESC LIMIT 10;

-- 拒答记录（含原因码与用户可见说明）
SELECT id, query_text, refusal_reason, refusal_message, top_rerank_score, refusal_threshold
FROM refusal_records ORDER BY id DESC LIMIT 20;

-- Citation Accuracy：答案引用条数与 evidence 校验
SELECT request_id, citation_count, evidence_passed, citations_json
FROM chat_answers WHERE refused = 0 ORDER BY id DESC LIMIT 10;

-- 某次请求的 Top5 重排片段（含页码）
SELECT rank, chunk_id, page_start, page_end, score_rerank, company_name, section_title
FROM retrieval_hits
WHERE request_id = ? AND stage = 'rerank'
ORDER BY rank;

-- 文档与 chunk 数量
SELECT doc_id, company_name, chunk_count, retrievable_chunk_count FROM documents;
```

## 健康检查

`GET /health` 响应中的 `audit` 字段包含 `enabled`、`backend`、`url_masked`。
