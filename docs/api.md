# FastAPI 接口

入口：`src/rag_api.py` → `uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src`

## 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务与 Pipeline 状态 |
| POST | `/upload` | PDF 入库（解析→分块→向量→Milvus+BM25） |
| POST | `/search` | 检索+重排（不生成答案） |
| POST | `/chat` | 完整 RAG（需 Ollama） |
| POST | `/eval` | 异步批量评测 |
| GET | `/jobs/{job_id}` | 查询上传/评测任务状态 |

## 请求体（/search、/chat）

```json
{
  "query": "澜起科技2026年EPS预测是多少？",
  "stock_code": "688008",
  "query_type": "factual",
  "recall_route": "hybrid"
}
```

## 异步入库

`POST /upload` 表单增加 `background=true`，立即返回 `job_id`，轮询 `GET /jobs/{job_id}`。

## 批量评测（等价 CLI）

`POST /eval` 示例：

```json
{
  "eval_type": "generation",
  "skip_ragas": true,
  "save_detail": true
}
```

| eval_type | 等价命令 |
|-----------|----------|
| `generation` | `python src/eval_generation.py --skip-ragas --save-detail` |
| `retrieval` | `python src/eval_retrieval.py --route hybrid` |
| `ragas` | `python src/eval_ragas.py --resume` |

## 审计

默认写入 `data/audit/rag_audit.db`（可用 `RAG_AUDIT_ENABLED=0` 关闭）。详见 [audit-db.md](audit-db.md)。
