# FastAPI 接口（历史入口）

本文件已整合进 [service-ops.md](service-ops.md)。

建议统一从 [service-ops.md](service-ops.md) 阅读：

- 服务入口与启动方式
- `/health`、`/cache/stats`、`/upload`、`/search`、`/chat`、`/eval`、`/jobs/{job_id}`
- 后台任务与运维注意事项
- 审计库与部署排查

当前兼容启动方式不变：

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
```
