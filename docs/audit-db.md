# 审计数据库（历史入口）

本文件已整合进 [service-ops.md](service-ops.md)。

现在建议统一从 [service-ops.md](service-ops.md) 阅读：

- 审计库的用途
- 服务与后台任务关系
- 默认数据库位置和开关
- 运维与排查路径

当前默认数据库位置仍然是：

- `data/audit/rag_audit.db`

如果你要看实现，优先阅读：

- `src/api/audit.py`
- `src/db/engine.py`
- `src/db/models.py`
- `src/db/tracker.py`
