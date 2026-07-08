# 服务接口与运维

本文档统一说明 API、后台任务、审计库以及迁移部署相关信息。

## 1. 服务入口

兼容入口：

- `src/rag_api.py`

实际应用实现：

- `src/api/main.py`

常用启动方式：

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
```

## 2. API 能力

当前主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务、模型、审计和缓存状态 |
| `GET` | `/cache/stats` | 语义缓存统计 |
| `POST` | `/upload` | PDF 入库 |
| `POST` | `/search` | 检索 + 重排，不生成答案 |
| `POST` | `/chat` | 完整 RAG 问答 |
| `POST` | `/eval` | 异步启动批量评测 |
| `GET` | `/jobs/{job_id}` | 查询后台任务状态 |

其中：

- `/search` 和 `/chat` 复用统一 `RAGPipeline`
- `/upload` 复用 `src/pipeline/ingest.py`
- `/eval` 通过任务系统异步调用评测脚本

## 3. 后台任务

相关文件：

- `src/api/jobs.py`

后台任务主要服务两类场景：

- `POST /upload?background=1`
- `POST /eval`

设计目标：

- 请求快速返回 `job_id`
- 长耗时操作放到后台执行
- 前端或调用方通过 `GET /jobs/{job_id}` 轮询状态和输出路径

## 4. 审计数据库

相关文件：

- `src/api/audit.py`
- `src/db/engine.py`
- `src/db/models.py`
- `src/db/tracker.py`

默认数据库：

- `sqlite:///data/audit/rag_audit.db`

开关：

- `RAG_AUDIT_ENABLED=0` 可关闭

审计库记录了这些核心对象：

- 请求主表
- 文档元数据
- chunk 元数据与正文预览
- 查询日志
- 检索 / 重排命中片段
- 最终答案与引用
- 拒答记录
- 上传阶段状态

作用包括：

- 排查服务问题
- 对照引用和拒答行为
- 支持缓存和前端观测
- 为后续分析提供结构化日志

## 5. 健康检查

`GET /health` 会汇总这些状态：

- pipeline 是否已初始化
- 模型是否已加载
- 审计库是否可用
- 缓存是否启用
- 默认检索参数

适合用来做：

- 服务存活检查
- 启动后自检
- 部署后的基础验收

## 6. 常见运行方式

### 本地问答服务

```bash
conda activate commercial-rag
pip install -r requirements.txt
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
```

### 批量评测

```bash
python src/eval_retrieval.py --compare-routes --top-k 10
python src/eval_rerank.py
python src/eval_generation.py --skip-ragas --save-detail
```

### 增量入库

推荐通过 API 的 `/upload` 或直接调用 `src/pipeline/ingest.py` 对单文档进行处理。

## 7. 迁移与打包

迁移到新环境时，最关键的是区分“必须同步的索引资产”和“可以重建的中间产物”。

通常至少需要关注这些文件：

- `data/vector/milvus.db`
- `data/vector/bm25_index.pkl`
- `data/parsed/chunks.jsonl`
- `data/parsed/doc_manifest.jsonl`
- `data/eval/eval_questions.jsonl`

如果要继续做生成评测或 RAGAS，还应保留：

- `data/eval/eval_generation_detail.jsonl`

如果服务器只做评测，不重新解析 PDF，则这些目录不是必须：

- `data/raw_pdfs/`
- `data/parsed/mineru/`

项目里已经提供打包脚本：

- `scripts/pack_for_autodl.sh`
- `scripts/pack_for_autodl.ps1`

## 8. 运行注意事项

- 评测和服务不要同时争用 `milvus.db`
- 生成评测依赖本地 Ollama
- RAGAS 依赖额外 LangChain / ragas 版本栈，建议单独管理
- 如果环境是 AutoDL 或远程 GPU 机器，优先确认 PyTorch 和模型缓存配置

## 9. 典型排查路径

### 服务起不来

先看：

- `/health`
- `data/logs/`
- 审计库是否可写
- 模型依赖是否齐全

### 检索或问答报错

优先判断：

- Milvus / BM25 索引文件是否存在
- 当前环境是否内存不足
- 生成场景下 Ollama 是否已启动

### 上传或评测卡住

优先检查：

- 后台 `job_id` 状态
- 是否有其他进程占用 `milvus.db`
- 长耗时阶段是在解析、Rerank 还是生成

## 10. 相关文档

- [project-overview.md](project-overview.md)：整体入口
- [data-pipeline.md](data-pipeline.md)：离线构建和索引
- [evaluation.md](evaluation.md)：评测脚本和结果文件
- [cache.md](cache.md)：缓存与观测
