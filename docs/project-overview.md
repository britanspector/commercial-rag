# 项目总览

`commercial-rag` 是一个面向中文金融研报的 RAG 工程项目。它的重点不只是“问答能不能跑通”，而是把知识库构建、检索优化、拒答契约、缓存、评测、API 和前端控制台组织成一条可复用的完整链路。

## 项目目标

项目要解决的问题是：把大量金融研报 PDF 转成可检索、可引用、可评测的知识库，并对外提供统一的检索与问答服务。

当前默认链路：

```text
PDF -> MinerU 解析 -> 分块 -> 向量索引 + BM25 索引
    -> query_rewrite -> hybrid_retrieve -> rerank
    -> evidence_check -> answer_generate
```

## 当前状态

- 数据规模：`200` 份研报，四个行业各 `50` 份
- 知识库规模：`10,263` 个 chunk，其中 `7,382` 个可检索
- 评测集：`150` 题，位于 `data/eval/eval_questions.jsonl`
- 当前结果：Hybrid `Recall@10 = 92.0%`，Rerank 管线事实准确率 `88.0%`

结果快照见 [eval-results.md](eval-results.md)，问题分析见 [eval-badcase-analysis.md](eval-badcase-analysis.md)。

## 代码主线

项目有两条主线：离线知识库构建和在线查询服务。

### 1. 离线知识库构建

这一部分把 `data/raw_pdfs/` 下的研报变成可检索索引。

- `src/parse_pdf_mineru.py`：调用 MinerU 解析 PDF
- `src/chunk_mineru.py`：把解析结果切成可检索 chunk
- `src/embed_chunks.py`：将 chunk 写入 Milvus Lite
- `src/build_bm25_index.py`：从 chunk 构建 BM25 索引
- `src/pipeline/ingest.py`：面向上传场景的单文档增量入库

### 2. 在线检索与问答

这一部分把问题送入统一的 RAG 流水线。

- `src/rag_pipeline.py`：统一编排 `query_rewrite -> hybrid_retrieve -> rerank -> evidence_check -> answer_generate`
- `src/retrieval.py`：向量、BM25 和 Hybrid 三路召回
- `src/pipeline/`：拆分后的主流程步骤实现
- `src/api/main.py`：FastAPI 服务入口
- `src/rag_chat.py`：CLI 问答入口

### 3. 横切能力

- `src/cache/`：语义缓存、失效和统计
- `src/eval_*.py`：检索、Rerank、生成、RAGAS 评测
- `src/db/`：审计持久化
- `frontend/`：上传、问答、检索调试、评测、缓存监控控制台

## 数据目录

`data/` 是项目的运行数据区，当前包含这些一级目录：

- `data/raw_pdfs/`：原始研报 PDF
- `data/parsed/`：解析结果、文档清单、分块结果和统计
- `data/vector/`：Milvus Lite 和 BM25 索引
- `data/eval/`：评测集、评测结果和评测日志
- `data/audit/`：审计数据库
- `data/logs/`：API、前端和模型服务日志

## 前端功能地图

前端控制台目前围绕这些页面组织：

- 概览
- PDF 上传 / 知识库构建
- RAG 问答
- 检索调试
- 自动化评测
- 缓存监控
- Agent Trace / Multi-Agent 工作流预留页

对应导航定义在 `frontend/src/navigation.tsx`。

## 推荐阅读顺序

1. [data-pipeline.md](data-pipeline.md)：先理解知识库是怎么构建出来的
2. [rag-pipeline.md](rag-pipeline.md)：再理解在线检索和回答链路
3. [evaluation.md](evaluation.md)：然后看项目如何验证效果
4. [cache.md](cache.md)：最后看缓存如何接入主链路
5. [service-ops.md](service-ops.md)：需要 API、部署和排障时再深入

## 当前重点与后续方向

- 对比题、多主体召回和证据覆盖仍是重点优化方向
- 生成侧还在继续优化报告期对齐、精确数字摘录和拒答策略
- 数据规模进一步扩大后，需要评估更成熟的 Milvus 部署方式
- Agent 化能力已开始设计，路线图见 [agent-architecture.md](agent-architecture.md)
