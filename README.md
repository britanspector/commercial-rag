# commercial-rag

面向中文金融研报的 RAG 工程项目，覆盖从 PDF 解析、知识库构建到检索、问答、缓存、评测和服务化的完整链路。

当前默认主线是：**PDF 解析 -> 分块 -> 向量 + BM25 索引 -> 混合召回 -> Rerank -> 证据校验 -> 引用生成 / 拒答**。

## 当前能力

- 数据规模：`200` 份研报，覆盖半导体、电力、互联网电商、白色家电四个行业
- 知识库规模：`10,263` 个 chunk，其中 `7,382` 个可检索
- 评测集：`150` 题，位于 `data/eval/eval_questions.jsonl`
- 当前结果：Hybrid `Recall@10 = 92.0%`，Rerank 管线事实准确率 `88.0%`

完整结果见 [docs/eval-results.md](docs/eval-results.md)。

## 系统主链路

```text
原始 PDF
  -> MinerU 解析
  -> 结构化分块
  -> 向量索引 + BM25 索引
  -> query_rewrite
  -> hybrid_retrieve
  -> rerank
  -> evidence_check
  -> answer_generate
```

项目里有两条互相衔接的主线：

1. 离线知识库构建：把 `data/raw_pdfs/` 下的研报转成 `chunks.jsonl`、Milvus Lite 和 BM25 索引。
2. 在线查询服务：通过统一的 `RAGPipeline` 复用检索、重排、证据校验和回答生成逻辑。

对应的核心入口：

- `src/parse_pdf_mineru.py`：PDF 解析
- `src/chunk_mineru.py`：分块
- `src/embed_chunks.py`：向量化
- `src/build_bm25_index.py`：BM25 索引
- `src/rag_pipeline.py`：统一 RAG 编排
- `src/pipeline/ingest.py`：单份 PDF 增量入库
- `src/api/main.py`：FastAPI 服务
- `frontend/src/navigation.tsx`：前端功能导航

## 快速开始

### 最小体验路径

如果索引已经准备好，可以直接体验 CLI 或 API：

```bash
conda activate commercial-rag
pip install -r requirements.txt

python src/rag_chat.py "京仪装备2026E毛利率预测是多少？"

uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
```

### 全量构建路径

如果要从原始 PDF 重建知识库，按下面顺序执行：

```bash
conda activate commercial-rag

# 先按本机 GPU/驱动/CUDA 选择合适的 PyTorch 版本
# 例如：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

python src/parse_pdf_mineru.py
python src/chunk_mineru.py
python src/embed_chunks.py
python src/build_bm25_index.py
```

更细的环境配置、GPU 注意事项、断点续跑和输出文件说明，请看 [docs/data-pipeline.md](docs/data-pipeline.md)。

## 常见使用方式

```bash
# 检索评测
python src/eval_retrieval.py --compare-routes --top-k 10

# Rerank 评测
python src/eval_rerank.py

# 全链路生成评测
python src/eval_generation.py --skip-ragas --save-detail

# 基于 detail 文件补跑 RAGAS
python src/eval_ragas.py --resume
```

更完整的评测说明见 [docs/evaluation.md](docs/evaluation.md)。

## 项目结构

```text
commercial-rag/
├── data/       # 原始 PDF、解析产物、索引、评测数据、日志与审计库
├── src/        # 核心后端实现：解析、索引、检索、RAG、缓存、API、评测
├── frontend/   # 前端控制台：上传、检索调试、问答、评测、缓存监控
├── docs/       # 按项目主线整理后的说明文档
├── scripts/    # 评测集构建、实验脚本、打包脚本
└── requirements.txt
```

`data/` 的详细含义见 [docs/project-overview.md](docs/project-overview.md) 和 [docs/data-pipeline.md](docs/data-pipeline.md)。

## 文档导航

按阅读顺序建议先看这些文档：

- [docs/project-overview.md](docs/project-overview.md)：项目定位、代码入口、数据目录和运行方式总览
- [docs/data-pipeline.md](docs/data-pipeline.md)：从 PDF 到索引的离线知识库构建链路
- [docs/rag-pipeline.md](docs/rag-pipeline.md)：在线检索、Rerank、证据校验和回答生成
- [docs/evaluation.md](docs/evaluation.md)：检索、Rerank、生成和 RAGAS 评测体系
- [docs/eval-results.md](docs/eval-results.md)：当前结果快照
- [docs/eval-badcase-analysis.md](docs/eval-badcase-analysis.md)：Badcase 与后续优化方向
- [docs/cache.md](docs/cache.md)：语义缓存设计、接入点和评测结果
- [docs/service-ops.md](docs/service-ops.md)：API、审计库、后台任务、部署与迁移
- [docs/agent-architecture.md](docs/agent-architecture.md)：未来 Agent 化路线图

## 当前边界与后续方向

- 目前主要面向中文金融研报场景，知识库和评测集都围绕该领域构建。
- 生成评测依赖本地 Ollama，RAGAS 栈需要额外依赖和模型服务。
- 对比题、多实体召回、报告期对齐和拒答策略仍是持续优化重点。
- 当数据规模继续扩大时，需要评估 Milvus Standalone 与更成熟的向量索引方案。
