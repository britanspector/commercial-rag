# 数据与索引流水线

本文档说明项目如何把原始研报 PDF 转成可检索知识库。主线分为四步：PDF 解析、分块、向量化、BM25 索引构建。

## 整体流程

```text
data/raw_pdfs/*.pdf
  -> parse_pdf_mineru.py
  -> data/parsed/mineru/ + documents.jsonl + doc_manifest.jsonl
  -> chunk_mineru.py
  -> data/parsed/chunks.jsonl
  -> embed_chunks.py
  -> data/vector/milvus.db
  -> build_bm25_index.py
  -> data/vector/bm25_index.pkl
```

## 1. PDF 解析

入口文件：

- `src/parse_pdf_mineru.py`
- `src/pdf_paths.py`

职责：

- 扫描 `data/raw_pdfs/<industry>/`
- 调用 MinerU 解析 PDF
- 生成 Markdown、结构化中间文件和文档清单

主要输出：

- `data/parsed/mineru/<doc_id>/`：MinerU 原始输出
- `data/parsed/documents.jsonl`：每份 PDF 一行的全文 Markdown
- `data/parsed/parse_summary.csv`：解析统计
- `data/parsed/doc_manifest.jsonl`：文档清单和来源路径

常见注意事项：

- MinerU 的设备通过环境变量生效，不是旧版 CLI 参数
- GPU 是否真正启用，取决于本机 PyTorch 是否是 CUDA 版
- `mineru/` 目录体积较大，适合保留在构建环境，不一定需要随评测环境分发

## 2. 分块

入口文件：

- `src/chunk_mineru.py`

职责：

- 读取 MinerU 解析结果
- 按内容类型和 token 预算切成适合检索的小块
- 标记可检索与不可检索内容

当前主策略是 `mineru_paragraph_v3`。

### 正文块

- 按段落合并，优先在句号、分号等边界切分
- 长文本允许少量 overlap
- `embedding_text` 上限约 `512 tokens`

### 表格块

- 长表按行组拆分
- 每个子块重复必要的表头、单位和元数据
- `embedding_text` 会加入自然语言化描述，便于向量检索
- 同一张表的子块共享 `table_id`

### 噪声处理

- 免责声明、分析师信息、链接等标为 `is_retrievable=false`
- 风险提示等业务相关正文仍可检索

### P2 增强

当前索引已经包含这些增强：

- `rating_headline`：封面/摘要中的评级短块
- `comparable_table`：可比公司表标记
- 表格指标语义化：把 EPS、归母净利润等指标写入 `embedding_text`
- 附录合并与 chunk id 重排

关键字段：

- `embedding_text`
- `content_type`
- `is_retrievable`
- `table_raw`
- `table_id`
- `company_name`
- `stock_code`
- `report_title`
- `report_date`
- `rating`

主要输出：

- `data/parsed/chunks.jsonl`
- `data/parsed/chunk_summary.csv`

## 3. 向量化与 Milvus Lite

入口文件：

- `src/embed_chunks.py`
- `src/milvus_store.py`

职责：

- 读取 `chunks.jsonl`
- 使用 `BAAI/bge-large-zh-v1.5` 生成向量
- 写入本地 `Milvus Lite`

当前默认配置：

- 向量模型：`BAAI/bge-large-zh-v1.5`
- 向量维度：`1024`
- 相似度：`COSINE`
- 查询侧前缀：`query: `

主要输出：

- `data/vector/milvus.db`
- `data/parsed/embed_summary.csv`

## 4. BM25 索引

入口文件：

- `src/build_bm25_index.py`
- `src/bm25_store.py`

职责：

- 从 `chunks.jsonl` 构建词法检索索引
- 为 Hybrid 检索提供 BM25 分数

主要输出：

- `data/vector/bm25_index.pkl`

## 5. 增量入库

入口文件：

- `src/pipeline/ingest.py`

它把“解析 -> 分块 -> 向量化 -> Milvus + BM25 更新”封装成单文档入库能力，服务于 `POST /upload` 和后续在线知识库扩充场景。

与离线全量脚本的区别：

- 复用现有流水线逻辑
- 按 `doc_id` 替换对应 JSONL 记录和 chunk
- 增量更新索引，而不是重建全库

## 6. 运行顺序

```bash
conda activate commercial-rag

# 按本机环境先安装 CPU 或 CUDA 版 PyTorch
pip install -r requirements.txt

python src/parse_pdf_mineru.py
python src/chunk_mineru.py
python src/embed_chunks.py
python src/build_bm25_index.py
```

如果只验证现有索引是否可用，可以跳过前两步，只运行检查脚本或直接进入 `rag_chat.py` / API。

## 7. 数据产物速查

| 路径 | 作用 |
|------|------|
| `data/raw_pdfs/` | 原始 PDF |
| `data/parsed/mineru/` | MinerU 原始解析结果 |
| `data/parsed/documents.jsonl` | 每份 PDF 的完整 Markdown |
| `data/parsed/doc_manifest.jsonl` | 文档清单 |
| `data/parsed/chunks.jsonl` | 检索主数据 |
| `data/vector/milvus.db` | 向量库 |
| `data/vector/bm25_index.pkl` | BM25 索引 |

## 8. 索引扩展与限制

当前项目默认使用 Milvus Lite，因为它适合本地研发、单机实验和中等规模知识库。

如果未来数据量继续增加，需要关注：

- Milvus Lite 的本地单机特性
- 更复杂的索引类型实验
- 迁移到 Milvus Standalone 的部署成本与收益

相关背景见历史文档 `milvus-index-comparison.md`。
