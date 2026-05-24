# Embedding 与 Milvus Lite 向量库

## 作用

将 `chunks.jsonl` 用 **bge-large-zh-v1.5** 向量化，写入本地 **Milvus Lite** 数据库。

## 安装

```bash
conda activate commercial-rag
pip install -r requirements-embed.txt
```

## 运行

```bash
# 1. 向量化并写入 Milvus Lite
python src/embed_chunks.py

# 2. 检索 smoke test
python src/check_milvus.py
python src/check_milvus.py "华峰测控2025年净利润是多少"
```

## 输入 / 输出

| 路径 | 说明 |
|------|------|
| `data/parsed/chunks.jsonl` | 分块结果（输入） |
| `data/vector/milvus.db` | Milvus Lite 本地库 |
| `data/parsed/embed_summary.csv` | 向量化统计 |

## 配置（`src/embed_chunks.py`）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBED_MODEL` | `BAAI/bge-large-zh-v1.5` | 1024 维中文向量模型 |
| `EMBED_DEVICE` | `auto` | `cuda` / `cpu`；GPU 显存不足时改为 `cpu` |
| `EMBED_BATCH_SIZE` | 8 | 批大小 |
| `BGE_QUERY_PREFIX` | `query: ` | 查询侧 instruction（bge 官方推荐） |

## Windows 说明

- 需单独安装：`pip install milvus-lite`
- 写入时**不要频繁 flush**；脚本在 embedding 完成后再打开 Milvus 写入
- 检索时先 encode 查询，再打开 Milvus（`check_milvus.py` 已按此顺序）

## 检索说明

- **文档入库**：直接 encode chunk 正文
- **用户查询**：加 `query: ` 前缀后再 encode
- **相似度**：COSINE（向量已 L2 归一化）

Milvus collection 字段：`chunk_id`, `doc_id`, `filename`, `section_title`, `text`, `page_start`, `page_end`, `contains_table`, `token_count`, `stock_code` 等。

## 后续扩展

- 混合检索：BM25 + 向量
- 重排：`bge-reranker-v2-m3`
- 迁移到 Milvus Standalone：只需把 `MilvusClient` 的 uri 改为远程地址
