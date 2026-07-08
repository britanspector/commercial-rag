# Embedding 与 Milvus Lite（历史入口）

本文件已整合进 [data-pipeline.md](data-pipeline.md)。

建议统一从下面文档阅读：

- [data-pipeline.md](data-pipeline.md)：向量化、Milvus Lite、BM25 索引的统一说明
- [rag-pipeline.md](rag-pipeline.md)：索引如何被在线检索链路使用

当前向量化入口仍然是：

```bash
python src/embed_chunks.py
```

如果只想做向量库 smoke test，可以继续使用：

```bash
python src/check_milvus.py
```

原文中关于模型、设备、`query: ` 前缀、Milvus Lite 和 Windows 注意事项的内容，已经并入新的数据流水线文档。
