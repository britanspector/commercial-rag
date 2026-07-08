# 分块说明（历史入口）

本文件已整合进 [data-pipeline.md](data-pipeline.md)。

建议改为从下面入口阅读：

- [data-pipeline.md](data-pipeline.md)：包含分块在内的完整离线知识库构建链路
- [rag-pipeline.md](rag-pipeline.md)：解释这些 chunk 如何参与检索、Rerank 和回答生成

当前分块入口仍是：

```bash
python src/chunk_mineru.py
```

原文里关于这些内容的说明，已经并入新文档：

- `mineru_paragraph_v3` 分块策略
- 表格块和噪声块处理
- `rating_headline`、`comparable_table` 等 P2 增强
- `chunks.jsonl` 的关键字段
