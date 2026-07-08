# Rerank 与生成说明（历史入口）

本文件已拆分整合到两篇主文档中：

- [rag-pipeline.md](rag-pipeline.md)：在线检索、Rerank、证据校验和回答生成
- [evaluation.md](evaluation.md)：Rerank 评测、生成评测和结果文件

当前相关入口仍然是：

```bash
python src/eval_rerank.py
python src/rag_chat.py "京仪装备2026E毛利率预测是多少？"
```

原文中的这些内容现在分别在新文档维护：

- Rerank 在主链路中的位置
- 引用生成和拒答逻辑
- Rerank 评测命令与结果文件
- 关键参数和默认阈值
