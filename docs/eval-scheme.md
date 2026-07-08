# 检索评测说明（历史入口）

本文件已整合进 [evaluation.md](evaluation.md)。

建议改为从下面文档阅读：

- [evaluation.md](evaluation.md)：统一的检索、Rerank、生成和 RAGAS 评测体系
- [eval-results.md](eval-results.md)：当前评测结果快照
- [eval-badcase-analysis.md](eval-badcase-analysis.md)：当前主要失败面和后续优化方向

如果你只想运行检索评测，当前入口仍然是：

```bash
python src/eval_retrieval.py --compare-routes --top-k 10
```

原文中的这些内容已并入新的评测主文档：

- 150 题评测集结构
- `vector / bm25 / hybrid` 三路召回说明
- 输出文件与指标定义
- 检索评测与生成评测之间的关系
