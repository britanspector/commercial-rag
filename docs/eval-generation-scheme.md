# 生成质量评测（历史入口）

本文件已整合进 [evaluation.md](evaluation.md)。

建议统一从下面文档阅读：

- [evaluation.md](evaluation.md)：完整评测体系，包括生成评测和 RAGAS
- [service-ops.md](service-ops.md)：`POST /eval`、后台任务和服务运行说明
- [eval-results.md](eval-results.md)：当前结果快照

如果你只想运行生成评测，当前入口仍然是：

```bash
python src/eval_generation.py --skip-ragas --save-detail
python src/eval_ragas.py --resume
```

原文中的这些内容已并入新的评测主文档：

- 生成评测指标
- RAGAS 运行方式
- 输出文件说明
- 生成评测、检索评测与 Rerank 评测的关系

