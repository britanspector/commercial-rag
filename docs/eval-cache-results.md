# 缓存三模式评测（历史入口）

本文件已整合进 [cache.md](cache.md)。

这里原先记录的是三种缓存模式下的命中率、延迟和质量对比，现在统一归入缓存主文档维护。

原始结果文件仍保留在：

- `data/eval/eval_cache_results.csv`
- `data/eval/eval_cache_comparison.csv`
- `data/eval/eval_cache_report.md`

如果需要复现该专项评测，可继续使用原命令：

```bash
PYTHONPATH=src python src/eval_cache.py --modes off,l1,l1l2 --skip-ragas
```
