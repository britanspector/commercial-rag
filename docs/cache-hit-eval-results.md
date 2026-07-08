# 缓存命中专项评测（历史入口）

本文件已整合进 [cache.md](cache.md)。

这里原先记录的是 paraphrase / 语序改写 / 礼貌词改写等缓存命中专项结果，现在统一作为缓存主文档的一部分维护。

相关原始文件仍保留在：

- `data/eval/cache_hit_pairs.jsonl`
- `data/eval/cache_hit_results_*.csv`
- `data/eval/cache_hit_report_*.md`
- `data/eval/logs/cache_hit_*.log`

如需复现专项评测，可继续使用：

```bash
PYTHONPATH=src python src/eval_cache_hit.py --modes l1_only,l1_l2 --stock-code-mode empty
```
