# 分块（Chunk）说明

在 `feature-chunk` 分支上运行。

## 作用

把 MinerU 的 `*_content_list.json` 切成适合 RAG 检索的文本块，写入 `chunks.jsonl`。

## 运行

```bash
conda activate commercial-rag
python src/chunk_mineru.py
python src/check_chunks.py
```

## 输入 / 输出

| 路径 | 说明 |
|------|------|
| `data/parsed/mineru/<doc_id>/.../*_content_list.json` | MinerU 结构化解析结果（含 page_idx） |
| `data/parsed/chunks.jsonl` | 每个 chunk 一行 |
| `data/parsed/chunk_summary.csv` | 每份文档的分块统计 |

## 分块策略

1. 按 MinerU 标题层级（`text_level` 1/2）划分章节
2. 同一章节内合并正文与表格
3. 超过 1200 字符的章节按段落二次切分
4. 跳过「免责声明」「投资评级说明」等章节
5. `page_idx + 1` 转为 PDF 页码，写入 `page_start` / `page_end`

## chunk 字段示例

- `chunk_id`：唯一 ID
- `doc_id` / `filename`：来源文档
- `section_title`：章节标题
- `text`：chunk 正文
- `page_start` / `page_end`：页码范围（溯源用）
- `metadata`：公司名、股票代码、报告日期等
