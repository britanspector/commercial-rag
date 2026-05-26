# commercial-rag

金融研报 RAG 数据处理流水线：PDF 解析 → 分块 → 向量化 → 检索与离线评测。

当前数据规模：**24 份研报**（半导体 / 电力 / 互联网电商各 8 份），约 **991** 个可检索 chunk。

---

## 推荐运行顺序

```bash
conda activate commercial-rag

pip install -r requirements-mineru.txt
pip install -r requirements-chunk.txt
pip install -r requirements-embed.txt
pip install milvus-lite   # Windows 需单独安装

python src/parse_pdf_mineru.py
python src/chunk_mineru.py
python src/embed_chunks.py
python src/check_milvus.py "澜起科技2026年EPS是多少"
python src/eval_retrieval.py
```

---

## 1. PDF 解析（MinerU，方案 B）

**功能实现代码：** `src/parse_pdf_mineru.py`  
通过子进程调用 **MinerU CLI**（`mineru`）解析 PDF；`src/pdf_paths.py` 负责在 `data/raw_pdfs` 子目录中递归发现 PDF、写入文档清单。  
其他能力：从 Markdown 抽取公司名/股票代码/评级等元数据；**断点续跑**（已有 `*_content_list_v2.json` 则跳过）；单份失败记日志并继续；每份成功后 **append** 写入 `documents.jsonl`（避免 24 份全文堆在内存）。

**辅助脚本：**

| 文件 | 作用 |
|------|------|
| `src/pdf_paths.py` | 路径常量、行业文件夹映射、`discover_pdf_files()`、`doc_manifest.jsonl` |
| `src/check_parser_mineru.py` | 抽查 MinerU 解析结果 |
| `src/parse_pdf_pages.py` | 方案 A（PyMuPDF + pdfplumber），当前主线不用 |

**数据依赖：** `data/raw_pdfs/`  
按行业分子目录，例如：

```
data/raw_pdfs/semi-conductor/*.pdf      # 半导体 8 份
data/raw_pdfs/power-electronics/*.pdf   # 电力 8 份
data/raw_pdfs/e-commercial/*.pdf        # 互联网电商 8 份
```

**输出结果：**

| 路径 | 内容 |
|------|------|
| `data/parsed/mineru/<doc_id>/.../auto/*_content_list_v2.json` | MinerU v2 结构化版面（分块主输入） |
| `data/parsed/mineru/<doc_id>/.../*.md` | 同文档 Markdown |
| `data/parsed/documents.jsonl` | 每行一份文档：`doc_id`、`text`（全文 MD）、`metadata`、`industry` 等 |
| `data/parsed/doc_manifest.jsonl` | 文档清单：`doc_id`、行业、`source_pdf_path`（供分块溯源） |
| `data/parsed/parse_summary.csv` | 每份 PDF 解析成功/失败统计 |

`content_list_v2.json` 为按页的段落/表格/标题列表（JSON 数组的数组），供 `chunk_mineru.py` 读取。

**运行方法：**

```bash
python src/parse_pdf_mineru.py
```

配置见 `src/parse_pdf_mineru.py` 顶部：`MINERU_DEVICE`、`MINERU_BACKEND`、`RESUME_SKIP_PARSED`。详见 `docs/parse-scheme-b.md`。

---

## 2. 分块（Chunk）

**功能实现代码：** `src/chunk_mineru.py`（策略名 `mineru_paragraph_v3`）  
读取 MinerU 的 `*_content_list_v2.json`，按段落合并正文、按表拆分表格；生成 `embedding_text`（含公司/章节/单位等上下文 + 表格自然语言转写）；过滤免责声明、分析师邮箱等噪声；从正文前几页抽取 `company_name`、`broker`、`report_title` 等；结合 `doc_manifest.jsonl` 生成 **`display_name`**（用户可见的研报标题，而非编号文件名）。

**辅助脚本：** `src/check_chunks.py` — 随机抽样查看 chunk 内容与 token 数。

**数据依赖：**

- `data/parsed/mineru/**/**_content_list_v2.json`
- `data/parsed/doc_manifest.jsonl`（可选，用于行业与 `source_pdf_path`）

**输出结果：**

| 路径 | 内容 |
|------|------|
| `data/parsed/chunks.jsonl` | 每行一个 chunk（JSON），核心字段见下 |
| `data/parsed/chunk_summary.csv` | 每份文档的 chunk 数量、可检索数、最大 token |

**单条 chunk 结构（摘要）：**

```json
{
  "chunk_id": "H3_AP202605201822487813_1_0001",
  "doc_id": "H3_AP202605201822487813_1",
  "display_name": "澜起科技（688008） — … [半导体, 东海证券, 2026-05-19, 增持]",
  "company_name": "澜起科技",
  "stock_code": "688008",
  "section_title": "盈利预测与估值简表",
  "page_start": 2,
  "page_end": 2,
  "content_type": "text | table | noise",
  "is_retrievable": true,
  "embedding_text": "送入向量模型的文本（含元数据块 + 正文/表格转写）",
  "table_raw": "表格原始 pipe 文本（仅 table）",
  "table_id": "同表子块共享 ID",
  "industry_label": "半导体"
}
```

**运行方法：**

```bash
python src/chunk_mineru.py
python src/check_chunks.py
```

详见 `docs/chunk-scheme.md`。

---

## 3. 向量化与向量库（Milvus Lite）

**功能实现代码：**

| 文件 | 作用 |
|------|------|
| `src/embed_chunks.py` | 用 **bge-large-zh-v1.5** 对可检索 chunk 的 `embedding_text` 编码，写入 Milvus |
| `src/milvus_store.py` | Milvus Lite 封装：建表、批量插入、COSINE 检索；`reset_local_db()` 避免 Windows 锁文件问题 |

仅 **`is_retrievable=true`** 的 chunk 入库。查询侧检索时需加 `query: ` 前缀（`check_milvus.py` / `eval_retrieval.py` 已处理）。

**数据依赖：** `data/parsed/chunks.jsonl`

**输出结果：**

| 路径 | 内容 |
|------|------|
| `data/vector/milvus.db` | Milvus Lite 本地库，collection `rag_chunks`，约 991 条向量 |
| `data/parsed/embed_summary.csv` | 模型名、设备、写入条数等 |

**运行方法：**

```bash
python src/embed_chunks.py
python src/check_milvus.py
python src/check_milvus.py "你的自然语言问题"
```

GPU OOM 时可在 `embed_chunks.py` 中设 `EMBED_DEVICE = "cpu"`。详见 `docs/embed-scheme.md`。

---

## 4. 检索离线评测

**功能实现代码：** `src/eval_retrieval.py`  
对评测集批量向量检索（Top-K），按 `stock_code` / `doc_id` / 关键词 / `gold_chunk_ids` 判定是否命中，计算 **Recall@5、Recall@10、MRR**，并导出未命中 case。

**数据依赖：**

- `data/eval/eval_questions.jsonl` — **28 题**人工标注（含 `query`、`gold_answer`、`gold_chunk_ids` 等）
- `data/vector/milvus.db` — 需先执行 `embed_chunks.py`

**输出结果：**

| 路径 | 内容 |
|------|------|
| `data/eval/eval_metrics.csv` | 汇总指标 |
| `data/eval/eval_results.csv` | 每题命中 rank、Top1 chunk 等 |
| `data/eval/eval_misses.jsonl` | 未命中题目（用于改 chunk 规则） |
| `data/eval/eval_detail.jsonl` | 全量评测 JSON |

**运行方法：**

```bash
python src/eval_retrieval.py --dry-run   # 仅校验评测题，不访问 Milvus
python src/eval_retrieval.py             # 正式评测（默认 Top-10）
python src/eval_retrieval.py --top-k 10
```

详见 `docs/eval-scheme.md`。

---

## 5. 目录结构一览

```
commercial-rag/
├── data/
│   ├── raw_pdfs/              # 原始 PDF（按行业子目录）
│   ├── parsed/
│   │   ├── mineru/            # MinerU 原始输出
│   │   ├── chunks.jsonl       # 分块结果
│   │   ├── documents.jsonl
│   │   ├── doc_manifest.jsonl
│   │   └── *.csv              # 各阶段统计
│   ├── vector/
│   │   └── milvus.db          # 向量库
│   └── eval/                  # 评测集与评测结果
├── src/                       # 见上文各模块
├── docs/                      # 分块 / 解析 / 向量 / 评测说明
├── requirements-*.txt
└── notes/                     # 项目笔记（技术选型等）
```

---

## 6. 依赖与环境

| 文件 | 阶段 |
|------|------|
| `requirements-mineru.txt` | MinerU 解析 |
| `requirements-chunk.txt` | 分块（transformers tokenizer） |
| `requirements-embed.txt` | sentence-transformers + pymilvus + torch |

建议使用 Conda 环境 `commercial-rag`。MinerU 使用 GPU 需安装 **CUDA 版 PyTorch**（见 `requirements-mineru.txt` 注释）。

---

## 7. 当前基线（参考）

在 24 份研报、991 可检索 chunk 上，首轮向量检索评测（28 题）约为：

- Recall@5：**67.9%**
- Recall@10：**71.4%**
- MRR：**0.612**

未命中多集中在：复杂可比公司表、首页评级/目标价、风险提示、跨公司混淆。可对照 `data/eval/eval_misses.jsonl` 迭代 `chunk_mineru.py` 规则。

---

## 8. 后续扩展（见笔记）

`notes/RAG项目笔记/note1.md` 中规划：BM25 混合检索、`bge-reranker-v2-m3` 重排、RAG 问答与引用展示等；当前仓库尚未实现。
