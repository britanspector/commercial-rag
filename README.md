# commercial-rag

金融研报 RAG 数据处理与检索评测流水线：**PDF 解析 → 分块 → 向量 + BM25 混合召回 → Rerank → 引用生成 / 拒答**。

**当前规模**：**200 份研报**（四行业 × 50）→ **7,382** 可检索 chunk。  
**评测集**：150 题（factual 100 / comparative 26 / summary 23），`data/eval/eval_questions.jsonl`。

> **最新评测结果**：[docs/eval-results.md](docs/eval-results.md)  
> 项目历程与实验对照：[docs/midterm-summary.md](docs/midterm-summary.md)  
> AutoDL / Cursor Agent：[docs/CURSOR_AGENT_CONTEXT.md](docs/CURSOR_AGENT_CONTEXT.md)

---

## 技术路线（当前最优）

```
PDF → MinerU → chunk_mineru (P2) → bge-large-zh-v1.5 + BM25
                              ↓
                    混合召回 pool=200 (0.35/0.65)
                              ↓
                    bge-reranker-v2-m3 Top30 → Top-5
                              ↓
                    引用生成 + 低分拒答 (threshold=0.35)
```

| 组件 | 选型 |
|------|------|
| 解析 | MinerU CLI（`src/parse_pdf_mineru.py`） |
| 分块 | `mineru_paragraph_v3` + P2（评级块、表语义、可比表） |
| Embedding | `BAAI/bge-large-zh-v1.5`（1024 维） |
| 向量库 | Milvus Lite（COSINE），7,382 条 |
| 词法 | BM25Okapi + jieba |
| 混合召回 | min-max 加权（**向量 0.35 / BM25 0.65**），pool **200** |
| Rerank | `BAAI/bge-reranker-v2-m3`，Top-30 → Top-5 |
| 生成 | 模板引用 + Top-1 rerank 低分拒答 |

---

## 实验结果摘要（150 题，P2 索引，2026-05-28）

| 指标 | 数值 |
|------|------|
| 混合 Recall@10 | **92.0%**（138/150） |
| 混合 MRR@10 | **0.836** |
| Rerank 答案事实准确率 | **88.0%** |
| 检索未命中（hard miss） | 12 题 |

### 三路召回（Top-10，混合权重 0.35）

| 路线 | Recall@10 | MRR |
|------|-----------|-----|
| 纯向量 | 86.0% | 0.748 |
| 纯 BM25 | 92.0% | 0.750 |
| **混合** | **92.0%** | **0.836** |

### Rerank（混合初召回）

| 策略 | Recall@5 | 事实准确率 |
|------|----------|-----------|
| 混合直接 Top5 | 92.7% | 84.0% |
| **混合 + Rerank** | 90.7% | **88.0%** |

相对 150 题优化前基线（Recall@10 84.7% / 答案 82.7%），P0+P1+P2 累计 **+7.3pp / +5.3pp**。

完整数据、权重扫描、Badcase 与复现命令见 [docs/eval-results.md](docs/eval-results.md)。

---

## 推荐运行顺序

```bash
conda activate commercial-rag

# 先按本机 GPU/驱动/CUDA 选择 PyTorch 版本（示例为 cu124）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 无 GPU 或只跑 CPU 时可用：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python src/parse_pdf_mineru.py
python src/chunk_mineru.py
python src/embed_chunks.py
python src/build_bm25_index.py

python src/eval_retrieval.py --compare-routes --top-k 10
python src/eval_rerank.py

python src/rag_chat.py "京仪装备2026E毛利率预测是多少？"
```

GPU（AutoDL / 本地 CUDA）：

```bash
# PyTorch CUDA 版本需与本机显卡驱动、CUDA 运行时匹配
# 下例仅为 cu124 示例，请按本机环境替换 cu124 / cu121 / cpu 等
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# MinerU: 在 src/parse_pdf_mineru.py 设 MINERU_DEVICE = "cuda"
```

---

## 模块说明

### 1. PDF 解析（MinerU）

- **代码**：`src/parse_pdf_mineru.py`、`src/pdf_paths.py`
- **输入**：`data/raw_pdfs/<industry>/*.pdf`
- **输出**：`data/parsed/mineru/`、`documents.jsonl`、`doc_manifest.jsonl`
- **文档**：[docs/parse-scheme-b.md](docs/parse-scheme-b.md)

### 2. 分块

- **代码**：`src/chunk_mineru.py`（策略 `mineru_paragraph_v3`）
- **输出**：`data/parsed/chunks.jsonl`
- **文档**：[docs/chunk-scheme.md](docs/chunk-scheme.md)

### 3. 向量化与 BM25

| 文件 | 作用 |
|------|------|
| `src/embed_chunks.py` | bge-large-zh → `data/vector/milvus.db` |
| `src/build_bm25_index.py` | BM25 → `data/vector/bm25_index.pkl` |
| `src/milvus_store.py` | Milvus Lite 封装 |

- **文档**：[docs/embed-scheme.md](docs/embed-scheme.md)

### 4. 检索评测（三路召回）

- **代码**：`src/retrieval.py`、`src/eval_retrieval.py`
- **路线**：`vector` / `bm25` / `hybrid`
- **输出**：`data/eval/eval_route_comparison.csv`
- **文档**：[docs/eval-scheme.md](docs/eval-scheme.md)

### 5. Rerank 与 RAG

| 文件 | 作用 |
|------|------|
| `src/reranker.py` | bge-reranker-v2-m3（CrossEncoder 回退） |
| `src/eval_rerank.py` | 混合 Top20→Rerank vs 混合 Top5 对比 |
| `src/rag_pipeline.py` | RAG 流水线（⚠ 当前仍为纯向量召回，待接 hybrid） |
| `src/rag_chat.py` | CLI 问答 |

- **文档**：[docs/rerank-scheme.md](docs/rerank-scheme.md)

---

## 目录结构

```
commercial-rag/
├── data/
│   ├── raw_pdfs/              # 原始 PDF（按行业子目录，.gitignore）
│   ├── parsed/
│   │   ├── mineru/            # MinerU 输出（体积大）
│   │   ├── chunks.jsonl       # 分块结果
│   │   ├── documents.jsonl
│   │   └── doc_manifest.jsonl
│   ├── vector/
│   │   ├── milvus.db          # 向量库
│   │   └── bm25_index.pkl     # BM25 索引
│   └── eval/                  # 评测集与实验 CSV
├── src/
├── scripts/
│   ├── build_eval_questions_150.py
│   ├── eval_hybrid_weight_sweep.py
│   ├── pack_for_autodl.ps1    # Windows 打包（未执行）
│   └── pack_for_autodl.sh     # Linux 打包（未执行）
├── docs/
│   ├── eval-results.md        # 当前评测结果快照（推荐）
│   ├── midterm-summary.md     # 项目历程与 POC 对照
│   ├── eval-badcase-analysis.md
│   ├── CURSOR_AGENT_CONTEXT.md
│   └── …
├── requirements.txt
└── notes/                     # 个人笔记（可选）
```

---

## 迁移 AutoDL：文件分级与打包

| 级别 | 内容 | 适用场景 |
|------|------|----------|
| **minimal** | 代码 + docs + 评测集 | 200 份全量重跑 |
| **essential** | + chunks / milvus / bm25 | POC 迁移，跳过 embed |
| **recommended** | + mineru/ | 跳过 PDF 解析 |
| **full** | + raw_pdfs/ | 完整 24 份 POC 镜像 |

**不必打包**：`__pycache__/`、`notes/.obsidian/`、临时 pool 缓存。  
**可选单独拷贝**：HuggingFace 模型缓存（`HF_HOME`），服务器可联网重下。

```powershell
# Windows（生成 zip，不自动上传）
.\scripts\pack_for_autodl.ps1 -Tier essential
.\scripts\pack_for_autodl.ps1 -Tier full
```

```bash
# Linux / AutoDL
bash scripts/pack_for_autodl.sh --tier essential
bash scripts/pack_for_autodl.sh --tier full
```

解压后在新 Cursor 窗口让 Agent 先读：`docs/CURSOR_AGENT_CONTEXT.md`

---

## 依赖

| 文件 | 阶段 |
|------|------|
| `requirements.txt` | 全流程统一依赖（MinerU / Chunk / Embedding / BM25 / Rerank） |

---

## 后续规划

- P3：对比题 per-entity 召回、封面评级块、must 不满足拒答（见 `docs/eval-badcase-analysis.md`）
- `rag_pipeline.py` 统一为混合 + Rerank 生产链路
- 800 份量级评估 Milvus Standalone + IVF/HNSW（见 [docs/milvus-index-comparison.md](docs/milvus-index-comparison.md)）
- RAGAS 自动化评测（后置）
