# Cursor Agent 上下文文档（AutoDL / SSH 新窗口必读）

> **用途**：通过 Cursor SSH 连接 AutoDL 时，新窗口无历史对话。请让 Agent **先读本文档**，再执行用户指令。  
> **项目**：commercial-rag — 中文金融研报 RAG  
> **当前阶段**：**200 份研报 × 4 行业** 已入库；P0/P1/P2 优化与 150 题评测完成（2026-05-28）

---

## 1. 项目是什么

端到端 RAG 流水线：

```
PDF → MinerU 解析 → chunk_mineru 分块 → bge-large-zh 向量 + BM25
    → Milvus Lite 检索 → (混合召回) → bge-reranker-v2-m3 Rerank → 引用生成/拒答
```

**仓库根目录**：`commercial-rag/`（用户 clone 或解压后的路径）

---

## 2. 当前最优技术路线（实验结论）

| 组件 | 选型 |
|------|------|
| 解析 | MinerU CLI，`src/parse_pdf_mineru.py` |
| 分块 | `mineru_paragraph_v3`，`src/chunk_mineru.py` |
| Embedding | `BAAI/bge-large-zh-v1.5`，1024 维，查询加 `query: ` 前缀 |
| 向量库 | Milvus Lite → `data/vector/milvus.db` |
| 词法 | BM25 + jieba → `data/vector/bm25_index.pkl` |
| **推荐召回** | **混合 0.35/0.65**（`DEFAULT_HYBRID_VECTOR_WEIGHT`），pool **200** |
| Rerank | `BAAI/bge-reranker-v2-m3`，Top-30 → Top-5 |
| 拒答阈值 | rerank normalize 分 < **0.35** |

**150 题当前最优（P2 索引）**：混合 Recall@10 **92.0%**，Rerank 答案准确率 **88.0%**。

详细数字见：`docs/eval-results.md`、`docs/midterm-summary.md`

---

## 3. 重要代码入口

| 文件 | 作用 |
|------|------|
| `src/parse_pdf_mineru.py` | PDF → MinerU 输出 |
| `src/chunk_mineru.py` | 分块 → `chunks.jsonl` |
| `src/embed_chunks.py` | 向量化 → Milvus |
| `src/build_bm25_index.py` | BM25 索引 |
| `src/retrieval.py` | 三路召回 + HybridRetriever |
| `src/reranker.py` | BGEReranker（FlagEmbedding / CrossEncoder 回退） |
| `src/eval_retrieval.py` | 三路召回对比评测 |
| `src/eval_rerank.py` | 混合+Rerank 评测（phase1 混合召回，phase2 子进程 Rerank） |
| `src/eval_rerank_phase2.py` | 仅 Rerank 阶段（省内存） |
| `src/rag_pipeline.py` | RAG 流水线 — **⚠ 仍为纯向量召回，未接 hybrid** |
| `src/rag_chat.py` | CLI 问答 |

---

## 4. 已知坑（Agent 勿重复踩）

1. **Milvus COSINE 返回距离**（越小越好），混合融合必须用 `1 - distance`，已在 `retrieval.py` 修复。
2. **eval_rerank 分进程**：phase1 加载 Embedding+Milvus+BM25，phase2 单独加载 Reranker，避免 Windows 内存爆。AutoDL GPU 上可考虑合并，但 Rerank 评测 90 题 CPU 约 28 分钟。
3. **transformers 5.x** 与 FlagEmbedding 不兼容（`prepare_for_model`），`reranker.py` 启动时会 smoke test，失败则回退 **CrossEncoder**。
4. **q06 类问题**：Rerank 有时把附录 chunk 排到盈利预测正文前；混合 direct Top5 反而对。
5. **`.gitignore` 忽略** `data/raw_pdfs/`、`data/parsed/`、`data/vector/`，迁移需手动打包或 scp。
6. **PDF 目录结构**：`data/raw_pdfs/{semi-conductor,power-electronics,e-commercial,appliance}/` 各 **50** 份。
7. **HF 镜像**（AutoDL）：`HF_ENDPOINT=https://hf-mirror.com`，缓存建议 `/root/autodl-tmp/hf_cache`。

---

## 5. 数据路径约定

```
data/
├── raw_pdfs/              # 原始 PDF（按行业子目录）
├── parsed/
│   ├── mineru/            # MinerU 原始输出（体积大）
│   ├── chunks.jsonl       # 分块主文件
│   ├── doc_manifest.jsonl
│   └── documents.jsonl
├── vector/
│   ├── milvus.db
│   └── bm25_index.pkl
└── eval/
    ├── eval_questions.jsonl   # 150 题评测集（备份 eval_questions_90.jsonl）
    └── eval_*.csv             # 实验结果
```

**当前规模**：200 PDF → 10,263 chunks，**7,382** 可检索（含 114 `rating_headline`、46 `comparable_table`）。

---

## 6. AutoDL 环境建议

```bash
conda create -n commercial-rag python=3.11 -y
conda activate commercial-rag

# 先安装 PyTorch（需按本机 GPU/驱动/CUDA 版本选择，下面仅为 cu124 示例）
# 无 GPU 可改为 CPU 版 index-url
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# 模型缓存（可选）
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
```

**推荐工作目录**：`/root/autodl-tmp/commercial-rag`（数据盘，空间大）

**GPU 用途**：
- MinerU 解析：`MINERU_DEVICE=cuda`（`parse_pdf_mineru.py` 顶部）
- Embedding / Rerank：自动 `cuda` if available（`resolve_device()`）

---

## 7. 标准运行顺序（200 份扩展）

```bash
# 1. 放置 PDF 到 data/raw_pdfs/<industry>/
# 2. 解析（耗时长，建议 nohup / tmux）
python src/parse_pdf_mineru.py

# 3. 分块
python src/chunk_mineru.py

# 4. 索引
python src/embed_chunks.py
python src/build_bm25_index.py

# 5. 评测
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache
python src/eval_retrieval.py --compare-routes --top-k 10
python scripts/eval_hybrid_weight_sweep.py   # 可选：权重分组
python src/eval_rerank.py

# 6. 交互
python src/rag_chat.py "问题"
```

---

## 8. 用户下一步计划（Agent 优先级）

1. **P3 检索**：对比题 per-entity 召回；封面评级 `rating_headline` 补全（通宝类等）。
2. **P3 答案**：报告期过滤、EPS 定向摘录、must 不满足则拒答。
3. **统一生产链路**：`rag_pipeline.py` 接 `HybridRetriever`（评测已 hybrid，CLI 可能未改）。
4. **Badcase 闭环**：`docs/eval-badcase-analysis.md`（12 miss / 18 答案错）。
5. **规模监控**：800 份时 Milvus Standalone；见 `docs/milvus-index-comparison.md`。

---

## 9. Git 分支（若用户已 push）

- `main`：含 hybrid 召回
- `feature-rerank`：Rerank + 混合评测改动

用户规则：**不要主动 commit**，除非明确要求。

---

## 10. 关键文档索引

| 文档 | 内容 |
|------|------|
| `README.md` | 使用说明与目录 |
| `docs/eval-results.md` | **当前评测结果快照（首选）** |
| `docs/midterm-summary.md` | 项目历程、P0~P2、POC 对照 |
| `docs/eval-badcase-analysis.md` | P2 后 Badcase 与 P3 方案 |
| `docs/eval-scheme.md` | 评测集与三路召回 |
| `docs/rerank-scheme.md` | Rerank + 拒答 |
| `docs/chunk-scheme.md` | 分块策略 |
| `docs/parse-scheme-b.md` | MinerU 解析 |
| `docs/milvus-index-comparison.md` | 索引调研 |

---

## 11. 给 Agent 的操作提示

- 修改前先读相关 `src/` 与 `docs/` 文件，保持与现有命名/结构一致。
- 大规模重跑前确认 `data/` 路径与 GPU 内存；Embedding 批大小在 `embed_chunks.py`。
- 打包/迁移用 `scripts/pack_for_autodl.sh`（Linux）或 `scripts/pack_for_autodl.ps1`（Windows），**不要打包 `__pycache__` 和 Obsidian 笔记**。
- 实验结果写入 `data/eval/`，同步更新 `docs/eval-results.md` 与 `docs/midterm-summary.md`。
