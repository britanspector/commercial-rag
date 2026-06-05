# 服务器迁移与 Agent 交接文档

> **用途**：将 `commercial-rag` 部署到服务器后，供新 Cursor Agent / 协作者快速同步上下文。  
> **更新日期**：2026-06-04  
> **读者**：接手的 Agent — 请先读本文，再执行全量 RAGAS 与后续开发。  
> **关联**：通用技术背景见 [`CURSOR_AGENT_CONTEXT.md`](CURSOR_AGENT_CONTEXT.md)；生成评测方案见 [`eval-generation-scheme.md`](eval-generation-scheme.md)。

---

## 1. 项目一句话

中文金融研报 **RAG**：PDF → MinerU 解析 → 分块 → **bge-large-zh + BM25 混合召回** → **bge-reranker-v2-m3** → **证据校验 + 引用生成 / 拒答**；150 题评测集 + 可选 **RAGAS（本地 Ollama Qwen3-8B）**。

---

## 2. 当前进度总览

| 阶段 | 状态 | 说明 |
|------|------|------|
| 数据入库 | ✅ 完成 | 200 份 PDF，约 7,382 可检索 chunk |
| 检索 / Rerank 评测 | ✅ 完成 | 混合 Recall@10 **92%**，Rerank 事实准确率 **88%**（见 `eval-results.md`） |
| Pipeline 生产链路 | ✅ 已接 hybrid | `rag_pipeline.py`：`query_rewrite → hybrid → rerank → evidence_check → answer` |
| FastAPI + 审计库 | ✅ 已实现 | `/upload` `/search` `/chat` `/eval`；见 `docs/api.md`、`audit-db.md` |
| **生成质量 150 题（规则指标）** | ✅ 已完成 | 本地跑通 `--skip-ragas --save-detail`，约 178s |
| **RAGAS 全量** | ✅ 已完成 | faith≈0.85 / rel≈0.46（130 非拒答题）；`eval_generation_detail_ragas.jsonl` |
| P3 Badcase / 生成优化 | 📋 计划中 | 冗长答案、comparative 拒答、refusal 指标对齐等 |

---

## 3. 150 题生成评测结果（阶段一，无 RAGAS）

数据来源：`data/eval/eval_generation_detail.jsonl`（**150 行**，Pipeline 全链路 + 规则指标）。

| 指标 | 数值 | 备注 |
|------|------|------|
| 题量 | 150 | factual 100 / comparative 26 / summary 23 |
| refusal_rate | **12.7%** | 19 题拒答 |
| refusal_accuracy | **74.0%** | `refused == (not retrieval_hit)` |
| citation_accuracy | **86.3%** | 仅非拒答题（131 题） |
| answer_factually_supported | **78.7%** | 规则 must/gold |
| answer_supported_rule | **91.3%** | 规则综合 |
| retrieval_hit_rate | **73.3%** | 重排 Top-K 至少一条相关 |

**日志侧现象（优化参考）**：

- 部分题 `refusal_ok=False` 但 **rerank 很高仍拒答** → 多为 `evidence_check`（如 `comparative_insufficient`），非检索失败。
- 部分题 **检索未命中仍作答** → `refusal_accuracy` 拉低的主因之一。
- 代码已增加辅助指标 `refusal_accuracy_evidence`（与 `evidence_check.passed` 对齐）；若 detail 为旧跑批可能无此字段，重跑 `eval_generation.py` 会有。

---

## 4. RAGAS 本地试跑（3 题，2026-06-04）

环境：**Ollama `qwen3:8b`** + **bge-large-zh（CPU）** + LangChain **0.2.x 成套**。

| 指标 | 3 题均值 |
|------|----------|
| faithfulness_ragas | **0.889** |
| answer_relevancy_ragas | **0.552** |
| ragas_scored_n | 3 |

| 题号 | Faith | Relevancy | 说明 |
|------|-------|-----------|------|
| q01 EPS | 1.00 | 0.55 | 事实正确，答案偏长拉低 relevancy |
| q02 PE | 0.67 | 0.44 | 含可比表多公司数据，忠实性被扣分 |
| q03 评级 | 1.00 | 0.66 | 结论「增持」正确 |

耗时约 **662s**（均题 ~221s）；日志常见 `LLM returned 1 generations instead of requested 3`（Ollama 对 Answer Relevancy 的影响，属预期内降级）。

**⚠ 文件注意**：最后一次 `--limit 3` 的 RAGAS 会**覆盖** `data/eval/eval_generation_results.csv`（当前仅 3 行）。**150 题 Pipeline 明细仍在** `eval_generation_detail.jsonl`，全量 RAGAS 应以该文件为输入，勿丢。

---

## 5. 服务器上必带 / 可重建资产

### 5.1 必须打包或同步（`.gitignore` 常忽略）

```
data/vector/milvus.db          # Milvus Lite
data/vector/bm25_index.pkl
data/parsed/chunks.jsonl
data/parsed/doc_manifest.jsonl
data/eval/eval_questions.jsonl
data/eval/eval_generation_detail.jsonl   # 150 题 Pipeline 结果（RAGAS 输入）
models/ 或 HF 缓存                         # bge / reranker 权重（见 hf_env.py）
```

可选：`data/raw_pdfs/`、`data/parsed/mineru/`（体积大，服务器若只评测可不带）。

打包脚本：`scripts/pack_for_autodl.sh` / `pack_for_autodl.ps1`。

### 5.2 可在服务器重建

```bash
python src/embed_chunks.py
python src/build_bm25_index.py
```

### 5.3 环境

- Python **3.10+**（conda 环境名建议 `commercial-rag`）
- GPU：Embedding / Rerank /（可选）MinerU；**RAGAS 建议 GPU 给 Ollama，bge 用 CPU**
- 先装 **PyTorch**（按 CUDA 版本），再 `pip install -r requirements.txt`
- RAGAS 专用：`pip install -r requirements-ragas.txt`（LangChain 0.2.x 锁版本）

---

## 6. 服务器推荐执行顺序（全量 RAGAS）

```bash
conda activate commercial-rag
cd /path/to/commercial-rag

# 1. 依赖（勿混装 langchain-openai 1.x）
pip install -r requirements-ragas.txt

# 2. Ollama（若用本地 RAGAS）
ollama pull qwen3:8b
ollama serve   # 或 systemd 常驻

# 3. 确认 150 题 detail 存在
wc -l data/eval/eval_generation_detail.jsonl   # 应为 150

# 4. 勿与 uvicorn / 其它进程同时占 milvus.db（仅 Pipeline 评测需要）
export PYTHONUNBUFFERED=1
export RAGAS_BACKEND=ollama
export RAGAS_LLM_MODEL=qwen3:8b
export RAGAS_EMBED_BACKEND=bge_local

# 5. 全量 RAGAS（不重跑 Milvus Pipeline）
python src/eval_ragas.py --resume

# 调试：先 5 题
# python src/eval_ragas.py --limit 5
```

**输出文件**（`data/eval/`）：

| 文件 | 内容 |
|------|------|
| `eval_generation_detail_ragas.jsonl` | 带 RAGAS 分的完整 JSON |
| `eval_generation_results.csv` | 扁平表（全量跑后恢复 150 行） |
| `eval_generation_metrics.csv` | 含 `faithfulness_ragas`、`answer_relevancy_ragas` |

---

## 7. RAGAS / LangChain 依赖（已踩坑）

**禁止混装**：

| 包 | 锁定版本 |
|----|----------|
| langchain-core | **0.2.43** |
| langchain-community | **0.2.19** |
| langchain-openai | **0.1.25**（不是 0.2.14，也不是 1.x） |
| langchain | 0.2.17 |
| numpy | **&lt;2**（community 0.2.19 要求） |

常见错误：

1. `langchain_community.chat_models.vertexai` → community 太新（0.4.x），降为 0.2.19  
2. `ContextOverflowError` → `langchain-openai 1.x` 与 `core 0.2.43` 混装，降为 **0.1.25**  
3. `model 'qwen3:8b' not found` → `ollama pull qwen3:8b` 或设置 `RAGAS_LLM_MODEL`

启动时 `eval_ragas.py` 会调用 `check_langchain_stack()` / `ensure_ollama_model()`。

---

## 8. 评测与 RAGAS 代码地图

| 路径 | 作用 |
|------|------|
| `src/eval_generation.py` | 150 题 Pipeline 批量评测；`--skip-ragas`、`--save-detail`、`--resume` |
| `src/eval_generation_common.py` | Citation / Refusal / 规则支持度；RAGAS 调度入口 |
| `src/eval_ragas.py` | **仅 RAGAS**，读 `eval_generation_detail.jsonl` |
| `src/eval_ragas_runner.py` | Ollama + bge 逐题 `Faithfulness` / `AnswerRelevancy` |
| `src/eval_ragas_config.py` | 环境变量与默认 `qwen3:8b` |
| `src/rag_pipeline.py` | 与 `/chat` 一致的全链路 |
| `src/pipeline/evidence_check.py` | 拒答规则（rerank、comparative、stock 等） |
| `src/db/` | 审计持久化（upload/search/chat） |
| `requirements-ragas.txt` | RAGAS 锁版本清单 |

---

## 9. API 与审计（若服务器同时跑服务）

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- 审计库：`RAG_DATABASE_URL`（默认 SQLite `data/audit/rag_audit.db`）
- **评测期间**不要与 `eval_generation.py` 并行写 `milvus.db`（会锁库）

---

## 10. 后续工作优先级（Agent 建议）

1. **服务器全量 RAGAS** → 更新 `eval_generation_metrics.csv`，对比 Faithfulness vs `citation_accuracy` / `answer_factually_supported`。  
2. **从 detail 恢复 150 行 CSV**（若被 3 题试跑覆盖）：可从 `eval_generation_detail.jsonl` 重导，或重跑 `--skip-ragas`（耗时会再跑 Pipeline）。  
3. **生成优化**：事实题「先结论后引用」、压缩可比表复述 → 提升 Answer Relevancy。  
4. **拒答策略**：区分 `refusal_accuracy`（检索命中定义）与 `refusal_accuracy_evidence`（证据链定义），comparative 单独分析。  
5. **P3 检索/答案**：见 `eval-badcase-analysis.md`、`CURSOR_AGENT_CONTEXT.md` §8。  
6. **规模**：800 份时评估 Milvus Standalone（`milvus-index-comparison.md`）。

---

## 11. 关键文档索引

| 文档 | 内容 |
|------|------|
| **本文** `SERVER_MIGRATION_HANDOFF.md` | 迁移 + 进度 + RAGAS 服务器步骤 |
| `CURSOR_AGENT_CONTEXT.md` | 技术路线、数据路径、AutoDL 习惯 |
| `eval-generation-scheme.md` | 生成指标定义与命令 |
| `eval-results.md` | 检索/Rerank 数字快照 |
| `audit-db.md` | 审计表结构 |
| `eval-badcase-analysis.md` | Badcase 与 P3 |
| `README.md` | 用户向总览 |

---

## 12. Git / Agent 约定

- **不要主动 `git commit`**，除非用户明确要求。  
- 修改评测结果后，可选同步 `docs/eval-results.md` 或在本文件 §3/§4 更新数字。  
- 保持 `src/` 命名与现有 Pipeline 模块风格一致；避免大范围无关重构。

---

## 13. 给新 Agent 的极简 Checklist

- [ ] 读本文 + `eval-generation-scheme.md`  
- [ ] 确认 `data/eval/eval_generation_detail.jsonl` 为 **150 行**  
- [ ] 安装 `requirements-ragas.txt`，`pip show langchain-openai` 为 **0.1.25**  
- [ ] `ollama pull qwen3:8b` 且 `ollama serve` 可用  
- [ ] 停掉占用 `milvus.db` 的 uvicorn（若只跑 RAGAS 可忽略 Milvus）  
- [ ] `python src/eval_ragas.py --resume`（建议 `tmux` / `nohup`）  
- [ ] 完成后检查 `faithfulness_ragas`、`ragas_scored_n` ≈ 非拒答题数量  

---

*本文档随服务器全量 RAGAS 结果更新 §4 数字即可闭环阶段二评测。*
