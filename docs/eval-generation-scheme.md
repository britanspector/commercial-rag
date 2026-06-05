# 生成质量评测（阶段一）



在 **150 题**评测集（`data/eval/eval_questions.jsonl`）上批量跑 **RAG Pipeline 全链路**（与 `/chat` 一致），输出自动化指标与逐题 CSV，形成阶段一闭环。



## 指标



| 指标 | 类型 | 说明 |

|------|------|------|

| **Faithfulness** | RAGAS | 答案是否忠于 `contexts`（Top 重排片段） |

| **Answer Relevancy** | RAGAS | 答案与问题的相关程度 |

| **Citation Accuracy** | 自定义 | 非拒答时：有引用、有页码/文档、引用 chunk 相关、must 词命中、doc/stock 对齐、正文含引用标记 |

| **Refusal Accuracy** | 自定义 | `refused` 是否等于 `should_refuse`（`should_refuse = not retrieval_hit`） |

| **refusal_accuracy_evidence** | 自定义 | `refused` 是否等于 `not evidence_check.passed`（与 evidence_check 规则对齐，便于分析 comparative 等拒答） |

| **answer_factually_supported** | 规则 | `must_contain_any` / `gold_answer` 匹配（与 rerank 评测一致） |



## 运行



```bash

conda activate commercial-rag



# 索引就绪

python src/embed_chunks.py

python src/build_bm25_index.py



# 校验

python src/eval_generation.py --dry-run



# 阶段一：全量 Pipeline + 规则指标（推荐先 save-detail，便于后续 RAGAS）

# 评测前请停止 uvicorn，避免 milvus.db 被占用

$env:PYTHONUNBUFFERED="1"

python src/eval_generation.py --skip-ragas --save-detail



# 断点续跑（跳过 results.csv 已有 question_id）

python src/eval_generation.py --skip-ragas --save-detail --resume

```



### 本地 RAGAS（Ollama Qwen3-8B，不重跑 Pipeline）



适合 RTX 4060 8GB：Pipeline 与 RAGAS 串行跑，Embedding 用 CPU bge，GPU 留给 Ollama。



```bash

ollama pull qwen3:8b   # 名称须与 `ollama list` 一致，否则设 $env:RAGAS_LLM_MODEL="你的标签"

ollama serve

# LangChain 须 0.2.x 成套（勿混装 openai 1.x）：
# pip install langchain-openai==0.1.25 langchain-core==0.2.43 langchain-community==0.2.19



pip install ragas datasets langchain-openai langchain-community



# 基于 eval_generation_detail.jsonl 补跑（约 130+ 非拒答题，每题数秒～数十秒）

$env:PYTHONUNBUFFERED="1"

python src/eval_ragas.py



# 调试

python src/eval_ragas.py --limit 3

python src/eval_ragas.py --resume   # 仅补尚未有 ragas_faithfulness 的题

```



### 一次性 Pipeline + RAGAS



```bash

$env:RAGAS_BACKEND="ollama"

$env:RAGAS_LLM_MODEL="qwen3:8b"

python src/eval_generation.py --save-detail

```



### RAGAS 环境变量



| 变量 | 说明 |

|------|------|

| `RAGAS_BACKEND` | `ollama`（默认 auto→本地）\| `openai` |

| `RAGAS_LLM_MODEL` | Ollama 模型，默认 `qwen3:8b` |

| `RAGAS_OLLAMA_BASE` | 默认 `http://localhost:11434/v1` |

| `RAGAS_EMBED_BACKEND` | `bge_local`（默认）\| `openai` |

| `OPENAI_API_KEY` | 仅 `RAGAS_BACKEND=openai` 时需要 |



安装（**LangChain 须统一 0.2.x**，勿混装 1.x）：

```bash
pip install langchain-openai==0.1.25 langchain-core==0.2.43 langchain-community==0.2.19
pip install "ragas>=0.2.10,<0.3" datasets
# 或：pip install -r requirements-ragas.txt
```

若曾装过 `langchain-openai 1.x`，需先降级 `langchain-openai`，否则会报 `ContextOverflowError` 导入失败。



## 输出文件



| 文件 | 内容 |

|------|------|

| `eval_generation_results.csv` | 每题：答案摘要、拒答、引用分、refusal_correct、RAGAS 分等 |

| `eval_generation_metrics.csv` | 整体均值 |

| `eval_generation_metrics_by_query_type.csv` | 按 factual / comparative / summary 分组 |

| `eval_generation_detail.jsonl` | `--save-detail` 时完整 JSON（含 `ragas_contexts`，供 `eval_ragas.py`） |

| `eval_generation_detail_ragas.jsonl` | `eval_ragas.py` 输出，带 RAGAS 分 |



## 与检索评测的关系



```

eval_retrieval.py   → Recall@K / MRR（检索）

eval_rerank.py      → Rerank + 规则答案准确率

eval_generation.py  → 全 Pipeline + Citation/Refusal + 可选 RAGAS（生成质量）

eval_ragas.py       → 仅 RAGAS 补跑（基于 detail.jsonl）

```



## 代码入口



- `src/eval_generation.py` — Pipeline 批量评测 CLI

- `src/eval_ragas.py` — 本地 RAGAS 补跑 CLI

- `src/eval_generation_common.py` — 指标与 RAGAS 调度

- `src/eval_ragas_runner.py` — Ollama + bge 逐题打分

- `src/eval_ragas_config.py` — 环境变量与默认配置

