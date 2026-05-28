# Rerank 与 RAG 生成

## 模型

- **初召回**：混合（Milvus 向量 + BM25，默认各 0.5）
- **Reranker**：`BAAI/bge-reranker-v2-m3`（FlagEmbedding）
- **对比**：混合直接 Top-5 vs 混合 Top-20 → Rerank → Top-5

## 安装

```bash
# 若需 GPU 加速，先按本机显卡驱动/CUDA 选择对应 PyTorch 版本（下例仅示例）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python src/embed_chunks.py
python src/build_bm25_index.py
```

## Rerank 评测

```bash
python src/eval_rerank.py
python src/eval_rerank.py --skip-answer
```

输出：

| 文件 | 说明 |
|------|------|
| `data/eval/eval_rerank_comparison.csv` | Recall@5 / Top-1 / MRR 对比 |
| `data/eval/eval_rerank_answer_comparison.csv` | 答案事实准确率、拒答合理率 |

## 引用溯源与拒答

`src/rag_answer.py` / `src/rag_pipeline.py`：

- 回答正文标注 `[1]`、`[2]`…，末尾输出 **【参考文献】**（公司、章节、页码、chunk_id、rerank 分）
- 当 **Top-1 rerank 分数 < 阈值**（默认 `0.35`，normalize=True）时返回：

  > 我不确定，当前检索到的证据不足以可靠回答该问题。

CLI 体验：

```bash
python src/rag_chat.py "京仪装备2026E毛利率预测是多少？"
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `RECALL_POOL` | 20 | 向量初召回数 |
| `FINAL_TOP_K` | 5 | Rerank 后保留数 |
| `DEFAULT_RERANK_REFUSAL_THRESHOLD` | 0.35 | 拒答阈值 |

```bash
python src/eval_rerank.py --refusal-threshold 0.4
python src/rag_chat.py --refusal-threshold 0.4 "问题"
```
