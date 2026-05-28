"""RAG 生成相关常量（轻量模块，避免循环依赖）。"""

REFUSAL_MESSAGE = "我不确定，当前检索到的证据不足以可靠回答该问题。"

# bge-reranker-v2-m3 normalize=True 时的默认拒答阈值（0~1，越高越严格）
DEFAULT_RERANK_REFUSAL_THRESHOLD = 0.35

DEFAULT_RECALL_TOP_K = 30
DEFAULT_RERANK_TOP_K = 5
