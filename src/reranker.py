"""
BGE Reranker 封装（优先 FlagEmbedding bge-reranker-v2-m3，失败时回退 sentence-transformers CrossEncoder）。
"""

from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from hf_env import bootstrap_hf_cache, resolve_local_model_path

bootstrap_hf_cache()

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_BATCH_SIZE = 4
RERANK_MAX_LENGTH = 512
PASSAGE_MAX_CHARS = 1200


def hit_passage_text(hit: dict) -> str:
    body = str(hit.get("embedding_text") or hit.get("text") or "")
    parts = [
        hit.get("company_name", ""),
        hit.get("section_title", ""),
        body,
    ]
    passage = "\n".join(part for part in parts if part).strip()
    if len(passage) > PASSAGE_MAX_CHARS:
        passage = passage[:PASSAGE_MAX_CHARS] + "…"
    return passage


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class BGEReranker:
    def __init__(self, device: str | None = None, use_fp16: bool = True) -> None:
        from embed_chunks import resolve_device

        bootstrap_hf_cache()

        self.device = device or resolve_device()
        self._backend = "flag"
        model_path = resolve_local_model_path(RERANK_MODEL)

        print(f"正在加载 Reranker：{RERANK_MODEL}")
        print(f"  设备：{self.device}，本地路径：{model_path}")
        gc.collect()

        try:
            from FlagEmbedding import FlagReranker

            fp16 = use_fp16 and self.device == "cuda"
            self.model = FlagReranker(
                model_path,
                use_fp16=fp16,
                device=self.device,
            )
            # transformers 5.x 移除了 prepare_for_model，FlagReranker 推理会失败
            try:
                self.model.compute_score(
                    [["smoke", "test"]],
                    batch_size=1,
                    max_length=32,
                    normalize=False,
                )
            except AttributeError:
                raise RuntimeError(
                    "FlagReranker 与当前 transformers 版本不兼容（缺少 prepare_for_model）"
                )
            print("  FlagEmbedding Reranker 加载完成。")
        except Exception as error:
            print(f"  [警告] FlagReranker 加载失败（{error}），回退 CrossEncoder ...")
            gc.collect()
            from sentence_transformers import CrossEncoder

            self._backend = "cross_encoder"
            self.model = CrossEncoder(
                model_path,
                max_length=RERANK_MAX_LENGTH,
                device=self.device,
            )
            print("  CrossEncoder 加载完成。")

    def score_pairs(
        self,
        query: str,
        passages: list[str],
        normalize: bool = True,
    ) -> list[float]:
        if not passages:
            return []

        pairs = [[query, passage] for passage in passages]

        if self._backend == "flag":
            scores = self.model.compute_score(
                pairs,
                batch_size=RERANK_BATCH_SIZE,
                max_length=RERANK_MAX_LENGTH,
                normalize=normalize,
            )
            if isinstance(scores, (int, float)):
                return [float(scores)]
            return [float(score) for score in scores]

        raw_scores = self.model.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
        values = [float(score) for score in raw_scores]
        if not normalize:
            return values
        return [_sigmoid(value) for value in values]

    def rerank_hits(
        self,
        query: str,
        hits: list[dict],
        top_k: int,
        normalize: bool = True,
    ) -> list[dict]:
        if not hits:
            return []

        passages = [hit_passage_text(hit) for hit in hits]
        scores = self.score_pairs(query, passages, normalize=normalize)

        ranked: list[dict] = []
        for hit, score in zip(hits, scores):
            item = dict(hit)
            item["score_rerank"] = float(score)
            item["score"] = float(score)
            ranked.append(item)

        ranked.sort(key=lambda item: item["score_rerank"], reverse=True)
        return ranked[:top_k]
