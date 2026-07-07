"""对比题多主体检索单元测试。"""

from __future__ import annotations

import unittest

from pipeline.comparative_rerank import (
    _company_matches_entity,
    _filter_hits_for_entity,
    distinct_companies_in_hits,
    rerank_comparative,
)
from query_enhance import build_comparative_sub_queries, build_entity_sub_query
from rag_types import EntitySubQuery


class BuildEntitySubQueryTests(unittest.TestCase):
    def test_guanghe_huaneng_eps_2026(self) -> None:
        q = "中国广核和华能国际2026年的EPS对比？"
        subs = dict(build_comparative_sub_queries(q, ["中国广核", "华能国际"]))
        self.assertIn("中国广核", subs)
        self.assertIn("华能国际", subs)
        self.assertIn("2026", subs["中国广核"])
        self.assertIn("EPS", subs["中国广核"].upper())
        self.assertNotIn("华能国际", subs["中国广核"])
        self.assertNotIn("中国广核", subs["华能国际"])

    def test_lanqi_huaneng_eps(self) -> None:
        q = "澜起科技和华能国际2026年的EPS对比？"
        a = build_entity_sub_query("澜起科技", q, other_entities=["华能国际"])
        self.assertIn("澜起科技", a)
        self.assertNotIn("华能国际", a)


class ComparativeRerankTests(unittest.TestCase):
    class _FakeReranker:
        def rerank_hits(self, query: str, hits: list[dict], *, top_k: int, normalize: bool = True):
            _ = normalize
            scored = []
            for i, h in enumerate(hits):
                item = dict(h)
                item["score_rerank"] = 1.0 - i * 0.01 + (0.001 if query.startswith(h.get("company_name", "")[:2]) else 0)
                scored.append(item)
            scored.sort(key=lambda x: x["score_rerank"], reverse=True)
            return scored[:top_k]

    def test_company_match(self) -> None:
        self.assertTrue(_company_matches_entity("华能国际", "华能国际"))
        self.assertTrue(_company_matches_entity("中国广核（003816）", "中国广核"))

    def test_filter_bucket(self) -> None:
        hits = [
            {"chunk_id": "a", "company_name": "中国广核"},
            {"chunk_id": "b", "company_name": "华能国际"},
        ]
        bucket = _filter_hits_for_entity(hits, "华能国际")
        self.assertEqual(len(bucket), 1)
        self.assertEqual(bucket[0]["company_name"], "华能国际")

    def test_quota_merge_two_companies(self) -> None:
        hits = [
            {"chunk_id": f"g{i}", "company_name": "中国广核", "text": "eps"}
            for i in range(10)
        ] + [
            {"chunk_id": f"h{i}", "company_name": "华能国际", "text": "eps"}
            for i in range(10)
        ]
        subs = [
            EntitySubQuery(entity="中国广核", query="中国广核 2026年 EPS"),
            EntitySubQuery(entity="华能国际", query="华能国际 2026年 EPS"),
        ]
        result = rerank_comparative(
            "中国广核和华能国际2026年的EPS对比？",
            hits,
            self._FakeReranker(),
            compare_entities=["中国广核", "华能国际"],
            entity_sub_queries=subs,
            top_k=5,
        )
        companies = distinct_companies_in_hits(result.hits)
        self.assertGreaterEqual(len(companies), 2)
        self.assertEqual(len(result.hits), 5)


if __name__ == "__main__":
    unittest.main()
