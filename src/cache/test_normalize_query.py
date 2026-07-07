"""normalize_query 单元测试。"""

from __future__ import annotations

import unittest

from cache.policy import normalize_query


class NormalizeQueryTests(unittest.TestCase):
    def test_regression_example2_l1_equivalent(self) -> None:
        a = normalize_query("澜起科技2026年EPS是多少？")
        b = normalize_query("请问澜起科技2026年EPS是？")
        self.assertEqual(a, b)
        self.assertEqual(a, "澜起科技2026年EPS")

    def test_regression_example1_l1_equivalent_after_reorder(self) -> None:
        """同义词 + 语序归一后应 L1 等价。"""
        a = normalize_query("澜起科技2024年的EPS是多少？")
        b = normalize_query("请告诉我2024年澜起科技的每股收益？")
        self.assertEqual(a, "澜起科技2024年EPS")
        self.assertEqual(b, "澜起科技2024年EPS")
        self.assertEqual(a, b)

    def test_regression_example1_same_after_reorder(self) -> None:
        """语序一致 + 同义词时应 L1 等价。"""
        a = normalize_query("澜起科技2024年EPS是多少？")
        b = normalize_query("澜起科技2024年每股收益？")
        self.assertEqual(a, b)

    def test_polite_prefix_stripped(self) -> None:
        self.assertEqual(
            normalize_query("请问焦点科技2026E EPS预测"),
            normalize_query("焦点科技2026E EPS预测"),
        )

    def test_pe_synonym(self) -> None:
        self.assertEqual(
            normalize_query("澜起科技2027年市盈率预测"),
            normalize_query("澜起科技2027年PE预测"),
        )

    def test_revenue_synonym(self) -> None:
        self.assertEqual(
            normalize_query("华凯易佰2025年营业收入是多少"),
            normalize_query("华凯易佰2025年营收"),
        )

    def test_filler_suffix(self) -> None:
        self.assertEqual(
            normalize_query("澜起科技投资评级是什么（帮忙查一下）"),
            normalize_query("澜起科技投资评级"),
        )

    def test_whitespace_nfkc(self) -> None:
        self.assertEqual(
            normalize_query("  澜起科技　2026年EPS  "),
            "澜起科技2026年EPS",
        )


    def test_regression_storage_key_metadata_consistent(self) -> None:
        """礼貌前缀不应改变 metadata 中的公司 hint。"""
        from cache.key_builder import build_cache_key
        from cache.metadata_builder import build_metadata_filters
        from cache.types import CacheScope
        from rag_types import RAGQuery

        def key_hash(q: str) -> str:
            rq = RAGQuery(query=q, stock_code="", query_type="factual")
            meta = build_metadata_filters(rq)
            k = build_cache_key(
                scope=CacheScope.CHAT,
                query=q,
                config_fingerprint="cfg",
                index_fingerprint="idx",
                metadata_filter_fingerprint=meta.fingerprint(),
            )
            return k.storage_key_hash()

        self.assertEqual(
            key_hash("澜起科技2026年EPS是多少？"),
            key_hash("请问澜起科技2026年EPS是？"),
        )


if __name__ == "__main__":
    unittest.main()
