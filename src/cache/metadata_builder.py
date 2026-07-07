"""构建与 enrichment Pipeline 结果的 metadata filters。"""

from __future__ import annotations

from cache.index_fingerprint import compute_index_fingerprint
from cache.metadata_extract import extract_company_hint, extract_report_year
from cache.types import CacheMetadataFilters
from rag_types import RAGPipelineResult, RAGQuery, RAGSearchResult


def build_metadata_filters(
    rag_query: RAGQuery,
    *,
    index_fingerprint: str | None = None,
) -> CacheMetadataFilters:
    index_fp = index_fingerprint or compute_index_fingerprint()
    return CacheMetadataFilters(
        stock_code=rag_query.stock_code,
        query_type=rag_query.query_type,
        company_name=extract_company_hint(rag_query.query, stock_code=rag_query.stock_code),
        report_year=extract_report_year(rag_query.query),
        doc_version=index_fp,
    )


def enrich_metadata_from_result(
    metadata: CacheMetadataFilters,
    result: RAGSearchResult | RAGPipelineResult,
) -> CacheMetadataFilters:
    """从 Pipeline 结果 top hit 回填公司 / doc_id，强化跨公司防护。"""
    hits = result.rerank_hits
    if not hits:
        return metadata

    top = hits[0]
    company_name = top.company_name.strip() or metadata.company_name
    doc_id = top.doc_id.strip() or metadata.doc_id
    return CacheMetadataFilters(
        stock_code=metadata.stock_code or "",
        query_type=metadata.query_type,
        company_name=company_name,
        report_year=metadata.report_year,
        doc_id=doc_id,
        doc_version=metadata.doc_version,
    )
