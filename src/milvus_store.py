"""
Milvus Lite 向量库封装。

使用本地文件作为数据库（无需单独启动 Milvus 服务）：
    client = MilvusClient("./data/vector/milvus.db")
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pymilvus import MilvusClient


COLLECTION_NAME = "rag_chunks"
METRIC_TYPE = "COSINE"
PRIMARY_KEY_MAX_LENGTH = 128
VARCHAR_MAX_LENGTH = 65535


def reset_local_db(db_path: Path) -> None:
    """删除本地 Milvus Lite 目录，避免 Windows 上 manifest 重命名冲突。"""
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)


class MilvusChunkStore:
    def __init__(self, db_path: Path, vector_dim: int) -> None:
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(uri=str(self.db_path))

    def has_collection(self) -> bool:
        return self.client.has_collection(COLLECTION_NAME)

    def recreate_collection(self) -> None:
        if self.has_collection():
            self.client.drop_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=self.vector_dim,
            metric_type=METRIC_TYPE,
            id_type="string",
            max_length=PRIMARY_KEY_MAX_LENGTH,
        )

    def insert_batch(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.client.insert(collection_name=COLLECTION_NAME, data=rows)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除指定文档的全部向量（用于重新上传覆盖）。"""
        if not doc_id or not self.has_collection():
            return
        safe_doc_id = doc_id.replace('"', '\\"')
        self.client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{safe_doc_id}"',
        )

    def flush(self) -> None:
        """Milvus Lite 在 Windows 上 flush 可能报错，默认依赖 close 持久化。"""
        try:
            self.client.flush(COLLECTION_NAME)
        except Exception as error:
            print(f"[警告] Milvus flush 跳过：{error}")

    def load(self) -> None:
        self.client.load_collection(COLLECTION_NAME)

    def close(self) -> None:
        self.client.close()

    def count(self) -> int:
        stats = self.client.get_collection_stats(COLLECTION_NAME)
        return int(stats.get("row_count", 0))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        if output_fields is None:
            output_fields = [
                "chunk_id",
                "doc_id",
                "filename",
                "display_name",
                "company_name",
                "report_title",
                "broker",
                "industry_label",
                "source_pdf_path",
                "section_title",
                "text",
                "page_start",
                "page_end",
                "contains_table",
                "token_count",
                "stock_code",
            ]

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=output_fields,
            search_params={"metric_type": METRIC_TYPE},
        )

        hits: list[dict] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            hits.append(
                {
                    "score": hit.get("distance"),
                    "chunk_id": entity.get("chunk_id"),
                    "doc_id": entity.get("doc_id"),
                    "filename": entity.get("filename"),
                    "display_name": entity.get("display_name"),
                    "company_name": entity.get("company_name"),
                    "report_title": entity.get("report_title"),
                    "broker": entity.get("broker"),
                    "industry_label": entity.get("industry_label"),
                    "source_pdf_path": entity.get("source_pdf_path"),
                    "section_title": entity.get("section_title"),
                    "text": entity.get("text"),
                    "page_start": entity.get("page_start"),
                    "page_end": entity.get("page_end"),
                    "contains_table": entity.get("contains_table"),
                    "token_count": entity.get("token_count"),
                    "stock_code": entity.get("stock_code"),
                    "rating": entity.get("rating", ""),
                }
            )
        return hits


def chunk_record_to_milvus_row(record: dict, vector: list[float]) -> dict:
    return {
        "id": record["chunk_id"],
        "vector": vector,
        "chunk_id": record["chunk_id"],
        "doc_id": record["doc_id"],
        "filename": record["filename"],
        "display_name": str(record.get("display_name", "")),
        "section_title": record.get("section_title", ""),
        "text": record.get("embedding_text") or record["text"],
        "page_start": int(record.get("page_start") or 0),
        "page_end": int(record.get("page_end") or 0),
        "contains_table": bool(record.get("contains_table", False)),
        "content_type": record.get("content_type", ""),
        "token_count": int(record.get("embedding_token_count") or record.get("token_count") or 0),
        "stock_code": str(record.get("stock_code", "")),
        "report_date": str(record.get("report_date", "")),
        "rating": str(record.get("rating", "")),
        "company_name": str(record.get("company_name", "")),
        "report_title": str(record.get("report_title", "")),
        "broker": str(record.get("broker", "")),
        "industry": str(record.get("industry", "")),
        "industry_label": str(record.get("industry_label", "")),
        "source_pdf_path": str(record.get("source_pdf_path", "")),
        "table_id": str(record.get("table_id", "")),
    }
