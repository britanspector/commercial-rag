"""L2 语义缓存占位后端（后续替换为 Redis Vector / Milvus）。"""

from __future__ import annotations

from cache.backends.base import SemanticCacheBackend
from cache.types import CacheEntry, CacheInvalidateFilter, CacheQueryContext


class NullSemanticBackend(SemanticCacheBackend):
    """
    未实现语义检索时的空后端：始终 miss，store 为 no-op。
    保留接口以便 CacheManager 统一编排。
    """

    @property
    def implemented(self) -> bool:
        return False

    def lookup(self, context: CacheQueryContext) -> CacheEntry | None:
        return None

    def put(self, entry: CacheEntry) -> None:
        return None

    def invalidate(self, filter_: CacheInvalidateFilter) -> int:
        return 0

    def count(self) -> int:
        return 0
