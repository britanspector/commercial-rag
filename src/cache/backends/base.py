"""缓存后端抽象（L1 精确 / L2 语义）。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cache.types import CacheEntry, CacheInvalidateFilter, CacheInvalidateResult, CacheQueryContext


class ExactCacheBackend(ABC):
    """L1：精确 key 缓存（当前内存实现，后续可换 Redis String）。"""

    @abstractmethod
    def get(self, storage_key: str) -> CacheEntry | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def invalidate(self, filter_: CacheInvalidateFilter) -> int:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    def delete(self, storage_key: str) -> bool:
        """删除单条精确缓存（默认走 invalidate）。"""
        removed = self.invalidate(CacheInvalidateFilter(storage_key_prefix=storage_key))
        return removed > 0

    def ping(self) -> bool:
        return True


class SemanticCacheBackend(ABC):
    """L2：语义近邻缓存（当前占位，后续 Redis Vector / Milvus）。"""

    @abstractmethod
    def lookup(self, context: CacheQueryContext) -> CacheEntry | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def invalidate(self, filter_: CacheInvalidateFilter) -> int:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def implemented(self) -> bool:
        """后端是否已具备真实语义检索能力。"""
        raise NotImplementedError

    def ping(self) -> bool:
        return self.implemented
