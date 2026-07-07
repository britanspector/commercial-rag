"""进程内 L1 精确缓存（LRU + TTL）。"""

from __future__ import annotations

import threading
from collections import OrderedDict

from cache.backends.base import ExactCacheBackend
from cache.policy import is_entry_expired
from cache.types import CacheEntry, CacheInvalidateFilter


class MemoryExactBackend(ExactCacheBackend):
    def __init__(self, *, max_entries: int = 10_000) -> None:
        self._max_entries = max(1, max_entries)
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, storage_key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(storage_key)
            if entry is None:
                return None
            if is_entry_expired(entry):
                del self._entries[storage_key]
                return None
            self._entries.move_to_end(storage_key)
            return entry

    def put(self, entry: CacheEntry) -> None:
        key = entry.key.storage_key()
        with self._lock:
            if key in self._entries:
                del self._entries[key]
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._evict_if_needed()

    def invalidate(self, filter_: CacheInvalidateFilter) -> int:
        with self._lock:
            if filter_.all_entries:
                removed = len(self._entries)
                self._entries.clear()
                return removed

            keys_to_remove: list[str] = []
            for storage_key, entry in self._entries.items():
                if self._match_filter(storage_key, entry, filter_):
                    keys_to_remove.append(storage_key)
            for storage_key in keys_to_remove:
                del self._entries[storage_key]
            return len(keys_to_remove)

    def delete(self, storage_key: str) -> bool:
        with self._lock:
            if storage_key not in self._entries:
                return False
            del self._entries[storage_key]
            return True

    def count(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._entries)

    def _evict_if_needed(self) -> None:
        self._purge_expired_locked()
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def _purge_expired_locked(self) -> None:
        expired = [key for key, entry in self._entries.items() if is_entry_expired(entry)]
        for key in expired:
            del self._entries[key]

    @staticmethod
    def _match_filter(storage_key: str, entry: CacheEntry, filter_: CacheInvalidateFilter) -> bool:
        if filter_.scope is not None and entry.key.scope != filter_.scope:
            return False
        if filter_.index_fingerprint is not None:
            if entry.key.index_fingerprint != filter_.index_fingerprint:
                return False
        if filter_.storage_key_prefix is not None:
            if not storage_key.startswith(filter_.storage_key_prefix):
                return False
        if filter_.doc_id:
            doc_id = filter_.doc_id.strip()
            if not doc_id:
                return False
            if not any(chunk_id.startswith(doc_id) or doc_id in chunk_id for chunk_id in entry.chunk_ids):
                return False
        return True
