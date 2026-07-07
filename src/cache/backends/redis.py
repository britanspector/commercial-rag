"""Redis L1 精确缓存（String + TTL）。"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from cache.backends.base import ExactCacheBackend
from cache.backends.serialization import entry_from_json, entry_to_json
from cache.policy import is_entry_expired
from cache.types import CacheEntry, CacheInvalidateFilter

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)


class RedisExactBackend(ExactCacheBackend):
    """
    L1 Redis 后端：SETEX 存 JSON，key 为 prefix + SHA256(logical_key)。

    Redis 不可用时 get 返回 None、put/invalidate 静默跳过，不抛异常。
    """

    def __init__(
        self,
        *,
        url: str,
        key_prefix: str = "rag:cache",
        timeout_s: float = 1.0,
        max_entries: int = 10_000,
        client: Redis | None = None,
    ) -> None:
        self._prefix = key_prefix.rstrip(":")
        self._max_entries = max(1, max_entries)
        self._timeout_s = timeout_s
        self._client: Redis | None = client
        self._available = False
        self._last_error: str | None = None

        if client is not None:
            from redis.exceptions import RedisError

            self._redis_error_type = RedisError
            self._available = True
            return

        try:
            from redis import Redis
            from redis.exceptions import RedisError

            self._redis_error_type = RedisError
            self._client = Redis.from_url(
                url,
                socket_timeout=timeout_s,
                socket_connect_timeout=timeout_s,
                decode_responses=False,
            )
            self._client.ping()
            self._available = True
            logger.info("Redis L1 cache connected prefix=%s", self._prefix)
        except Exception as exc:
            self._last_error = str(exc)
            self._client = None
            self._available = False
            logger.warning("Redis L1 cache unavailable, will degrade to miss: %s", exc)

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _index_key(self) -> str:
        return f"{self._prefix}:l1:_index"

    def _lru_key(self) -> str:
        return f"{self._prefix}:l1:_lru"

    def _entry_redis_key(self, storage_key: str) -> str:
        import hashlib

        digest = hashlib.sha256(storage_key.encode("utf-8")).hexdigest()
        return f"{self._prefix}:l1:{digest}"

    def get(self, storage_key: str) -> CacheEntry | None:
        if not self.available:
            return None
        assert self._client is not None
        redis_key = self._entry_redis_key(storage_key)
        try:
            raw = self._client.get(redis_key)
            if raw is None:
                return None
            entry = entry_from_json(raw)
            if is_entry_expired(entry):
                self._delete_keys([redis_key], storage_keys=[storage_key])
                return None
            return entry
        except self._redis_error_type as exc:  # type: ignore[attr-defined]
            self._handle_error("get", exc)
            return None
        except Exception as exc:
            self._handle_error("get", exc)
            return None

    def put(self, entry: CacheEntry) -> None:
        if not self.available:
            return None
        assert self._client is not None
        storage_key = entry.key.storage_key()
        redis_key = self._entry_redis_key(storage_key)
        ttl_s = max(1, int(entry.ttl_s))
        try:
            pipe = self._client.pipeline()
            pipe.setex(redis_key, ttl_s, entry_to_json(entry))
            pipe.sadd(self._index_key(), redis_key)
            pipe.zadd(self._lru_key(), {redis_key: time.time()})
            pipe.execute()
            self._evict_if_needed()
        except self._redis_error_type as exc:  # type: ignore[attr-defined]
            self._handle_error("put", exc)
        except Exception as exc:
            self._handle_error("put", exc)
        return None

    def invalidate(self, filter_: CacheInvalidateFilter) -> int:
        if not self.available:
            return 0
        assert self._client is not None
        try:
            if filter_.all_entries:
                keys = list(self._client.smembers(self._index_key()))
                return self._delete_keys(keys)

            keys = list(self._client.smembers(self._index_key()))
            if not keys:
                return 0

            to_remove: list[bytes | str] = []
            for redis_key in keys:
                rk = redis_key.decode("utf-8") if isinstance(redis_key, bytes) else str(redis_key)
                if filter_.storage_key_prefix is not None:
                    raw = self._client.get(redis_key)
                    if raw is None:
                        to_remove.append(redis_key)
                        continue
                    entry = entry_from_json(raw)
                    if entry.key.storage_key().startswith(filter_.storage_key_prefix):
                        to_remove.append(redis_key)
                    continue

                raw = self._client.get(redis_key)
                if raw is None:
                    to_remove.append(redis_key)
                    continue
                entry = entry_from_json(raw)
                if self._match_filter(entry, filter_):
                    to_remove.append(redis_key)

            return self._delete_keys(to_remove)
        except self._redis_error_type as exc:  # type: ignore[attr-defined]
            self._handle_error("invalidate", exc)
            return 0
        except Exception as exc:
            self._handle_error("invalidate", exc)
            return 0

    def count(self) -> int:
        if not self.available:
            return 0
        assert self._client is not None
        try:
            return int(self._client.scard(self._index_key()))
        except Exception:
            return 0

    def delete(self, storage_key: str) -> bool:
        if not self.available:
            return False
        assert self._client is not None
        redis_key = self._entry_redis_key(storage_key)
        try:
            removed = self._delete_keys([redis_key])
            return removed > 0
        except Exception as exc:
            self._handle_error("delete", exc)
            return False

    def ping(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.ping()
            self._available = True
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._available = False
            return False

    def _delete_keys(
        self,
        redis_keys: list[bytes | str],
        *,
        storage_keys: list[str] | None = None,
    ) -> int:
        if not redis_keys or not self._client:
            return 0
        pipe = self._client.pipeline()
        for redis_key in redis_keys:
            pipe.delete(redis_key)
            pipe.srem(self._index_key(), redis_key)
            pipe.zrem(self._lru_key(), redis_key)
        pipe.execute()
        return len(redis_keys)

    def _evict_if_needed(self) -> None:
        if not self._client:
            return
        try:
            count = int(self._client.scard(self._index_key()))
            overflow = count - self._max_entries
            if overflow <= 0:
                return
            oldest = self._client.zrange(self._lru_key(), 0, overflow - 1)
            if oldest:
                self._delete_keys(list(oldest))
        except Exception as exc:
            logger.debug("Redis L1 LRU evict skipped: %s", exc)

    def _handle_error(self, op: str, exc: Exception) -> None:
        self._last_error = str(exc)
        logger.warning("Redis L1 %s failed (degraded to miss/no-op): %s", op, exc)

    @staticmethod
    def _match_filter(entry: CacheEntry, filter_: CacheInvalidateFilter) -> bool:
        if filter_.scope is not None and entry.key.scope != filter_.scope:
            return False
        if filter_.index_fingerprint is not None:
            if entry.key.index_fingerprint != filter_.index_fingerprint:
                return False
        if filter_.doc_id:
            doc_id = filter_.doc_id.strip()
            if not doc_id:
                return False
            if not any(
                chunk_id.startswith(doc_id) or doc_id in chunk_id for chunk_id in entry.chunk_ids
            ):
                return False
        return True
