"""
lfu_cache.py — O(1) LFU partition cache for Quake, with telemetry.

Design
------
LFUCache implements the classic O(1) LFU algorithm using three structures:
  - key_freq : dict[key → int]         current access frequency per key
  - freq_map : dict[int → OrderedDict]  per-frequency FIFO bucket of keys
  - min_freq : int                      current minimum frequency (for O(1) eviction)

On get(key):
  - Miss → increment miss counter, return False
  - Hit  → move key from freq_map[f] to freq_map[f+1], return True

On put(key):
  - If at capacity, evict the head of freq_map[min_freq] (LFU + LRU tiebreak)
  - Insert key with frequency 1, reset min_freq = 1

All operations are O(1) average: dict lookups and OrderedDict FIFO pops are O(1),
and min_freq is maintained incrementally rather than scanned.

CachedQuakeIndex wraps a QuakeIndex and adds partition-level LFU telemetry:
  1. Replicate Quake's internal centroid search to determine which partition IDs
     would be probed for a given query.
  2. For each probed partition ID, call LFUCache.get() to record a hit or miss.
  3. Put missed IDs into the cache.
  4. Delegate the real search to index.search() as normal.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

try:
    import quake
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    build_libs = sorted(repo_root.glob("build/lib.*"))
    for bl in build_libs:
        if (bl / "quake").exists():
            sys.path.insert(0, str(bl))
            break
    import quake


@dataclass
class CacheStats:
    """Accumulated cache telemetry across all search() calls."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_lookups: int = 0
    capacity: int = 0
    cache_get_time_s: float = 0.0
    cache_put_time_s: float = 0.0
    cache_overhead_s: float = 0.0
    partition_id_time_s: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_lookups if self.total_lookups > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.total_lookups if self.total_lookups > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(capacity={self.capacity}, hits={self.hits}, "
            f"misses={self.misses}, hit_rate={self.hit_rate:.3f}, "
            f"evictions={self.evictions}, total_lookups={self.total_lookups}, "
            f"cache_overhead_s={self.cache_overhead_s:.6f})"
        )

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_lookups": self.total_lookups,
            "hit_rate": round(self.hit_rate, 6),
            "miss_rate": round(self.miss_rate, 6),
            "cache_get_time_s": round(self.cache_get_time_s, 6),
            "cache_put_time_s": round(self.cache_put_time_s, 6),
            "cache_overhead_s": round(self.cache_overhead_s, 6),
            "partition_id_time_s": round(self.partition_id_time_s, 6),
        }


class LFUCache:
    """
    Least-Frequently-Used cache with O(1) get/put.

    Eviction policy: when capacity is full, the key with the lowest access
    frequency is evicted. Ties are broken by insertion order (LRU within the
    same frequency bucket).

    capacity=0 means "no cache" — every lookup is a miss and nothing is stored.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self._capacity: int = capacity
        self._key_freq: Dict[Any, int] = {}           # key → freq
        self._freq_map: Dict[int, OrderedDict] = {}   # freq → {key: None} (FIFO within bucket)
        self._min_freq: int = 0
        self._stats = CacheStats(capacity=capacity)

    @property
    def size(self) -> int:
        return len(self._key_freq)

    @property
    def capacity(self) -> int:
        return self._capacity

    def get(self, key: Any) -> bool:
        """Record a lookup for `key`. Returns True (hit) or False (miss)."""
        self._stats.total_lookups += 1
        if self._capacity == 0:
            self._stats.misses += 1
            return False

        if key not in self._key_freq:
            self._stats.misses += 1
            return False

        self._increment_freq(key)
        self._stats.hits += 1
        return True

    def put(self, key: Any) -> None:
        """Insert `key` into the cache. Evicts the LFU entry if at capacity. No-op when capacity == 0."""
        if self._capacity == 0:
            return
        if key in self._key_freq:
            return
        if len(self._key_freq) >= self._capacity:
            self._evict()
        self._key_freq[key] = 1
        self._freq_map.setdefault(1, OrderedDict())[key] = None
        self._min_freq = 1

    def invalidate(self, key: Any) -> None:
        """Remove a key from the cache (e.g. on write/mutation)."""
        if key in self._key_freq:
            freq = self._key_freq.pop(key)
            self._freq_map[freq].pop(key, None)

    def reset_stats(self) -> None:
        self._stats = CacheStats(capacity=self._capacity)

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            total_lookups=self._stats.total_lookups,
            capacity=self._capacity,
            cache_get_time_s=self._stats.cache_get_time_s,
            cache_put_time_s=self._stats.cache_put_time_s,
            cache_overhead_s=self._stats.cache_overhead_s,
            partition_id_time_s=self._stats.partition_id_time_s,
        )

    def _increment_freq(self, key: Any) -> None:
        freq = self._key_freq[key]
        self._freq_map[freq].pop(key)
        if not self._freq_map[freq] and freq == self._min_freq:
            self._min_freq = freq + 1
        self._key_freq[key] = freq + 1
        self._freq_map.setdefault(freq + 1, OrderedDict())[key] = None

    def _evict(self) -> None:
        """Evict the LFU (and LRU within that freq bucket) key."""
        bucket = self._freq_map.get(self._min_freq)
        if not bucket:
            return
        victim, _ = bucket.popitem(last=False)
        del self._key_freq[victim]
        self._stats.evictions += 1


class CachedQuakeIndex:
    """
    Wraps a QuakeIndex and adds a partition-level LFU read cache.

    Parameters
    ----------
    index : quake.QuakeIndex
        A fully built QuakeIndex.
    cache_fraction : float
        Cache capacity as a fraction of nlist. 0.0 = no cache, 1.0 = full cache.
    """

    def __init__(self, index: quake.QuakeIndex, cache_fraction: float) -> None:
        if not 0.0 <= cache_fraction <= 1.0:
            raise ValueError("cache_fraction must be in [0, 1]")
        self._index = index
        self._cache_fraction = cache_fraction
        capacity = max(0, int(round(cache_fraction * index.nlist())))
        self._cache = LFUCache(capacity)

    def search(self, query: torch.Tensor, search_params: quake.SearchParams) -> quake.SearchResult:
        """Run search and update LFU cache telemetry."""
        t_pid = time.perf_counter()
        probed_pids = self._get_probed_partition_ids(query, search_params.nprobe)
        self._cache._stats.partition_id_time_s += time.perf_counter() - t_pid

        for pid in probed_pids:
            t_get = time.perf_counter()
            hit = self._cache.get(pid)
            self._cache._stats.cache_get_time_s += time.perf_counter() - t_get

            if not hit:
                t_put = time.perf_counter()
                self._cache.put(pid)
                self._cache._stats.cache_put_time_s += time.perf_counter() - t_put

        self._cache._stats.cache_overhead_s = (self._cache._stats.cache_get_time_s + self._cache._stats.cache_put_time_s)

        return self._index.search(query, search_params)

    def cache_stats(self) -> CacheStats:
        return self._cache.stats()

    def reset_cache_stats(self) -> None:
        self._cache.reset_stats()

    @property
    def cache_fraction(self) -> float:
        return self._cache_fraction

    @property
    def cache_capacity(self) -> int:
        return self._cache.capacity

    def _get_probed_partition_ids(self, query: torch.Tensor, nprobe: int) -> List[int]:
        """
        Return the partition IDs Quake would probe for this query by replicating
        the parent centroid search. Falls back to dummy range if no parent exists.
        """
        try:
            parent = self._index.parent
            if parent is None:
                return list(range(nprobe))

            parent_params = quake.SearchParams()
            parent_params.nprobe = 1
            parent_params.k = nprobe

            q = query if query.dim() == 2 else query.unsqueeze(0)
            result = parent.search(q, parent_params)
            ids = result.ids.squeeze(0).tolist()
            return [int(i) for i in ids if i >= 0]
        except Exception:
            return list(range(nprobe))


def _selftest():
    print("Running LFU cache self-test...")

    # Test 1: basic LFU eviction order
    c = LFUCache(2)
    c.put("a"); c.put("b")
    assert c.get("a"); assert c.get("a")
    c.put("c")               # should evict "b" (freq 1), not "a" (freq 3)
    assert not c.get("b"), "b should have been evicted"
    assert c.get("a"), "a should still be in cache"
    print("  PASS: LFU eviction order")

    # Test 2: hit/miss counters
    c2 = LFUCache(3)
    c2.put(1); c2.put(2); c2.put(3)
    c2.get(1); c2.get(1); c2.get(2)
    c2.get(99)   # miss
    st = c2.stats()
    assert st.hits == 3, f"expected 3 hits, got {st.hits}"
    assert st.misses == 1, f"expected 1 miss, got {st.misses}"
    print("  PASS: hit/miss counters")

    # Test 3: capacity=0 is all-miss
    c3 = LFUCache(0)
    c3.put(1)
    assert not c3.get(1), "capacity=0 should always miss"
    assert c3.stats().hit_rate == 0.0
    print("  PASS: capacity=0 always misses")

    # Test 4: eviction count
    c4 = LFUCache(2)
    c4.put("x"); c4.put("y")
    c4.put("z")   # evicts least-frequently-used
    assert c4.stats().evictions == 1
    print("  PASS: eviction counter")

    print("All self-tests passed.")


if __name__ == "__main__":
    _selftest()
