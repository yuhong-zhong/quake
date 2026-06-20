// lru_cache.h — Persistent LRU partition cache for Quake S3 mode.
//
// Design: A CacheManager owns a hash map of CacheEntry objects and an LRU
// doubly-linked list. A single background thread handles LOAD (fire async S3),
// LOAD_COMPLETE (data arrived from S3), ACCESS (LRU reorder), and PREFETCH
// requests, plus eviction when usage exceeds a configurable threshold.
// Worker threads call get()/release() to pin/unpin partitions; data pointers
// are stable while pinned.
//
// Async flow:
//   Worker  ──LOAD/PREFETCH──▶  CacheManager bg thread
//                                    │ fires GetObjectAsync
//                                    ▼
//                               AWS S3 thread pool
//                                    │ callback enqueues
//                                    ▼
//                              LOAD_COMPLETE ──▶ CacheManager bg thread
//                                                    │ inserts data, notifies waiters

#ifndef LRU_CACHE_H
#define LRU_CACHE_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

class IndexPartition;

#ifdef QUAKE_USE_S3
#include <aws/s3/S3Client.h>
#endif

namespace quake {

// ───────────────────────────────────────────────── Load state machine
enum class LoadState {
    EMPTY,    // Entry exists in table but has no data.
    LOADING,  // A fetch from the data store is in progress.
    LOADED,   // Data is available for reading.
    ERROR     // Fetch failed.
};

// ───────────────────────────────────────────────── LRU doubly-linked list
struct LRUNode {
    size_t partition_id = 0;
    LRUNode* prev = nullptr;
    LRUNode* next = nullptr;
};

/// Intrusive doubly-linked list.  Head = most-recently used, tail = least.
class LRUList {
public:
    LRUList() = default;
    ~LRUList();

    /// Move an existing node to the head (most-recently used).
    void move_to_head(LRUNode* node);

    /// Insert a brand-new node at the head.
    void insert_at_head(LRUNode* node);

    /// Remove a node from the list (does NOT free it).
    void remove(LRUNode* node);

    /// Return the tail (least-recently used) node, or nullptr if empty.
    LRUNode* get_tail() const;

    size_t size() const { return size_; }

private:
    LRUNode* head_ = nullptr;
    LRUNode* tail_ = nullptr;
    size_t size_ = 0;
};

// ───────────────────────────────────────────────── Cache entry
struct CacheEntry {
    std::shared_ptr<IndexPartition> partition_data;
    std::atomic<int> pin_count{0};
    LRUNode* lru_node = nullptr;   // Owned by the entry; freed on destruction.
    LoadState load_state = LoadState::EMPTY;
    std::condition_variable load_cv;
    std::mutex mutex;

    CacheEntry() = default;
    ~CacheEntry();

    /// Increment pin count (caller must hold entry lock or ensure LOADED).
    void pin()  { pin_count.fetch_add(1, std::memory_order_relaxed); }
    /// Decrement pin count.
    void unpin() { pin_count.fetch_sub(1, std::memory_order_relaxed); }
};

// ───────────────────────────────────────────────── Request types
enum class RequestType {
    LOAD,           // Cache miss — fire async S3 fetch.
    LOAD_COMPLETE,  // S3 callback — data has arrived.
    ACCESS,         // Cache hit  — update LRU ordering.
    PREFETCH,       // Batch prefetch — fire async S3 fetches.
    STOP            // Shutdown sentinel.
};

struct CacheRequest {
    RequestType type;
    size_t partition_id = 0;
    std::vector<size_t> partition_ids;             // Used only for PREFETCH.
    std::shared_ptr<IndexPartition> loaded_data;   // Used only for LOAD_COMPLETE.
};

// ───────────────────────────────────────────────── Cache stats
struct CacheStats {
    std::atomic<int64_t> hits{0};
    std::atomic<int64_t> misses{0};
    std::atomic<int64_t> evictions{0};
    std::atomic<int64_t> total_lookups{0};
    std::atomic<int64_t> s3_load_time_ns{0};   // Total S3 download time (ns).
    std::atomic<int64_t> n_s3_downloads{0};     // Total number of S3 fetches.

    void reset() {
        hits.store(0, std::memory_order_relaxed);
        misses.store(0, std::memory_order_relaxed);
        evictions.store(0, std::memory_order_relaxed);
        total_lookups.store(0, std::memory_order_relaxed);
        s3_load_time_ns.store(0, std::memory_order_relaxed);
        n_s3_downloads.store(0, std::memory_order_relaxed);
    }
};

// ───────────────────────────────────────────────── DataStore interface
/// Abstract interface for loading partition data.  Concrete implementations
/// wrap S3 or local-disk backends.
class DataStore {
public:
    virtual ~DataStore() = default;
    /// Synchronously load a single partition.  Returns nullptr on failure.
    virtual std::shared_ptr<IndexPartition> load_partition(size_t partition_id) = 0;

    /// Callback type for async loads: (partition_id, data_or_nullptr, elapsed_ns).
    using LoadCallback = std::function<void(size_t partition_id,
                                           std::shared_ptr<IndexPartition> data,
                                           int64_t elapsed_ns)>;

    /// Asynchronously load a single partition.  Default falls back to sync.
    virtual void async_load_partition(size_t partition_id, LoadCallback callback) {
        auto data = load_partition(partition_id);
        callback(partition_id, std::move(data), 0);
    }

    /// Asynchronously load multiple partitions.  Default calls async_load_partition
    /// for each.
    virtual void async_load_partitions(const std::vector<size_t>& partition_ids,
                                       LoadCallback callback) {
        for (size_t pid : partition_ids) {
            async_load_partition(pid, callback);
        }
    }
};

// ───────────────────────────────────────────────── S3 DataStore
#ifdef QUAKE_USE_S3

/// Loads partitions from S3 using the same format as DynamicInvertedLists.
class S3DataStore : public DataStore {
public:
    S3DataStore(std::shared_ptr<Aws::S3::S3Client> client,
                std::string bucket,
                std::string prefix,
                int64_t code_size,
                const std::unordered_map<size_t, size_t>& num_vectors_map);

    /// Synchronous load (blocking).
    std::shared_ptr<IndexPartition> load_partition(size_t partition_id) override;

    /// Fire-and-forget async load using GetObjectAsync.
    void async_load_partition(size_t partition_id, LoadCallback callback) override;

    /// Batch async load — fires GetObjectAsync for each partition.
    void async_load_partitions(const std::vector<size_t>& partition_ids,
                               LoadCallback callback) override;

private:
    std::shared_ptr<Aws::S3::S3Client> client_;
    std::string bucket_;
    std::string prefix_;
    int64_t code_size_;
    const std::unordered_map<size_t, size_t>& num_vectors_map_;

    std::string partition_key(size_t pid) const {
        return prefix_ + "/partition_" + std::to_string(pid);
    }
};
#endif  // QUAKE_USE_S3

// ───────────────────────────────────────────────── CacheManager
class CacheManager {
public:
    /// @param capacity       Maximum number of partitions to cache (0 = unlimited/disabled).
    /// @param eviction_threshold Fraction of capacity at which eviction starts (e.g. 0.9).
    /// @param datastore      DataStore implementation for fetching partitions.
    CacheManager(size_t capacity,
                 float eviction_threshold,
                 std::shared_ptr<DataStore> datastore);

    ~CacheManager();

    // ── Lifecycle ──────────────────────────────────────────────────────
    /// Start the background cache management thread.
    void start();
    /// Signal the background thread to stop and join it.
    void stop();

    // ── Worker-thread API ──────────────────────────────────────────────
    /// Look up a partition.  On hit, pins and returns the entry immediately.
    /// On miss, blocks until the partition is loaded (or an error occurs).
    /// The caller MUST call release() when done reading the partition data.
    std::shared_ptr<CacheEntry> get(size_t partition_id);

    /// Release (unpin) a previously get()-ed partition.
    void release(size_t partition_id);

    /// Non-blocking batch prefetch.  Enqueues partitions for background loading.
    void prefetch(const std::vector<size_t>& partition_ids);

    /// Invalidate a cached entry (e.g. after a mutation).  Waits for pin_count
    /// to reach zero before evicting.
    void invalidate(size_t partition_id);

    // ── Stats ──────────────────────────────────────────────────────────
    const CacheStats& stats() const { return stats_; }
    size_t size() const;

private:
    // ── Background thread ──────────────────────────────────────────────
    void cache_management_loop();
    void process_load(size_t partition_id);
    void process_load_complete(size_t partition_id,
                               std::shared_ptr<IndexPartition> data);
    void process_access(size_t partition_id);
    void process_prefetch(const std::vector<size_t>& partition_ids);
    void evict();

    // ── Helpers ────────────────────────────────────────────────────────
    void enqueue(CacheRequest req);
    /// Fire an async S3 request.  The callback enqueues a LOAD_COMPLETE
    /// back into this CacheManager's queue.
    void fire_async_load(size_t partition_id);

    // ── State ──────────────────────────────────────────────────────────
    size_t capacity_;
    float eviction_threshold_;
    std::shared_ptr<DataStore> datastore_;

    // Cache table: partition_id → entry.  Protected by table_mutex_.
    std::unordered_map<size_t, std::shared_ptr<CacheEntry>> cache_table_;
    mutable std::mutex table_mutex_;

    // LRU list (only touched by the background thread, so no extra lock needed
    // beyond the fact that only the bg thread calls LRU methods).
    LRUList lru_list_;

    // Request queue: multiple producers (worker threads + S3 callbacks),
    // single consumer (bg thread).
    std::queue<CacheRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Background thread.
    std::thread bg_thread_;
    std::atomic<bool> running_{false};

    CacheStats stats_;
};

}  // namespace quake

#endif  // LRU_CACHE_H
