// lru_cache.cpp — Implementation of the persistent LRU partition cache.

#include "lru_cache.h"
#include "index_partition.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>

#ifdef QUAKE_USE_S3
#include <aws/s3/model/GetObjectRequest.h>
#endif

namespace quake {

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

// ═══════════════════════════════════════════════════════════════════
//  LRUList
// ═══════════════════════════════════════════════════════════════════

LRUList::~LRUList() {
    // Nodes are owned by CacheEntry; we don't free them here.
    head_ = tail_ = nullptr;
    size_ = 0;
}

void LRUList::insert_at_head(LRUNode* node) {
    if (!node) return;
    node->prev = nullptr;
    node->next = head_;
    if (head_) {
        head_->prev = node;
    }
    head_ = node;
    if (!tail_) {
        tail_ = node;
    }
    ++size_;
}

void LRUList::move_to_head(LRUNode* node) {
    if (!node || node == head_) return;
    remove(node);
    insert_at_head(node);
}

void LRUList::remove(LRUNode* node) {
    if (!node) return;
    if (node->prev) {
        node->prev->next = node->next;
    } else {
        head_ = node->next;
    }
    if (node->next) {
        node->next->prev = node->prev;
    } else {
        tail_ = node->prev;
    }
    node->prev = nullptr;
    node->next = nullptr;
    --size_;
}

LRUNode* LRUList::get_tail() const {
    return tail_;
}

// ═══════════════════════════════════════════════════════════════════
//  CacheEntry
// ═══════════════════════════════════════════════════════════════════

CacheEntry::~CacheEntry() {
    delete lru_node;
    lru_node = nullptr;
}

// ═══════════════════════════════════════════════════════════════════
//  S3DataStore
// ═══════════════════════════════════════════════════════════════════

#ifdef QUAKE_USE_S3
S3DataStore::S3DataStore(std::shared_ptr<Aws::S3::S3Client> client,
                         std::string bucket,
                         std::string prefix,
                         int64_t code_size,
                         const std::unordered_map<size_t, size_t>& num_vectors_map)
    : client_(std::move(client)),
      bucket_(std::move(bucket)),
      prefix_(std::move(prefix)),
      code_size_(code_size),
      num_vectors_map_(num_vectors_map) {}

std::shared_ptr<IndexPartition> S3DataStore::load_partition(size_t partition_id) {
    auto nv_it = num_vectors_map_.find(partition_id);
    if (nv_it == num_vectors_map_.end()) {
        std::cerr << "[S3DataStore] partition " << partition_id
                  << " not found in manifest" << std::endl;
        return nullptr;
    }
    size_t nv = nv_it->second;
    size_t csize = nv * static_cast<size_t>(code_size_);
    size_t isize = nv * sizeof(int64_t);  // idx_t = int64_t

    std::string key = partition_key(partition_id);
    Aws::S3::Model::GetObjectRequest req;
    req.SetBucket(bucket_);
    req.SetKey(key);

    auto t0 = high_resolution_clock::now();
    auto outcome = client_->GetObject(req);
    int64_t elapsed = duration_cast<nanoseconds>(
        high_resolution_clock::now() - t0).count();

    if (!outcome.IsSuccess()) {
        std::cerr << "[S3DataStore] GetObject failed for key=" << key << ": "
                  << outcome.GetError().GetMessage() << std::endl;
        return nullptr;
    }

    auto& body = outcome.GetResult().GetBody();
    uint8_t* codes = new uint8_t[csize];
    int64_t* ids = new int64_t[nv];
    body.read(reinterpret_cast<char*>(codes), csize);
    body.read(reinterpret_cast<char*>(ids), isize);

    auto part = std::make_shared<IndexPartition>(
        static_cast<int64_t>(nv), codes, ids, code_size_);
    delete[] codes;
    delete[] ids;
    return part;
}

void S3DataStore::async_load_partition(size_t partition_id, LoadCallback callback) {
    auto nv_it = num_vectors_map_.find(partition_id);
    if (nv_it == num_vectors_map_.end()) {
        std::cerr << "[S3DataStore::async] partition " << partition_id
                  << " not found in manifest" << std::endl;
        callback(partition_id, nullptr, 0);
        return;
    }

    size_t nv = nv_it->second;
    size_t csize = nv * static_cast<size_t>(code_size_);
    size_t isize = nv * sizeof(int64_t);
    int64_t cs = code_size_;

    Aws::S3::Model::GetObjectRequest req;
    req.SetBucket(bucket_);
    req.SetKey(partition_key(partition_id));

    auto t0 = high_resolution_clock::now();

    client_->GetObjectAsync(req,
        [partition_id, nv, csize, isize, cs, callback, t0]
        (const Aws::S3::S3Client*,
         const Aws::S3::Model::GetObjectRequest&,
         Aws::S3::Model::GetObjectOutcome outcome,
         const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) {

            int64_t elapsed = duration_cast<nanoseconds>(
                high_resolution_clock::now() - t0).count();

            if (!outcome.IsSuccess()) {
                std::cerr << "[S3DataStore::async] GetObject failed for partition "
                          << partition_id << ": "
                          << outcome.GetError().GetMessage() << std::endl;
                callback(partition_id, nullptr, elapsed);
                return;
            }

            auto& body = outcome.GetResult().GetBody();
            uint8_t* codes = new uint8_t[csize];
            int64_t* ids = new int64_t[nv];
            body.read(reinterpret_cast<char*>(codes), csize);
            body.read(reinterpret_cast<char*>(ids), isize);

            auto part = std::make_shared<IndexPartition>(
                static_cast<int64_t>(nv), codes, ids, cs);
            delete[] codes;
            delete[] ids;

            callback(partition_id, std::move(part), elapsed);
        }, nullptr);
}

void S3DataStore::async_load_partitions(const std::vector<size_t>& partition_ids,
                                         LoadCallback callback) {
    for (size_t pid : partition_ids) {
        async_load_partition(pid, callback);
    }
}
#endif  // QUAKE_USE_S3

// ═══════════════════════════════════════════════════════════════════
//  CacheManager
// ═══════════════════════════════════════════════════════════════════

CacheManager::CacheManager(size_t capacity,
                           float eviction_threshold,
                           std::shared_ptr<DataStore> datastore)
    : capacity_(capacity),
      eviction_threshold_(eviction_threshold),
      datastore_(std::move(datastore)) {}

CacheManager::~CacheManager() {
    stop();
}

// ── Lifecycle ──────────────────────────────────────────────────────

void CacheManager::start() {
    if (running_.load(std::memory_order_acquire)) return;
    running_.store(true, std::memory_order_release);
    bg_thread_ = std::thread(&CacheManager::cache_management_loop, this);
}

void CacheManager::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    // Enqueue a STOP sentinel.
    enqueue(CacheRequest{RequestType::STOP, 0, {}, nullptr});
    if (bg_thread_.joinable()) {
        bg_thread_.join();
    }
    running_.store(false, std::memory_order_release);
}

// ── Worker-thread API ──────────────────────────────────────────────

std::shared_ptr<CacheEntry> CacheManager::get(size_t partition_id) {
    stats_.total_lookups.fetch_add(1, std::memory_order_relaxed);

    std::shared_ptr<CacheEntry> entry;

    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        auto it = cache_table_.find(partition_id);
        if (it != cache_table_.end()) {
            entry = it->second;
        }
    }

    if (entry) {
        // Potential cache hit — check state.
        std::unique_lock<std::mutex> lk(entry->mutex);

        if (entry->load_state == LoadState::LOADED) {
            // Fast path: cache hit.
            entry->pin();
            stats_.hits.fetch_add(1, std::memory_order_relaxed);
            lk.unlock();

            // Enqueue ACCESS for LRU update (non-blocking).
            enqueue(CacheRequest{RequestType::ACCESS, partition_id, {}, nullptr});
            return entry;
        }

        if (entry->load_state == LoadState::LOADING) {
            // Another thread is already loading this partition.  Wait.
            stats_.misses.fetch_add(1, std::memory_order_relaxed);
            entry->load_cv.wait(lk, [&entry] {
                return entry->load_state == LoadState::LOADED ||
                       entry->load_state == LoadState::ERROR;
            });
            if (entry->load_state == LoadState::LOADED) {
                entry->pin();
                return entry;
            }
            // ERROR — fall through and attempt reload below.
        }
    }

    // Cache miss: create or re-use an entry and request a LOAD.
    stats_.misses.fetch_add(entry ? 0 : 1, std::memory_order_relaxed);

    if (!entry) {
        entry = std::make_shared<CacheEntry>();
        std::lock_guard<std::mutex> lk(table_mutex_);
        // Double-check: another thread may have created it.
        auto it = cache_table_.find(partition_id);
        if (it != cache_table_.end()) {
            entry = it->second;
        } else {
            cache_table_[partition_id] = entry;
        }
    }

    {
        std::unique_lock<std::mutex> lk(entry->mutex);

        // Re-check after acquiring the entry lock.
        if (entry->load_state == LoadState::LOADED) {
            entry->pin();
            lk.unlock();
            enqueue(CacheRequest{RequestType::ACCESS, partition_id, {}, nullptr});
            return entry;
        }
        if (entry->load_state == LoadState::LOADING) {
            entry->load_cv.wait(lk, [&entry] {
                return entry->load_state == LoadState::LOADED ||
                       entry->load_state == LoadState::ERROR;
            });
            if (entry->load_state == LoadState::LOADED) {
                entry->pin();
                return entry;
            }
            // ERROR — the bg thread will retry.
        }

        // Mark as LOADING so other threads know to wait.
        entry->load_state = LoadState::LOADING;
    }

    // Enqueue the LOAD request for the background thread.
    enqueue(CacheRequest{RequestType::LOAD, partition_id, {}, nullptr});

    // Wait for the async S3 callback to complete the load.
    {
        std::unique_lock<std::mutex> lk(entry->mutex);
        entry->load_cv.wait(lk, [&entry] {
            return entry->load_state == LoadState::LOADED ||
                   entry->load_state == LoadState::ERROR;
        });
        if (entry->load_state == LoadState::LOADED) {
            entry->pin();
            return entry;
        }
    }

    // Load failed — return the entry anyway (caller must check partition_data).
    return entry;
}

void CacheManager::release(size_t partition_id) {
    std::shared_ptr<CacheEntry> entry;
    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        auto it = cache_table_.find(partition_id);
        if (it == cache_table_.end()) return;
        entry = it->second;
    }
    if (entry) {
        entry->unpin();
    }
}

void CacheManager::prefetch(const std::vector<size_t>& partition_ids) {
    if (partition_ids.empty()) return;

    // Deduplicate and filter out already-loaded/loading entries.
    std::vector<size_t> to_fetch;
    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        for (size_t pid : partition_ids) {
            auto it = cache_table_.find(pid);
            if (it == cache_table_.end()) {
                to_fetch.push_back(pid);
            } else {
                std::lock_guard<std::mutex> elk(it->second->mutex);
                if (it->second->load_state != LoadState::LOADED &&
                    it->second->load_state != LoadState::LOADING) {
                    to_fetch.push_back(pid);
                }
            }
        }
    }

    if (!to_fetch.empty()) {
        enqueue(CacheRequest{RequestType::PREFETCH, 0, std::move(to_fetch), nullptr});
    }
}

void CacheManager::invalidate(size_t partition_id) {
    std::shared_ptr<CacheEntry> entry;
    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        auto it = cache_table_.find(partition_id);
        if (it == cache_table_.end()) return;
        entry = it->second;
        cache_table_.erase(it);
    }
    if (entry) {
        std::lock_guard<std::mutex> lk(entry->mutex);
        if (entry->lru_node) {
            lru_list_.remove(entry->lru_node);
        }
        entry->partition_data.reset();
        entry->load_state = LoadState::EMPTY;
    }
}

size_t CacheManager::size() const {
    std::lock_guard<std::mutex> lk(table_mutex_);
    return cache_table_.size();
}

// ── Background thread ──────────────────────────────────────────────

void CacheManager::enqueue(CacheRequest req) {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        request_queue_.push(std::move(req));
    }
    queue_cv_.notify_one();
}

void CacheManager::cache_management_loop() {
    while (true) {
        CacheRequest req;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [this] { return !request_queue_.empty(); });
            req = std::move(request_queue_.front());
            request_queue_.pop();
        }

        switch (req.type) {
            case RequestType::LOAD:
                process_load(req.partition_id);
                break;
            case RequestType::LOAD_COMPLETE:
                process_load_complete(req.partition_id, std::move(req.loaded_data));
                break;
            case RequestType::ACCESS:
                process_access(req.partition_id);
                break;
            case RequestType::PREFETCH:
                process_prefetch(req.partition_ids);
                break;
            case RequestType::STOP:
                return;
        }

        // After each operation, check if eviction is needed.
        if (capacity_ > 0) {
            evict();
        }
    }
}

void CacheManager::fire_async_load(size_t partition_id) {
    // Capture a raw pointer to `this` — safe because we join the bg thread
    // (and thus all outstanding S3 callbacks) before destruction.
    datastore_->async_load_partition(partition_id,
        [this](size_t pid, std::shared_ptr<IndexPartition> data, int64_t elapsed_ns) {
            // Record S3 timing stats.
            stats_.s3_load_time_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
            stats_.n_s3_downloads.fetch_add(1, std::memory_order_relaxed);

            // Enqueue a LOAD_COMPLETE event back into our queue.
            CacheRequest completion;
            completion.type = RequestType::LOAD_COMPLETE;
            completion.partition_id = pid;
            completion.loaded_data = std::move(data);
            enqueue(std::move(completion));
        });
}

void CacheManager::process_load(size_t partition_id) {
    std::shared_ptr<CacheEntry> entry;
    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        auto it = cache_table_.find(partition_id);
        if (it == cache_table_.end()) return;
        entry = it->second;
    }
    if (!entry) return;

    {
        std::lock_guard<std::mutex> lk(entry->mutex);
        // If already loaded (e.g. concurrent load), just notify.
        if (entry->load_state == LoadState::LOADED) {
            entry->load_cv.notify_all();
            return;
        }
    }

    // Fire async S3 request — returns immediately!
    // The callback will enqueue a LOAD_COMPLETE event.
    fire_async_load(partition_id);
}

void CacheManager::process_load_complete(size_t partition_id,
                                          std::shared_ptr<IndexPartition> data) {
    std::shared_ptr<CacheEntry> entry;
    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        auto it = cache_table_.find(partition_id);
        if (it == cache_table_.end()) return;
        entry = it->second;
    }
    if (!entry) return;

    {
        std::lock_guard<std::mutex> lk(entry->mutex);
        if (data) {
            entry->partition_data = std::move(data);

            // Create and insert LRU node.
            if (!entry->lru_node) {
                entry->lru_node = new LRUNode();
                entry->lru_node->partition_id = partition_id;
            }
            lru_list_.insert_at_head(entry->lru_node);

            entry->load_state = LoadState::LOADED;
        } else {
            entry->load_state = LoadState::ERROR;
        }
        entry->load_cv.notify_all();
    }
}

void CacheManager::process_access(size_t partition_id) {
    std::shared_ptr<CacheEntry> entry;
    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        auto it = cache_table_.find(partition_id);
        if (it == cache_table_.end()) return;
        entry = it->second;
    }
    if (entry && entry->lru_node) {
        lru_list_.move_to_head(entry->lru_node);
    }
}

void CacheManager::process_prefetch(const std::vector<size_t>& partition_ids) {
    for (size_t pid : partition_ids) {
        // Ensure an entry exists in the table.
        std::shared_ptr<CacheEntry> entry;
        {
            std::lock_guard<std::mutex> lk(table_mutex_);
            auto it = cache_table_.find(pid);
            if (it != cache_table_.end()) {
                entry = it->second;
                std::lock_guard<std::mutex> elk(entry->mutex);
                if (entry->load_state == LoadState::LOADED ||
                    entry->load_state == LoadState::LOADING) {
                    continue;  // Already loaded or in progress.
                }
            } else {
                entry = std::make_shared<CacheEntry>();
                cache_table_[pid] = entry;
            }
        }

        {
            std::lock_guard<std::mutex> lk(entry->mutex);
            if (entry->load_state == LoadState::LOADED ||
                entry->load_state == LoadState::LOADING) {
                continue;
            }
            entry->load_state = LoadState::LOADING;
        }

        // Fire async S3 request — returns immediately!
        fire_async_load(pid);
    }
}

void CacheManager::evict() {
    // Only evict when we exceed threshold.
    size_t current_size;
    {
        std::lock_guard<std::mutex> lk(table_mutex_);
        current_size = cache_table_.size();
    }

    size_t target = static_cast<size_t>(capacity_ * eviction_threshold_);
    if (current_size <= target) return;

    // Walk from tail (least recently used) and evict unpinned entries.
    while (current_size > target) {
        LRUNode* victim_node = lru_list_.get_tail();
        if (!victim_node) break;

        size_t victim_pid = victim_node->partition_id;
        std::shared_ptr<CacheEntry> victim;
        {
            std::lock_guard<std::mutex> lk(table_mutex_);
            auto it = cache_table_.find(victim_pid);
            if (it == cache_table_.end()) {
                // Node is stale — remove it from the LRU list and continue.
                lru_list_.remove(victim_node);
                continue;
            }
            victim = it->second;
        }

        {
            std::lock_guard<std::mutex> lk(victim->mutex);
            if (victim->pin_count.load(std::memory_order_relaxed) != 0 ||
                victim->load_state != LoadState::LOADED) {
                // Can't evict a pinned or non-loaded entry.
                // Move it to head to avoid re-visiting it immediately.
                lru_list_.move_to_head(victim_node);
                break;  // Stop eviction for this round.
            }

            // Evict: clear data, remove from LRU.
            lru_list_.remove(victim_node);
            victim->partition_data.reset();
            victim->load_state = LoadState::EMPTY;
        }

        {
            std::lock_guard<std::mutex> lk(table_mutex_);
            cache_table_.erase(victim_pid);
            current_size = cache_table_.size();
        }

        stats_.evictions.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace quake
