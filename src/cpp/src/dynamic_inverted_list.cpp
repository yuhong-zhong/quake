// dynamic_inverted_list.cpp

#include "dynamic_inverted_list.h"
#include <iostream>
#include <fstream>

#ifdef QUAKE_USE_S3
#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/core/client/ClientConfiguration.h>
#include <sstream>

static std::once_flag aws_init_flag;
static Aws::SDKOptions aws_sdk_options;

static void ensure_aws_initialized() {
    std::call_once(aws_init_flag, []() {
        Aws::InitAPI(aws_sdk_options);
    });
}
#endif

namespace faiss {
    ArrayInvertedLists *convert_to_array_invlists(DynamicInvertedLists *invlists,
                                                  std::unordered_map<size_t, size_t> &remap_ids) {
        auto ret = new ArrayInvertedLists(invlists->nlist, invlists->code_size);

        // iterate over all partitions
        size_t new_list_no = 0;
        for (auto &p: invlists->partitions_) {
            size_t old_list_no = p.first;
            shared_ptr<IndexPartition> part = p.second;

            if (part->num_vectors_ > 0) {
                ret->add_entries(new_list_no, part->num_vectors_, part->ids_, part->codes_);
            }
            remap_ids[old_list_no] = new_list_no;
            new_list_no += 1;
        }

        return ret;
    }

    DynamicInvertedLists *convert_from_array_invlists(ArrayInvertedLists *invlists) {
        auto ret = new DynamicInvertedLists(invlists->nlist, invlists->code_size);
        for (size_t list_no = 0; list_no < invlists->nlist; list_no++) {
            size_t list_size = invlists->list_size(list_no);
            if (list_size > 0) {
                ret->add_entries(list_no, list_size, invlists->get_ids(list_no), invlists->get_codes(list_no));
            } else {
                ret->add_list(list_no); // ensure partition exists even if empty
            }
        }
        return ret;
    }


    DynamicInvertedLists::DynamicInvertedLists(size_t nlist, size_t code_size)
        : InvertedLists(nlist, code_size) {
        d_ = code_size / sizeof(float);
        code_size_ = code_size;
        // Initialize empty partitions
        for (size_t i = 0; i < nlist; i++) {
            // IndexPartition ip;
            shared_ptr<IndexPartition> ip = std::make_shared<IndexPartition>();
            ip->set_code_size(code_size);
            partitions_[i] = ip;
        }
        curr_list_id_ = nlist;
    }

    DynamicInvertedLists::~DynamicInvertedLists() {
        // partitions_ will clean themselves up as IndexPartition destructor frees memory
    }

    size_t DynamicInvertedLists::ntotal() const {
        if (s3_mode_) {
            size_t ntotal = 0;
            for (auto &kv: s3_num_vectors_) {
                ntotal += kv.second;
            }
            return ntotal;
        }
        size_t ntotal = 0;
        for (auto &kv: partitions_) {
            ntotal += kv.second->num_vectors_;
        }
        return ntotal;
    }

    size_t DynamicInvertedLists::list_size(size_t list_no) const {
        if (s3_mode_) {
            auto it = s3_num_vectors_.find(list_no);
            if (it == s3_num_vectors_.end()) {
                throw std::runtime_error("S3 partition " + std::to_string(list_no) + " not in manifest");
            }
            return it->second;
        }
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " does not exist in list_size";
            throw std::runtime_error(err_message);
        }
        return static_cast<size_t>(it->second->num_vectors_);
    }

    const uint8_t *DynamicInvertedLists::get_codes(size_t list_no) const {
        if (s3_mode_) {
            // Check partitions_ first: mutations materialize there temporarily.
            auto pit = partitions_.find(list_no);
            if (pit != partitions_.end()) return pit->second->codes_;

            // If LRU cache is enabled, use it instead of temp_s3_.
            if (cache_manager_) {
                auto entry = cache_manager_->get(list_no);
                if (entry && entry->partition_data) {
                    return entry->partition_data->codes_;
                }
                throw std::runtime_error("Cache failed to load partition " +
                                         std::to_string(list_no));
            }

            std::lock_guard<std::mutex> lk(temp_s3_mutex_);
            if (!temp_s3_.count(list_no)) {
                auto t0 = high_resolution_clock::now();
                temp_s3_[list_no] = s3_fetch_partition(list_no);
                int64_t elapsed = duration_cast<nanoseconds>(
                    high_resolution_clock::now() - t0).count();
                s3_load_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
                n_s3_downloads_.fetch_add(1, std::memory_order_relaxed);
            }
            return temp_s3_.at(list_no)->codes_;
        }
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " does not exist in get_codes";
            throw std::runtime_error(err_message);
        }
        return it->second->codes_;
    }

    const idx_t *DynamicInvertedLists::get_ids(size_t list_no) const {
        if (s3_mode_) {
            // Check partitions_ first: mutations materialize there temporarily.
            auto pit = partitions_.find(list_no);
            if (pit != partitions_.end()) return pit->second->ids_;

            // If LRU cache is enabled, use it (partition was already pinned by get_codes).
            if (cache_manager_) {
                auto entry = cache_manager_->get(list_no);
                if (entry && entry->partition_data) {
                    // This second get() increments pin_count again; the caller
                    // must call release_partition() once for the codes+ids pair.
                    return entry->partition_data->ids_;
                }
                throw std::runtime_error("Cache failed to load partition " +
                                         std::to_string(list_no));
            }

            std::lock_guard<std::mutex> lk(temp_s3_mutex_);
            // get_codes() must have been called first for this partition
            auto it = temp_s3_.find(list_no);
            if (it == temp_s3_.end()) {
                throw std::runtime_error("S3 partition " + std::to_string(list_no) +
                                         " not downloaded; call get_codes() first");
            }
            return it->second->ids_;
        }
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " does not exist in get_ids";
            throw std::runtime_error(err_message);
        }
        return it->second->ids_;
    }

    void DynamicInvertedLists::release_codes(size_t list_no, const uint8_t *codes) const {
        // No action needed because get_codes does not allocate new memory
    }

    void DynamicInvertedLists::release_ids(size_t list_no, const idx_t *ids) const {
        // No action needed because get_ids does not allocate new memory
    }

    shared_ptr<IndexPartition> DynamicInvertedLists::s3_fetch_partition(size_t pid) const {
#ifdef QUAKE_USE_S3
        string key = s3_partition_key(pid);
        Aws::S3::Model::GetObjectRequest req;
        req.SetBucket(s3_bucket_);
        req.SetKey(key);
        auto outcome = s3_client_->GetObject(req);
        if (!outcome.IsSuccess()) {
            throw std::runtime_error(
                "S3 GetObject failed for key=" + key + ": " +
                outcome.GetError().GetMessage().c_str());
        }
        auto& body = outcome.GetResult().GetBody();
        size_t nv = s3_num_vectors_.at(pid);
        size_t csize = nv * static_cast<size_t>(code_size);
        size_t isize = nv * sizeof(idx_t);
        uint8_t *codes = new uint8_t[csize];
        idx_t   *ids   = new idx_t[nv];
        body.read(reinterpret_cast<char*>(codes), csize);
        body.read(reinterpret_cast<char*>(ids),   isize);
        auto part = std::make_shared<IndexPartition>(nv, codes, ids, code_size);
        delete[] codes;
        delete[] ids;
        return part;
#else
        throw std::runtime_error("Quake was built without S3 support (QUAKE_USE_S3 not set).");
#endif
    }

    void DynamicInvertedLists::prefetch_partitions(const std::vector<size_t>& pids) const {
        if (!s3_mode_ || pids.empty()) return;

        // If LRU cache is enabled, delegate entirely to it.
        if (cache_manager_) {
            cache_manager_->prefetch(pids);
            return;
        }

#ifdef QUAKE_USE_S3
        // Filter to partitions not yet cached.
        std::vector<size_t> to_fetch;
        {
            std::lock_guard<std::mutex> lk(temp_s3_mutex_);
            for (size_t pid : pids)
                if (!temp_s3_.count(pid)) to_fetch.push_back(pid);
        }
        if (to_fetch.empty()) return;

        const size_t n = to_fetch.size();
        std::vector<std::shared_ptr<IndexPartition>> results(n);
        std::atomic<size_t> n_done{0};
        std::mutex wait_mutex;
        std::condition_variable wait_cv;

        auto wall_t0 = high_resolution_clock::now();

        for (size_t i = 0; i < n; i++) {
            size_t pid = to_fetch[i];
            size_t nv = s3_num_vectors_.at(pid);
            size_t csize = nv * static_cast<size_t>(code_size);
            size_t isize = nv * sizeof(idx_t);
            int64_t cs = static_cast<int64_t>(code_size);

            Aws::S3::Model::GetObjectRequest req;
            req.SetBucket(s3_bucket_);
            req.SetKey(s3_partition_key(pid));

            s3_client_->GetObjectAsync(req,
                [i, nv, csize, isize, cs, n, &results, &n_done, &wait_mutex, &wait_cv]
                (const Aws::S3::S3Client*,
                 const Aws::S3::Model::GetObjectRequest&,
                 Aws::S3::Model::GetObjectOutcome outcome,
                 const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) {
                    if (outcome.IsSuccess()) {
                        auto& body = outcome.GetResult().GetBody();
                        uint8_t* codes = new uint8_t[csize];
                        idx_t*   ids   = new idx_t[nv];
                        body.read(reinterpret_cast<char*>(codes), csize);
                        body.read(reinterpret_cast<char*>(ids),   isize);
                        results[i] = std::make_shared<IndexPartition>(
                            static_cast<int64_t>(nv), codes, ids, cs);
                        delete[] codes;
                        delete[] ids;
                    }
                    if (n_done.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
                        std::lock_guard<std::mutex> lk(wait_mutex);
                        wait_cv.notify_one();
                    }
                }, nullptr);
        }

        {
            std::unique_lock<std::mutex> lk(wait_mutex);
            wait_cv.wait(lk, [&n_done, n] {
                return n_done.load(std::memory_order_acquire) == n;
            });
        }

        int64_t elapsed = duration_cast<nanoseconds>(
            high_resolution_clock::now() - wall_t0).count();

        {
            std::lock_guard<std::mutex> lk(temp_s3_mutex_);
            for (size_t i = 0; i < n; i++) {
                if (results[i] && !temp_s3_.count(to_fetch[i]))
                    temp_s3_[to_fetch[i]] = results[i];
            }
        }
        s3_load_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
        n_s3_downloads_.fetch_add(static_cast<int64_t>(n), std::memory_order_relaxed);
#endif
    }

    // ── S3 mutation helpers ──────────────────────────────────────────────────

    void DynamicInvertedLists::s3_ensure_partition_loaded(size_t pid) {
#ifdef QUAKE_USE_S3
        if (partitions_.count(pid)) return;  // already materialized

        shared_ptr<IndexPartition> part;
        {
            std::lock_guard<std::mutex> lk(temp_s3_mutex_);
            auto it = temp_s3_.find(pid);
            if (it != temp_s3_.end()) {
                part = it->second;
                temp_s3_.erase(it);
            }
        }
        if (!part) {
            part = s3_fetch_partition(pid);
        }
        partitions_[pid] = part;
        // id_to_location_ is not maintained in S3 mode (see map_add guard).
#else
        throw std::runtime_error("Quake was built without S3 support (QUAKE_USE_S3 not set).");
#endif
    }

    void DynamicInvertedLists::s3_upload_partition(size_t pid) {
#ifdef QUAKE_USE_S3
        auto it = partitions_.find(pid);
        if (it == partitions_.end()) {
            throw std::runtime_error("s3_upload_partition: partition " +
                                     std::to_string(pid) + " not in partitions_");
        }
        auto& part = it->second;
        size_t nv    = static_cast<size_t>(part->num_vectors_);
        size_t csize = nv * static_cast<size_t>(code_size);
        size_t isize = nv * sizeof(idx_t);

        auto ss = Aws::MakeShared<Aws::StringStream>("quake-s3-upload");
        if (nv > 0) {
            ss->write(reinterpret_cast<const char*>(part->codes_), csize);
            ss->write(reinterpret_cast<const char*>(part->ids_),   isize);
        }

        std::string key = s3_partition_key(pid);
        Aws::S3::Model::PutObjectRequest req;
        req.SetBucket(s3_bucket_);
        req.SetKey(key);
        req.SetBody(ss);
        req.SetContentLength(static_cast<long long>(csize + isize));

        auto outcome = s3_client_->PutObject(req);
        if (!outcome.IsSuccess()) {
            throw std::runtime_error(
                "S3 PutObject failed for key=" + key + ": " +
                outcome.GetError().GetMessage().c_str());
        }
        s3_num_vectors_[pid] = nv;
#else
        throw std::runtime_error("Quake was built without S3 support (QUAKE_USE_S3 not set).");
#endif
    }

    void DynamicInvertedLists::s3_delete_partition(size_t pid) {
#ifdef QUAKE_USE_S3
        std::string key = s3_partition_key(pid);
        Aws::S3::Model::DeleteObjectRequest req;
        req.SetBucket(s3_bucket_);
        req.SetKey(key);
        auto outcome = s3_client_->DeleteObject(req);
        if (!outcome.IsSuccess()) {
            throw std::runtime_error(
                "S3 DeleteObject failed for key=" + key + ": " +
                outcome.GetError().GetMessage().c_str());
        }
        s3_num_vectors_.erase(pid);
#else
        throw std::runtime_error("Quake was built without S3 support (QUAKE_USE_S3 not set).");
#endif
    }

    void DynamicInvertedLists::s3_evict_partition(size_t pid) {
        partitions_.erase(pid);
    }

    void DynamicInvertedLists::ensure_partition_loaded(size_t pid) {
        if (s3_mode_) {
            // Invalidate cached entry before mutating — we'll re-upload after.
            if (cache_manager_) cache_manager_->invalidate(pid);
            s3_ensure_partition_loaded(pid);
        }
    }

    void DynamicInvertedLists::flush_partition(size_t pid) {
        if (s3_mode_) s3_upload_partition(pid);
    }

    void DynamicInvertedLists::evict_partition(size_t pid) {
        if (s3_mode_) {
            // Also invalidate the cache entry so stale data isn't served.
            if (cache_manager_) cache_manager_->invalidate(pid);
            s3_evict_partition(pid);
        }
    }

    void DynamicInvertedLists::release_partition(size_t list_no) const {
        if (cache_manager_) {
            // Unpin twice: once for get_codes(), once for get_ids().
            cache_manager_->release(list_no);
            cache_manager_->release(list_no);
        }
    }

    void DynamicInvertedLists::init_cache(size_t capacity, float eviction_threshold) {
        if (!s3_mode_) {
            std::cerr << "[DynamicInvertedLists::init_cache] Cache only supported in S3 mode." << std::endl;
            return;
        }
#ifdef QUAKE_USE_S3
        auto datastore = std::make_shared<quake::S3DataStore>(
            s3_client_, s3_bucket_, s3_prefix_,
            static_cast<int64_t>(code_size), s3_num_vectors_);
        cache_manager_ = std::make_shared<quake::CacheManager>(
            capacity, eviction_threshold, datastore);
        cache_manager_->start();
        std::cout << "[DynamicInvertedLists] LRU cache initialized: capacity="
                  << capacity << ", eviction_threshold=" << eviction_threshold
                  << std::endl;
#else
        throw std::runtime_error("Quake was built without S3 support (QUAKE_USE_S3 not set).");
#endif
    }

    // ────────────────────────────────────────────────────────────────────────

    void DynamicInvertedLists::remove_entry(size_t list_no, idx_t id) {
        if (s3_mode_) s3_ensure_partition_loaded(list_no);
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " does not exist in remove_entry";
            throw std::runtime_error(err_message);
        }

        auto& part = it->second;
        bool modified = false;
        int64_t pos = (part->num_vectors_ > 0) ? part->find_id(id) : -1;
        if (pos != -1) {
            int64_t swapped = part->remove(pos);
            map_erase(id);
            if (swapped != -1) {                      // someone moved into `pos`
                idx_t moved_id = part->ids_[pos];
                map_swap(part.get(), pos, moved_id);
            }
            modified = true;
        }
        if (s3_mode_) {
            if (modified) s3_upload_partition(list_no);
            s3_evict_partition(list_no);
        }
    }

    void DynamicInvertedLists::remove_entries_from_partition(size_t list_no, vector<idx_t> vectors_to_remove) {
        if (s3_mode_) s3_ensure_partition_loaded(list_no);
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " does not exist in remove_entries_from_partition";
            throw std::runtime_error(err_message);
        }
        shared_ptr<IndexPartition> part = it->second;

        // create set from vector for faster lookup
        std::set<idx_t> vectors_to_remove_set(vectors_to_remove.begin(), vectors_to_remove.end());

        // We'll perform removals by scanning and removing matches.
        // Because remove() swaps last element in, we must be careful with iteration.
        for (int64_t i = 0; i < part->num_vectors_;) {
            if (vectors_to_remove_set.count(part->ids_[i])) {
                idx_t victim   = part->ids_[i];
                int64_t swapped = part->remove(i);
                map_erase(victim);
                if (swapped != -1)
                    map_swap(part.get(), i, part->ids_[i]);
            } else {
                i++;
            }
        }
        if (s3_mode_) { s3_upload_partition(list_no); s3_evict_partition(list_no); }
    }

    void DynamicInvertedLists::remove_vectors(std::set<idx_t> vectors_to_remove) {
        if (s3_mode_) {
            // Stream one partition at a time to bound memory usage.
            // Early exit once all target IDs have been found.
            size_t remaining = vectors_to_remove.size();
            std::vector<size_t> all_pids;
            all_pids.reserve(s3_num_vectors_.size());
            for (auto& kv : s3_num_vectors_) all_pids.push_back(kv.first);

            for (size_t pid : all_pids) {
                if (remaining == 0) break;
                s3_ensure_partition_loaded(pid);
                auto& part = partitions_.at(pid);
                size_t size_before = static_cast<size_t>(part->num_vectors_);
                for (int64_t i = 0; i < part->num_vectors_;) {
                    if (vectors_to_remove.count(part->ids_[i])) {
                        idx_t victim    = part->ids_[i];
                        int64_t swapped = part->remove(i);
                        map_erase(victim);  // no-op in S3 mode
                        if (swapped != -1) map_swap(part.get(), i, part->ids_[i]);
                        remaining--;
                    } else {
                        i++;
                    }
                }
                if (static_cast<size_t>(part->num_vectors_) != size_before)
                    s3_upload_partition(pid);
                s3_evict_partition(pid);
            }
            return;
        }
        // Remove from all partitions
        for (auto &kv: partitions_) {
            shared_ptr<IndexPartition> part = kv.second;
            for (int64_t i = 0; i < part->num_vectors_;) {
                if (vectors_to_remove.count(part->ids_[i])) {
                    idx_t victim   = part->ids_[i];
                    int64_t swapped = part->remove(i);
                    map_erase(victim);
                    if (swapped != -1)
                        map_swap(part.get(), i, part->ids_[i]);
                } else {
                    i++;
                }
            }
        }
    }

    void DynamicInvertedLists::build_map() {
        if (s3_mode_) return;  // id_to_location_ not maintained in S3 mode
        id_to_location_.clear();
        for (auto& kv : partitions_) {
            IndexPartition* part = kv.second.get();
            for (int64_t i = 0; i < part->num_vectors_; i++) {
                id_to_location_[part->ids_[i]] = {part, i};
            }
        }
    }


    size_t DynamicInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t *ids,
        const uint8_t *codes) {
        if (n_entry == 0) return 0;

        if (s3_mode_) s3_ensure_partition_loaded(list_no);

        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " does not exist in add_entries";
            throw std::runtime_error(err_message);
        }

        auto& part = it->second;
        if (part->code_size_ != static_cast<int64_t>(code_size))
            part->set_code_size(static_cast<int64_t>(code_size));

        const int64_t base = part->num_vectors_;      // size *before* append
        part->append(static_cast<int64_t>(n_entry), ids, codes);

        for (size_t i = 0; i < n_entry; ++i)
            map_add(part.get(), base + i, ids[i]);

        if (s3_mode_) { s3_upload_partition(list_no); s3_evict_partition(list_no); }
        return n_entry;
    }

    void DynamicInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t *ids,
        const uint8_t *codes) {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " does not exist in update_entries";
            throw std::runtime_error(err_message);
        }
        shared_ptr<IndexPartition> part = it->second;

        part->update((int64_t) offset, (int64_t) n_entry, ids, codes);
    }

void DynamicInvertedLists::batch_update_entries(
    size_t old_partition,
    int64_t* new_partitions,
    uint8_t* new_vectors,
    int64_t* new_ids,
    int num)
{
    /* 1. gather indices to move grouped by their *destination* */
    std::unordered_map<size_t, std::vector<int>> to_move;
    for (int i = 0; i < num; ++i) {
        size_t dst = static_cast<size_t>(new_partitions[i]);
        if (dst != old_partition)
            to_move[dst].push_back(i);
    }

    if (s3_mode_) {
        // Prefetch all needed partitions in parallel, then move to partitions_.
        std::vector<size_t> all_pids = {old_partition};
        for (auto& kv : to_move) all_pids.push_back(kv.first);
        prefetch_partitions(all_pids);
        for (size_t pid : all_pids) s3_ensure_partition_loaded(pid);
    }

    /* 2. FIRST remove them from the old partition
          (updates map_erase + map_swap) */
    auto old_it = partitions_.find(old_partition);
    if (old_it != partitions_.end()) {
        auto& old_part = old_it->second;
        for (auto& kv : to_move)
            for (int idx : kv.second) {
                idx_t victim   = static_cast<idx_t>(new_ids[idx]);
                int64_t pos    = old_part->find_id(victim);
                if (pos != -1) {
                    int64_t sw = old_part->remove(pos);
                    map_erase(victim);
                    if (sw != -1)
                        map_swap(old_part.get(), pos, old_part->ids_[pos]);
                }
            }
    }

    /* 3. THEN append them to their new partitions
          (updates map_add) */
    for (auto& kv : to_move) {
        size_t dst = kv.first;
        auto& idxs = kv.second;

        auto it = partitions_.find(dst);
        if (it == partitions_.end()) {
            add_list(dst);
            it = partitions_.find(dst);
        }
        auto& new_part = it->second;
        if (new_part->code_size_ != static_cast<int64_t>(code_size))
            new_part->set_code_size((int64_t)code_size);

        const int64_t base = new_part->num_vectors_;
        std::vector<idx_t>   ids_buf;   ids_buf.reserve(idxs.size());
        std::vector<uint8_t> codes_buf; codes_buf.reserve(idxs.size() * code_size);

        for (int j : idxs) {
            ids_buf  .push_back(static_cast<idx_t>(new_ids[j]));
            codes_buf.insert(codes_buf.end(),
                             new_vectors + j * code_size,
                             new_vectors + (j + 1) * code_size);
        }
        new_part->append((int64_t)idxs.size(), ids_buf.data(), codes_buf.data());

        for (size_t k = 0; k < idxs.size(); ++k)
            map_add(new_part.get(), base + k, ids_buf[k]);
    }

    if (s3_mode_) {
        s3_upload_partition(old_partition);
        s3_evict_partition(old_partition);
        for (auto& kv : to_move) {
            s3_upload_partition(kv.first);
            s3_evict_partition(kv.first);
        }
    }
}

    void DynamicInvertedLists::remove_list(size_t list_no) {
        if (s3_mode_) {
            partitions_.erase(list_no);  // evict if materialized (id_to_location_ not used)
            { std::lock_guard<std::mutex> lk(temp_s3_mutex_); temp_s3_.erase(list_no); }
            s3_delete_partition(list_no);  // deletes S3 object + erases from s3_num_vectors_
            nlist--;
            return;
        }
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            return;
        }

        for (int64_t i = 0; i < it->second->num_vectors_; i++) {
            map_erase(it->second->ids_[i]);
        }
        partitions_.erase(it);
        nlist--;
    }

    void DynamicInvertedLists::add_list(size_t list_no) {
        if (partitions_.find(list_no) != partitions_.end()) {
            string err_message = "List " + std::to_string(list_no) + " already exists in add_list";
            throw std::runtime_error(err_message);
        }
        if (s3_mode_ && s3_num_vectors_.count(list_no)) {
            throw std::runtime_error("List " + std::to_string(list_no) +
                                     " already exists in s3_num_vectors_ in add_list");
        }
        shared_ptr<IndexPartition> ip = std::make_shared<IndexPartition>();
        ip->set_code_size((int64_t) code_size);
        partitions_[list_no] = ip;
        nlist++;
        if (s3_mode_) {
            s3_num_vectors_[list_no] = 0;
            s3_upload_partition(list_no);  // upload empty object for S3 consistency
            s3_evict_partition(list_no);   // release immediately
        }
    }

    bool DynamicInvertedLists::id_in_list(size_t list_no, idx_t id) const {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            return false;
        }
        shared_ptr<IndexPartition> part = it->second;
        return part->find_id(id) != -1;
    }

    bool DynamicInvertedLists::get_vector_for_id(idx_t id, float *out) {
        if (id_to_location_.empty()) build_map();

        if (id < 0) {
            std::memset(out, 0, code_size_);
            return true;
        }

        auto it = id_to_location_.find(id);
        if (it == id_to_location_.end()) return false;

        IndexPartition* part = it->second.first;
        int64_t pos         = it->second.second;
        std::memcpy(out, part->codes_ + pos * part->code_size_, part->code_size_);
        return true;
    }

    vector<float*> DynamicInvertedLists::get_vectors_by_id(vector<int64_t> ids)
    {
        if (id_to_location_.empty()) build_map();

        vector<float*> ret; ret.reserve(ids.size());

        for (int64_t id : ids) {
            if (id < 0) {
                ret.push_back(nullptr);
                continue;
            }

            auto it = id_to_location_.find(id);
            if (it == id_to_location_.end())
                throw std::runtime_error("ID not found in any partition: " + std::to_string(id));

            IndexPartition* part = it->second.first;
            int64_t pos         = it->second.second;
            ret.push_back(reinterpret_cast<float*>(part->codes_ + pos * part->code_size_));
        }
        return ret;
    }

        // vector<float *> ret;
        // for (int64_t id : ids) {
        //     bool found = false;
        //     for (auto &kv: partitions_) {
        //         shared_ptr<IndexPartition> part = kv.second;
        //         int64_t pos = part->find_id(id);
        //         if (pos != -1) {
        //             ret.push_back(reinterpret_cast<float *>(part->codes_ + pos * part->code_size_));
        //             found = true;
        //             break;
        //         }
        //     }
        //     if (!found) {
        //         throw std::runtime_error("ID not found in any partition");
        //     }
        // }
        // return ret;
    // }

    size_t DynamicInvertedLists::get_new_list_id() {
        return curr_list_id_++;
    }

    void DynamicInvertedLists::reset() {
        partitions_.clear();
        id_to_location_.clear();
        nlist = 0;
        curr_list_id_ = 0;
    }

    void DynamicInvertedLists::resize(size_t nlist, size_t code_size) {
        // Not strictly needed because we use a map. But if required,
        // we can add or remove partitions. For now, do nothing.
    }

    void DynamicInvertedLists::save(const string &filename) {
        if (s3_mode_) {
            // S3 mode: write a manifest-only file (no chunk data).
            // Offsets are derived from s3_num_vectors_ so that metadata_only load
            // can recover the correct num_vectors for each partition.
            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs.is_open())
                throw std::runtime_error("Could not open file for writing: " + filename);

            std::vector<size_t> part_ids;
            part_ids.reserve(s3_num_vectors_.size());
            for (auto& kv : s3_num_vectors_) part_ids.push_back(kv.first);

            uint64_t num_partitions = static_cast<uint64_t>(part_ids.size());
            uint64_t record_size    = static_cast<uint64_t>(code_size) + sizeof(idx_t);

            // Build offsets array from s3_num_vectors_.
            std::vector<uint64_t> offsets(num_partitions + 1, 0ULL);
            for (uint64_t i = 0; i < num_partitions; i++)
                offsets[i + 1] = offsets[i] + s3_num_vectors_.at(part_ids[i]) * record_size;

            // Write header.
            ofs.write(reinterpret_cast<const char*>(&SerializationMagicNumber), sizeof(SerializationMagicNumber));
            ofs.write(reinterpret_cast<const char*>(&SerializationVersion),     sizeof(SerializationVersion));
            uint64_t nlist_64     = static_cast<uint64_t>(nlist);
            uint64_t code_size_64 = static_cast<uint64_t>(code_size);
            ofs.write(reinterpret_cast<const char*>(&nlist_64),        sizeof(nlist_64));
            ofs.write(reinterpret_cast<const char*>(&code_size_64),    sizeof(code_size_64));
            ofs.write(reinterpret_cast<const char*>(&num_partitions),  sizeof(num_partitions));
            // Write offsets array.
            ofs.write(reinterpret_cast<const char*>(offsets.data()),
                      offsets.size() * sizeof(uint64_t));
            // Write partition ID array.
            for (size_t i = 0; i < num_partitions; i++) {
                uint64_t pid_64 = static_cast<uint64_t>(part_ids[i]);
                ofs.write(reinterpret_cast<const char*>(&pid_64), sizeof(pid_64));
            }
            // No chunk data — vector data lives in S3.
            ofs.close();
            return;
        }

        /**
         * 1) Serialization Format:
         *    - 32-byte header:
         *        [ magic(4) | version(4) | nlist(8) | code_size(8) | num_partitions(8) ]
         *    - Offsets array (num_partitions + 1) of uint64_t
         *    - Partition ID array (num_partitions) of uint64_t
         *    - Concatenated chunks:
         *        For each partition i:
         *          [ codes (num_vectors * code_size) | ids (num_vectors * sizeof(idx_t)) ]
         *      Each chunk starts at offsets[i] (relative to start of chunks), ends at offsets[i+1].
         *
         * 2) Serialization Logic:
         *    - Gather partition IDs in a chosen order
         *    - Build offsets array by writing each partition’s codes/IDs
         *    - Write header
         *    - Write offsets array
         *    - Write partition ID array
         *    - Write partition chunks
         */
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + std::string(filename));
        }

        // Write header
        ofs.write(reinterpret_cast<const char *>(&SerializationMagicNumber), sizeof(SerializationMagicNumber));
        ofs.write(reinterpret_cast<const char *>(&SerializationVersion), sizeof(SerializationVersion));

        uint64_t nlist_64 = static_cast<uint64_t>(nlist);
        uint64_t code_size_64 = static_cast<uint64_t>(code_size);
        uint64_t num_partitions = static_cast<uint64_t>(partitions_.size());

        ofs.write(reinterpret_cast<const char *>(&nlist_64), sizeof(nlist_64));
        ofs.write(reinterpret_cast<const char *>(&code_size_64), sizeof(code_size_64));
        ofs.write(reinterpret_cast<const char *>(&num_partitions), sizeof(num_partitions));

        // Gather partition IDs
        vector<size_t> part_ids = vector<size_t>(partitions_.size());
        int i = 0;
        for (auto &kv: partitions_) {
            part_ids[i++] = kv.first;
        }
        // (Optional) sort(part_ids.begin(), part_ids.end());

        // Prepare offsets
        std::vector<uint64_t> offsets(num_partitions + 1, 0ULL);
        uint64_t offset_table_bytes = (num_partitions + 1) * sizeof(uint64_t);
        uint64_t partition_ids_bytes = num_partitions * sizeof(uint64_t);
        uint64_t start_of_chunks = 32 + offset_table_bytes + partition_ids_bytes;

        // Move file pointer to where chunks begin
        ofs.seekp(start_of_chunks, std::ios::beg);

        uint64_t current_offset = 0;
        for (size_t i = 0; i < num_partitions; i++) {
            offsets[i] = current_offset;
            shared_ptr<IndexPartition> part = partitions_.at(part_ids[i]);

            size_t nv = static_cast<size_t>(part->num_vectors_);
            size_t csize = nv * static_cast<size_t>(part->code_size_);
            size_t isize = nv * sizeof(idx_t);

            ofs.write(reinterpret_cast<const char *>(part->codes_), csize);
            ofs.write(reinterpret_cast<const char *>(part->ids_), isize);

            current_offset += (csize + isize);
        }
        offsets[num_partitions] = current_offset;

        // Go back and write offsets array, then partition ID array
        ofs.seekp(32, std::ios::beg);
        ofs.write(reinterpret_cast<const char *>(offsets.data()),
                  offsets.size() * sizeof(uint64_t));

        for (size_t i = 0; i < num_partitions; i++) {
            uint64_t pid_64 = static_cast<uint64_t>(part_ids[i]);
            ofs.write(reinterpret_cast<const char *>(&pid_64), sizeof(pid_64));
        }

        ofs.close();
    }

    void DynamicInvertedLists::load(const string &filename,
                                    bool metadata_only,
                                    const string &s3_bucket,
                                    const string &s3_prefix,
                                    const string &s3_region,
                                    const string &s3_endpoint) {
        /**
         * Deserialization Logic:
         *  - Read header (magic, version, nlist, code_size, num_partitions)
         *  - Read offsets array (num_partitions+1)
         *  - Read partition ID array (num_partitions)
         *  - For each partition i:
         *      chunk_size = offsets[i+1] - offsets[i]
         *      num_vectors = chunk_size / (code_size + sizeof(idx_t))
         *      Seek to start_of_chunks + offsets[i]
         *      Read codes (num_vectors*code_size)
         *      Read ids   (num_vectors*sizeof(idx_t))
         *      Construct IndexPartition and store in partitions_[pid].
         *
         *  When metadata_only=true (S3 mode):
         *      s3_bucket must be non-empty.
         *      Only the header and manifests are read; partition data is downloaded from S3
         *      on demand during search via get_codes()/get_ids().
         */
        if (metadata_only) {
            if (s3_bucket.empty()) {
                throw std::runtime_error(
                    "DynamicInvertedLists::load: metadata_only requires a non-empty s3_bucket.");
            }
        }

        reset();
        s3_mode_ = false;
        s3_num_vectors_.clear();

        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + std::string(filename));
        }

        // Read header
        uint32_t file_magic = 0;
        uint32_t file_version = 0;
        ifs.read(reinterpret_cast<char *>(&file_magic), sizeof(file_magic));
        ifs.read(reinterpret_cast<char *>(&file_version), sizeof(file_version));

        if (file_magic != SerializationMagicNumber) {
            throw std::runtime_error("Invalid file format (bad magic number).");
        }
        if (file_version != SerializationVersion) {
            throw std::runtime_error("Unsupported file version: " + std::to_string(file_version));
        }

        uint64_t nlist_64, code_size_64, num_partitions;
        ifs.read(reinterpret_cast<char *>(&nlist_64), sizeof(nlist_64));
        ifs.read(reinterpret_cast<char *>(&code_size_64), sizeof(code_size_64));
        ifs.read(reinterpret_cast<char *>(&num_partitions), sizeof(num_partitions));

        nlist = static_cast<size_t>(nlist_64);
        code_size = static_cast<size_t>(code_size_64);
        d_ = code_size / sizeof(float);

        // Read offsets
        std::vector<uint64_t> offsets(num_partitions + 1);
        ifs.read(reinterpret_cast<char *>(offsets.data()),
                 offsets.size() * sizeof(uint64_t));

        // Read partition IDs
        std::vector<uint64_t> pid_array(num_partitions);
        ifs.read(reinterpret_cast<char *>(pid_array.data()),
                 pid_array.size() * sizeof(uint64_t));

        ifs.close();

        if (metadata_only) {
            // S3 mode: build manifest (pid → num_vectors) without loading vector data.
            uint64_t record_size = static_cast<uint64_t>(code_size) + sizeof(idx_t);
            size_t max_list_id = 0;
            for (uint64_t i = 0; i < num_partitions; i++) {
                size_t pid = static_cast<size_t>(pid_array[i]);
                uint64_t chunk_size = offsets[i + 1] - offsets[i];
                uint64_t nv = (record_size > 0) ? chunk_size / record_size : 0;
                s3_num_vectors_[pid] = static_cast<size_t>(nv);
                max_list_id = std::max(max_list_id, pid);
            }
            curr_list_id_ = max_list_id + 1;

#ifdef QUAKE_USE_S3
            ensure_aws_initialized();
            Aws::Client::ClientConfiguration cfg;
            cfg.region = s3_region;
            if (!s3_endpoint.empty()) {
                cfg.endpointOverride = s3_endpoint;
            }
            s3_client_ = std::make_shared<Aws::S3::S3Client>(
                cfg,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                /*useVirtualAddressing=*/false);
            s3_bucket_ = s3_bucket;
            s3_prefix_ = s3_prefix;
            s3_mode_ = true;
#else
            throw std::runtime_error(
                "Quake was built without S3 support (QUAKE_USE_S3 not set).");
#endif
            return;
        }

        // Normal full load: re-open to read partition data.
        std::ifstream ifs2(filename, std::ios::binary);
        if (!ifs2.is_open()) {
            throw std::runtime_error("Could not re-open file for reading: " + std::string(filename));
        }

        // Calculate where chunks begin
        uint64_t offset_table_bytes = (num_partitions + 1) * sizeof(uint64_t);
        uint64_t partition_ids_bytes = num_partitions * sizeof(uint64_t);
        uint64_t start_of_chunks = 32 + offset_table_bytes + partition_ids_bytes;

        // Read each partition chunk
        for (uint64_t i = 0; i < num_partitions; i++) {
            size_t pid = static_cast<size_t>(pid_array[i]);

            uint64_t chunk_start = offsets[i];
            uint64_t chunk_end = offsets[i + 1];
            uint64_t chunk_size = chunk_end - chunk_start;

            uint64_t record_size = static_cast<uint64_t>(code_size) + sizeof(idx_t);
            if (chunk_size % record_size != 0) {
                throw std::runtime_error("Partition chunk size not divisible by (code_size+sizeof(idx_t))");
            }
            uint64_t nv64 = chunk_size / record_size; // num_vectors

            ifs2.seekg(start_of_chunks + chunk_start, std::ios::beg);

            size_t csize = static_cast<size_t>(nv64) * code_size;
            size_t isize = static_cast<size_t>(nv64) * sizeof(idx_t);
            uint8_t *codes = new uint8_t[csize];
            idx_t *ids = new idx_t[nv64];

            // Read codes and ids from file into allocated buffers
            ifs2.read(reinterpret_cast<char*>(codes), csize);
            ifs2.read(reinterpret_cast<char*>(ids), isize);

            // IndexPartition part = IndexPartition(nv64, codes, ids, code_size);
            shared_ptr<IndexPartition> part = std::make_shared<IndexPartition>(nv64, codes, ids, code_size);
            partitions_[pid] = part;

            // save to free codes and ids since IndexPartition makes its own copies
            delete[] codes;
            delete[] ids;
        }

        // Update curr_list_id_
        size_t max_list_id = 0;
        for (auto &kv: partitions_) {
            max_list_id = std::max(max_list_id, kv.first);
        }
        curr_list_id_ = max_list_id + 1;

        ifs2.close();

        build_map();
    }

    Tensor DynamicInvertedLists::get_partition_ids() {
        // Return a 1D tensor of partition IDs
        Tensor result = torch::empty({(int64_t) partitions_.size()}, torch::kInt64);
        auto result_accessor = result.accessor<int64_t, 1>();
        size_t i = 0;
        for (auto &kv: partitions_) {
            result_accessor[i] = static_cast<int64_t>(kv.first);
            i++;
        }
        return result;
    }

#ifdef QUAKE_USE_NUMA
void DynamicInvertedLists::set_numa_details(int num_numa_nodes, int next_numa_node) {
    total_numa_nodes_ = num_numa_nodes;
    next_numa_node_ = next_numa_node;
}

int DynamicInvertedLists::get_numa_node(size_t list_no) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in get_numa_node");
    }
    return it->second->numa_node_;
}

void DynamicInvertedLists::set_numa_node(size_t list_no, int new_numa_node, bool interleaved) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in set_numa_node");
    }
    it->second->set_numa_node(new_numa_node);
}

std::set<size_t> DynamicInvertedLists::get_unassigned_clusters() {
    // Now we need a way to track unassigned clusters.
    // If you consider "unassigned" as numa_node_ = -1:
    std::set<size_t> result;
    for (auto &kv : partitions_) {
        if (kv.second->numa_node_ == -1) {
            result.insert(kv.first);
        }
    }
    return result;
}

int DynamicInvertedLists::get_thread(size_t list_no) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in get_thread");
    }
    return it->second->core_id_;
}

void DynamicInvertedLists::set_thread(size_t list_no, int new_thread_id) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in set_thread");
    }
    it->second->core_id_ = new_thread_id;
}

#endif
} // namespace faiss
