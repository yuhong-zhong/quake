//
// Created by Jason on 9/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names
// Provides a dynamic, NUMA-aware inverted list implementation that extends the Faiss InvertedLists interface.
// It stores codes and IDs for each partition in a map of IndexPartition objects, supporting dynamic insertions,
// updates, removals, and conversion to/from the standard faiss::ArrayInvertedLists format.

#ifndef DYNAMIC_INVERTED_LIST_H
#define DYNAMIC_INVERTED_LIST_H

#include <common.h>
#include <faiss/invlists/InvertedLists.h>
#include <index_partition.h>
#include <clustering.h>

#ifdef QUAKE_USE_S3
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#endif

namespace faiss {
    /**
     * @brief A dynamic inverted list implementation using a map of IndexPartition objects.
     *
     * DynamicInvertedLists extends the Faiss InvertedLists interface to allow compatibility with Faiss.
     * It supports dynamic operations (addition, update, removal) across multiple partitions and includes
     * NUMA-aware functionality.
     */
    class DynamicInvertedLists : public InvertedLists {
    public:

        int curr_list_id_ = 0;         ///< Next available partition ID.
        int total_numa_nodes_ = 0;     ///< Total NUMA nodes available.
        int next_numa_node_ = 0;       ///< Next NUMA node to use (for round-robin allocation).
        int d_;                        ///< Dimensionality of the vectors (derived from code_size).
        int code_size_;                ///< Size in bytes of each vector code.
        unordered_map<size_t, shared_ptr<IndexPartition>> partitions_; ///< Map of partition ID to IndexPartition.
        unordered_map<int64_t, std::pair<IndexPartition*, int64_t>> id_to_location_;
        unordered_map<size_t, bool> tombstones_;

        // S3 mode: partition data is downloaded on demand instead of loaded from disk.
        bool s3_mode_ = false;
        std::string s3_bucket_;
        std::string s3_prefix_;
#ifdef QUAKE_USE_S3
        std::shared_ptr<Aws::S3::S3Client> s3_client_;
#endif
        /// Manifest: pid → num_vectors (populated when loading in metadata_only mode).
        unordered_map<size_t, size_t> s3_num_vectors_;
        /// Temporary per-query downloaded partitions; cleared before/after each search.
        mutable unordered_map<size_t, shared_ptr<IndexPartition>> temp_s3_;
        mutable std::mutex temp_s3_mutex_;
        /// Per-query S3 timing accumulators (reset by QueryCoordinator before each search).
        mutable std::atomic<int64_t> s3_load_time_ns_{0};
        mutable std::atomic<int64_t> n_s3_downloads_{0};

        // ── Block layer (S3 mode only) ──────────────────────────────────────
        bool use_blocks_ = false;          ///< True when block mode is active.
        int block_size_ = DEFAULT_BLOCK_SIZE;
        int memtable_flush_threshold_ = DEFAULT_MEMTABLE_FLUSH_THRESHOLD;
        size_t curr_block_id_ = 0;         ///< Next available globally-unique block ID.
        MetricType block_metric_ = faiss::METRIC_L2; ///< Metric for intra-block clustering.

        /// partition_id → list of block_ids (a block may appear in multiple partitions).
        unordered_map<size_t, std::vector<size_t>> partition_blocks_;
        /// block_id → num_vectors in that block.
        unordered_map<size_t, size_t> block_num_vectors_;
        /// block_id → reference count (number of partitions referencing this block).
        unordered_map<size_t, int> block_refcount_;
        /// block_id → set of tombstoned (deleted) vector IDs within that block.
        mutable unordered_map<size_t, std::unordered_set<idx_t>> block_tombstones_;

        /// Per-partition write buffer with its own mutex.
        struct Memtable {
            std::mutex mutex;
            shared_ptr<IndexPartition> data;
        };
        mutable unordered_map<size_t, std::unique_ptr<Memtable>> memtables_;
        mutable std::mutex memtables_map_mutex_; ///< Protects memtables_ map structure (not individual entries).

        /// Per-query downloaded blocks cache; cleared before/after each search.
        mutable unordered_map<size_t, shared_ptr<IndexPartition>> temp_s3_blocks_;
        /// block_id → IndexPartition data (offline block storage before S3 upload).
        unordered_map<size_t, shared_ptr<IndexPartition>> block_data_;

        /**
         * @brief Constructor for DynamicInvertedLists.
         *
         * Initializes the object with a given number of empty partitions and sets the code size.
         *
         * @param nlist Number of partitions to initialize.
         * @param code_size Size in bytes for each code.
         */
        DynamicInvertedLists(size_t nlist, size_t code_size);

        /**
         * @brief Destructor.
         *
         * Frees memory by relying on each IndexPartition’s destructor.
         */
        ~DynamicInvertedLists() override;

         /**
         * @brief Return the total number of vectors stored across all partitions.
         *
         * @return Total count of vectors.
         */
        size_t ntotal() const;

        /**
         * @brief Return the number of vectors in the specified partition.
         *
         * @param list_no Partition (list) number.
         * @return Count of vectors in the partition.
         * @throws std::runtime_error if the partition does not exist.
         */
        size_t list_size(size_t list_no) const override;

        /**
         * @brief Get the pointer to the encoded vectors for a partition.
         *
         * @param list_no Partition number.
         * @return Pointer to codes.
         * @throws std::runtime_error if the partition does not exist.
         */
        const uint8_t* get_codes(size_t list_no) const override;

        /**
         * @brief Get the pointer to the vector IDs for a partition.
         *
         * @param list_no Partition number.
         * @return Pointer to IDs.
         * @throws std::runtime_error if the partition does not exist.
         */
        const idx_t* get_ids(size_t list_no) const override;

        void build_map();

        /**
         * @brief Release the codes pointer.
         *
         * No action is needed because memory is managed internally.
         *
         * @param list_no Partition number.
         * @param codes Unused.
         */
        void release_codes(size_t list_no, const uint8_t *codes) const override;

        /**
         * @brief Release the IDs pointer.
         *
         * No action is needed because memory is managed internally.
         *
         * @param list_no Partition number.
         * @param ids Unused.
         */
        void release_ids(size_t list_no, const idx_t *ids) const override;

        /**
         * @brief Remove an entry with the given ID from a specified partition.
         *
         * @param list_no The partition number.
         * @param id The vector ID to remove.
         * @throws std::runtime_error if the partition does not exist.
         */
        void remove_entry(size_t list_no, idx_t id);

        /**
         * @brief Remove multiple entries from a partition.
         *
         * @param list_no Partition number.
         * @param vectors_to_remove A vector of IDs to remove.
         * @throws std::runtime_error if the partition does not exist.
         */
        void remove_entries_from_partition(size_t list_no, vector<idx_t> vectors_to_remove);

        /**
         * @brief Remove specified vectors from all partitions.
         *
         * @param vectors_to_remove A set of vector IDs to remove.
         */
        void remove_vectors(std::set<idx_t> vectors_to_remove);

        /**
         * @brief Append new entries (codes and IDs) to a partition.
         *
         * @param list_no Partition number.
         * @param n_entry Number of entries to add.
         * @param ids Pointer to the vector IDs.
         * @param codes Pointer to the encoded vectors.
         * @return Number of entries added.
         * @throws std::runtime_error if the partition does not exist.
         */
        size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t *ids,
            const uint8_t *codes) override;

        /**
         * @brief Update existing entries in a partition.
         *
         * Overwrites n_entry vectors starting at the given offset.
         *
         * @param list_no Partition number.
         * @param offset Starting index for update.
         * @param n_entry Number of entries to update.
         * @param ids Pointer to new IDs.
         * @param codes Pointer to new encoded vectors.
         * @throws std::runtime_error if the partition does not exist.
         */
        void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t *ids,
            const uint8_t *codes) override;

        /**
         * @brief Batch update: move vectors from one partition to new partitions.
         *
         * Moves vectors that have changed partitions from old_vector_partition.
         *
         * @param old_vector_partition Source partition.
         * @param new_vector_partitions Array of new partition numbers for each vector.
         * @param new_vectors Pointer to new encoded vectors.
         * @param new_vector_ids Pointer to new vector IDs.
         * @param num_vectors Number of vectors to process.
         */
        void batch_update_entries(
            size_t old_vector_partition,
            int64_t* new_vector_partitions,
            uint8_t* new_vectors,
            int64_t* new_vector_ids,
            int num_vectors);

        /**
         * @brief Remove a partition entirely.
         *
         * @param list_no Partition number to remove.
         */
        void remove_list(size_t list_no);

        /**
         * @brief Add a new, empty partition.
         *
         * @param list_no The partition number to add.
         * @throws std::runtime_error if the partition already exists.
         */
        void add_list(size_t list_no);

        /**
         * @brief Check if a given ID exists in a partition.
         *
         * @param list_no Partition number.
         * @param id Vector ID to check.
         * @return True if found, false otherwise.
         */
        bool id_in_list(size_t list_no, idx_t id) const;

        /**
         * @brief Retrieve a vector by its ID.
         *
         * Copies the encoded vector into the provided buffer (interpreted as floats).
         *
         * @param id Vector ID.
         * @param vector_values Buffer to store the vector.
         * @return True if the vector was found, false otherwise.
         */
        bool get_vector_for_id(idx_t id, float* vector_values);

        /**
         * @brief No-copy retrieve vectors by their IDs.
         *
         * @param ids Vector of IDs to retrieve.
         * @return Vector of pointers to the encoded vectors.
         */
        vector<float *> get_vectors_by_id(vector<int64_t> ids);

        /**
         * @brief Generate and return a new partition ID.
         *
         * @return New unique partition ID.
         */
        size_t get_new_list_id();

        /**
         * @brief Reset the entire dynamic inverted lists.
         *
         * Clears all partitions and resets counters.
         */
        void reset() override;

        /**
         * @brief Resize the inverted lists.
         *
         * This function is a no-op in the current implementation.
         *
         * @param nlist New number of partitions.
         * @param code_size New code size.
         */
        void resize(size_t nlist, size_t code_size) override;

        /**
         * @brief Set NUMA configuration for the inverted lists.
         *
         * @param num_numa_nodes Total number of NUMA nodes.
         * @param next_numa_node Next NUMA node to be used.
         */
        void set_numa_details(int num_numa_nodes, int next_numa_node);

        /**
         * @brief Get the NUMA node for a specified partition.
         *
         * @param list_no Partition number.
         * @return NUMA node of the partition.
         * @throws std::runtime_error if the partition does not exist.
         */
        int get_numa_node(size_t list_no);

        /**
         * @brief Set the NUMA node for a specified partition.
         *
         * @param list_no Partition number.
         * @param new_numa_node Target NUMA node.
         * @param interleaved Whether to use interleaved allocation (default false).
         * @throws std::runtime_error if the partition does not exist.
         */
        void set_numa_node(size_t list_no, int new_numa_node, bool interleaved = false);

        /**
         * @brief Get the set of partition IDs that have not been assigned a NUMA node.
         *
         * @return Set of partition IDs with numa_node_ == -1.
         */
        std::set<size_t> get_unassigned_clusters();

        /**
         * @brief Get the thread ID mapped to a partition.
         *
         * @param list_no Partition number.
         * @return Thread ID.
         * @throws std::runtime_error if the partition does not exist.
         */
        int get_thread(size_t list_no);

        /**
         * @brief Set the thread ID for a partition.
         *
         * @param list_no Partition number.
         * @param new_thread_id New thread ID.
         * @throws std::runtime_error if the partition does not exist.
         */
        void set_thread(size_t list_no, int new_thread_id);

        /**
         * @brief A view of a single block's data for scanning.
         */
        struct BlockView {
            const float* codes;
            const int64_t* ids;
            int64_t num_vectors;
            const std::unordered_set<idx_t>* tombstones; ///< null if none
        };

        /**
         * @brief Get per-block views for a partition (blocks must be prefetched).
         *
         * Returns views into prefetched block data + memtable so the caller can
         * scan each block directly without concatenation. The caller must skip
         * vectors whose ID appears in the tombstone set.
         */
        std::vector<BlockView> get_block_views(size_t list_no) const;

        /**
         * @brief Configure block layer for S3 mode.
         */
        void configure_blocks(int block_size, int memtable_flush_threshold, MetricType metric);

        /**
         * @brief Flush a single partition's memtable to S3 blocks (with intra-block clustering).
         */
        void flush_memtable(size_t pid);

        /**
         * @brief Create blocks from all in-memory partitions (offline, no S3 upload).
         *
         * Used during build() to pre-create block structure before saving to disk.
         */
        void create_blocks_from_partitions();

        /**
         * @brief Download a batch of S3 blocks in parallel.
         *
         * @param block_ids Block IDs to prefetch.
         */
        void prefetch_blocks(const std::vector<size_t>& block_ids) const;

        /**
         * @brief Download a batch of S3 partitions in parallel using GetObjectAsync.
         *
         * Fetches all pids not already cached in temp_s3_ concurrently via the AWS SDK's
         * internal thread pool. S3 load time is recorded as the wall-clock time of the
         * entire batch. No-op when not in S3 mode.
         *
         * @param pids Partition IDs to prefetch.
         */
        void prefetch_partitions(const std::vector<size_t>& pids) const;

        // S3 mutation helpers (no-ops when s3_mode_ is false).
        void ensure_partition_loaded(size_t pid);
        void flush_partition(size_t pid);
        void evict_partition(size_t pid);

        /**
         * @brief Save the dynamic inverted lists to a file.
         *
         * The file format includes a header, offsets array, partition ID array,
         * and concatenated data chunks for each partition.
         *
         * @param path The file path.
         * @throws std::runtime_error on file I/O errors.
         */
        void save(const std::string &path);

        /**
         * @brief Load the dynamic inverted lists from a file.
         *
         * When metadata_only=true, only the header and partition manifest are read (no vector
         * data). S3 params enable on-demand downloading of partition data during search.
         *
         * @param path         The file path.
         * @param metadata_only If true, skip loading partition data (for S3 mode).
         * @param s3_bucket    S3 bucket name (empty = no S3).
         * @param s3_prefix    S3 key prefix where partitions are stored.
         * @param s3_region    AWS region.
         * @param s3_endpoint  Optional custom endpoint URL (e.g. for MinIO).
         * @throws std::runtime_error on file I/O errors or invalid format.
         */
        void load(const std::string &path,
                  bool metadata_only = false,
                  const std::string &s3_bucket = "",
                  const std::string &s3_prefix = "",
                  const std::string &s3_region = "us-east-1",
                  const std::string &s3_endpoint = "");

        /**
         * @brief Retrieve a tensor of partition IDs.
         *
         * @return A 1D tensor containing all partition IDs.
         */
        Tensor get_partition_ids();

    private:
        /// Download a single partition from S3 and return it as an IndexPartition.
        shared_ptr<IndexPartition> s3_fetch_partition(size_t pid) const;
        void s3_ensure_partition_loaded(size_t pid);
        void s3_upload_partition(size_t pid);
        void s3_delete_partition(size_t pid);
        void s3_evict_partition(size_t pid);
        /// Build the S3 object key for a partition.
        std::string s3_partition_key(size_t pid) const {
            return s3_prefix_ + "/partition_" + std::to_string(pid);
        }

        // ── Block layer S3 helpers ──────────────────────────────────────────
        /// Build the S3 object key for a block.
        std::string s3_block_key(size_t block_id) const {
            return s3_prefix_ + "/block_" + std::to_string(block_id);
        }
        /// Download a single block from S3.
        shared_ptr<IndexPartition> s3_fetch_block(size_t block_id) const;
        /// Upload a block to S3.
        void s3_upload_block(size_t block_id, shared_ptr<IndexPartition> part);
        /// Delete a block from S3 (decrements refcount; deletes S3 object when refcount reaches 0).
        void s3_release_block(size_t block_id);
        /// Get or create the Memtable for a partition.
        Memtable& get_or_create_memtable(size_t pid) const;
        /// Cluster vectors into blocks; returns the new block_ids.
        /// When upload=true, uploads each block to S3. When false, stores in block_data_.
        std::vector<size_t> write_vectors_as_blocks(const uint8_t* codes, const idx_t* ids,
                                                     int64_t nv, int64_t code_sz,
                                                     bool upload = true);

        template<typename IdT>
        inline void map_add(IndexPartition* p, int64_t off, IdT id) noexcept {
            if (!s3_mode_) id_to_location_[id] = {p, off};
        }
        template<typename IdT>
        inline void map_erase(IdT id) noexcept {
            if (!s3_mode_) id_to_location_.erase(id);
        }
        template<typename IdT>
        inline void map_swap(IndexPartition* p, int64_t off, IdT id) noexcept {
            if (!s3_mode_) id_to_location_[id] = {p, off};
        }
    };

    /**
     * @brief Convert a DynamicInvertedLists object to an ArrayInvertedLists.
     *
     * This conversion produces a ArrayInvertedLists object with the same data as the original.
     *
     * @param invlists Pointer to a DynamicInvertedLists object.
     * @param remap_ids Output mapping from old list number to new list number.
     * @return Pointer to the ArrayInvertedLists object.
     */
    ArrayInvertedLists *convert_to_array_invlists(DynamicInvertedLists *invlists, std::unordered_map<size_t, size_t>& remap_ids);

    /**
     * @brief Convert an ArrayInvertedLists object back to a DynamicInvertedLists object.
     *
     * @param invlists Pointer to an ArrayInvertedLists object.
     * @return Pointer to the DynamicInvertedLists object.
     */
    DynamicInvertedLists *convert_from_array_invlists(ArrayInvertedLists *invlists);
} // namespace faiss

#endif //DYNAMIC_INVERTED_LIST_H