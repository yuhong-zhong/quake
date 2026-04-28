"""
Benchmark for Quake index with S3 partition storage — add vectors and query.

Phase 1: Build and save index (same as spacev1m.py, no GPU).
Phase 2: Upload partitions to S3 (one object per partition).
Phase 3: Load index in S3 mode.
Phase 4: Add N new vectors (ids 10_000_000 .. 10_000_000+N-1).
Phase 5: Maintenance (split / merge partitions).
Phase 6: Query loop — same recall_target=0.9 / APS as spacev1m_s3.py, with S3 stats.

Configure the S3 settings below (or via environment variables) before running.
"""

import quake
from quake.utils import compute_recall
from s3_utils import upload_index_to_s3

import argparse
import struct
import numpy as np
import os
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--skip-build", action="store_true",
                    help="Skip index building and S3 upload; load the existing index directly.")
parser.add_argument("--add-n", type=int, default=100_000,
                    help="Number of new vectors to add (default: 100000).")
parser.add_argument("--s3-prefetch-initial", type=int, default=4,
                    help="Number of S3 partitions to prefetch at query start (default: 4).")
parser.add_argument("--s3-prefetch-lookahead", type=int, default=4,
                    help="Number of S3 partitions to prefetch per batch (default: 4).")
parser.add_argument("--block-size", type=int, default=0,
                    help="Enable block mode with this many vectors per block (0 = legacy partition mode).")
parser.add_argument("--memtable-flush-threshold", type=int, default=8192,
                    help="Vectors per partition memtable before flushing to blocks (default: 8192).")
args = parser.parse_args()

# ── S3 configuration ────────────────────────────────────────────────────────
S3_BUCKET   = os.environ.get("QUAKE_S3_BUCKET",  "my-quake-bucket")
S3_PREFIX   = os.environ.get("QUAKE_S3_PREFIX",  "spacev1m_update")
AWS_REGION  = os.environ.get("QUAKE_S3_REGION",  "us-east-1")
S3_ENDPOINT = os.environ.get("QUAKE_S3_ENDPOINT", "")   # empty = AWS; set for MinIO
INDEX_DIR   = "quake_spacev10m_update.index"
# ────────────────────────────────────────────────────────────────────────────

ADD_N        = args.add_n
ADD_ID_START = 10_000_000   # ids for newly added vectors (beyond the base 10M)

print("Loading queries...")
fq = open('/users/yuhong/nvme1n1/SPTAG/datasets/SPACEV1B/query.bin', 'rb')
q_count     = struct.unpack('i', fq.read(4))[0]
q_dimension = struct.unpack('i', fq.read(4))[0]
queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

print("Loading truth...")
ftruth = open('/users/yuhong/nvme1n1/quake/spacev10m_gt.bin', 'rb')
t_count = struct.unpack('i', ftruth.read(4))[0]
topk    = struct.unpack('i', ftruth.read(4))[0]
truth_vids      = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
truth_distances = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))

if not args.skip_build:
    print("Loading dataset...")
    fdataset = open('/users/yuhong/nvme1n1/SPTAG/datasets/SPACEV1B/vectors.bin/vectors_merged.bin', 'rb')
    dataset_count     = struct.unpack('i', fdataset.read(4))[0]
    dataset_count     = min(dataset_count, ADD_ID_START + ADD_N)
    dataset_dimension = struct.unpack('i', fdataset.read(4))[0]
    dataset = np.frombuffer(fdataset.read(dataset_count * dataset_dimension),
                            dtype=np.int8).reshape((dataset_count, dataset_dimension))

    # Base dataset: first 10M vectors (ids 0..9_999_999)
    base_count = 10_000_000
    vectors = torch.from_numpy(dataset[:base_count].copy()).to(torch.float32)
    ids     = torch.arange(base_count)

    # ── Phase 1: Build and save ──────────────────────────────────────────────
    index = quake.QuakeIndex()
    build_params = quake.IndexBuildParams()
    build_params.nlist  = 1024
    build_params.metric = "l2"
    build_params.block_size = args.block_size
    build_params.memtable_flush_threshold = args.memtable_flush_threshold

    start_time = time.time()
    index.build(vectors, ids, build_params)
    print("Build time: {:.1f} s".format(time.time() - start_time))

    index.save(INDEX_DIR)
    print("Index saved to", INDEX_DIR)

    # ── Phase 2: Upload partitions to S3 ────────────────────────────────────
    upload_index_to_s3(INDEX_DIR, S3_BUCKET, S3_PREFIX,
                       region=AWS_REGION,
                       endpoint_url=S3_ENDPOINT if S3_ENDPOINT else None,
                       block_size=args.block_size)
else:
    print("Skipping index build and S3 upload.")

# ── Phase 3: Load in S3 mode ─────────────────────────────────────────────────
s3_index = quake.QuakeIndex()
s3_index.load(INDEX_DIR,
              s3_bucket=S3_BUCKET,
              s3_prefix=S3_PREFIX,
              s3_region=AWS_REGION,
              s3_endpoint=S3_ENDPOINT,
              block_size=args.block_size,
              memtable_flush_threshold=args.memtable_flush_threshold)
print("S3 index loaded (partition data fetched from S3 on demand)")
print("ntotal after load:", s3_index.ntotal())

# ── Phase 4: Add 100K vectors ────────────────────────────────────────────────
print(f"\nAdding {ADD_N} vectors (ids {ADD_ID_START}..{ADD_ID_START + ADD_N - 1})...")
fdataset2 = open('/users/yuhong/nvme1n1/SPTAG/datasets/SPACEV1B/vectors.bin/vectors_merged.bin', 'rb')
_ = fdataset2.read(8)  # skip count and dimension headers
dim = q_dimension
fdataset2.seek(8 + ADD_ID_START * dim)
add_data = np.frombuffer(fdataset2.read(ADD_N * dim), dtype=np.int8).reshape((ADD_N, dim))
vectors_to_add = torch.from_numpy(add_data.copy()).to(torch.float32)
ids_to_add     = torch.arange(ADD_ID_START, ADD_ID_START + ADD_N, dtype=torch.int64)

t0 = time.perf_counter()
s3_index.add(vectors_to_add, ids_to_add)
add_ms = (time.perf_counter() - t0) * 1e3
print(f"Add time: {add_ms:.1f} ms")
print("ntotal after add:", s3_index.ntotal())

# ── Phase 5: Maintenance ──────────────────────────────────────────────────────
print(f"\nRunning maintenance...")
nlist_before = s3_index.nlist()
t0 = time.perf_counter()
s3_index.maintenance()
maint_ms = (time.perf_counter() - t0) * 1e3
nlist_after = s3_index.nlist()
print(f"Maintenance time: {maint_ms:.1f} ms  |  nlist: {nlist_before} → {nlist_after}")

# Persist updated manifest to disk so the index can be reloaded later.
s3_index.save(INDEX_DIR)
print("Updated manifest saved to", INDEX_DIR)

# ── Phase 6: Query loop ───────────────────────────────────────────────────────
recall_target = 0.9
print(f"\nSearching (recall_target={recall_target})...")

for top_K in [10, 30, 50, 100]:
    recall_list             = []
    latency_list            = []
    scanned_partitions_list = []
    s3_load_ms_list         = []
    scan_ms_list            = []
    n_s3_list               = []

    for i in range(1000):
        query = torch.from_numpy(queries[i].copy()).to(torch.float32).reshape(1, -1)
        search_params = quake.SearchParams()
        search_params.k                   = top_K
        search_params.recall_target       = recall_target
        search_params.s3_prefetch_initial  = args.s3_prefetch_initial
        search_params.s3_prefetch_lookahead = args.s3_prefetch_lookahead

        t0 = time.perf_counter()
        result = s3_index.search(query, search_params)
        total_ms = (time.perf_counter() - t0) * 1e3

        latency_list.append(total_ms)
        recall_list.append(compute_recall(result.ids, truth_vids[i].reshape(1, -1), top_K))
        scanned_partitions_list.append(result.timing_info.partitions_scanned)
        s3_load_ms_list.append(result.timing_info.s3_load_time_ns / 1e6)
        scan_ms_list.append(result.timing_info.scan_time_ns / 1e6)
        n_s3_list.append(result.timing_info.n_s3_downloads)

        print(f"{i + 1}/1000\r", end="")
    print()

    print(f"Recall target: {recall_target}, top K: {top_K}")
    print("Recall: avg {:.2f}, p0 {:.2f}, p50 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
        np.mean(recall_list),
        np.percentile(recall_list, 0),  np.percentile(recall_list, 50),
        np.percentile(recall_list, 90), np.percentile(recall_list, 99),
        np.percentile(recall_list, 100)))
    print("Latency (ms): avg {:.2f}, p50 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
        np.mean(latency_list),
        np.percentile(latency_list, 50), np.percentile(latency_list, 90),
        np.percentile(latency_list, 99), np.percentile(latency_list, 100)))
    print("Scanned partitions: avg {:.1f}, p50 {:.1f}, p90 {:.1f}, p99 {:.1f}".format(
        np.mean(scanned_partitions_list),
        np.percentile(scanned_partitions_list, 50),
        np.percentile(scanned_partitions_list, 90),
        np.percentile(scanned_partitions_list, 99)))
    print("S3 load (ms): avg {:.2f}, p50 {:.2f}, p90 {:.2f}, p99 {:.2f}".format(
        np.mean(s3_load_ms_list),
        np.percentile(s3_load_ms_list, 50),
        np.percentile(s3_load_ms_list, 90),
        np.percentile(s3_load_ms_list, 99)))
    print("S3 downloads/query: avg {:.1f}, p50 {:.1f}, p90 {:.1f}, p99 {:.1f}".format(
        np.mean(n_s3_list),
        np.percentile(n_s3_list, 50),
        np.percentile(n_s3_list, 90),
        np.percentile(n_s3_list, 99)))
