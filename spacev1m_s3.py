"""
Benchmark for Quake index with S3 partition storage.

Phase 1: Build and save index (same as spacev1m.py, no GPU).
Phase 2: Upload partitions to S3 (one object per partition).
Phase 3: Load index in S3 mode (centroid/metadata in memory; partition data on S3).
Phase 4: Query loop — same recall_target=0.9 / APS as spacev1m.py, with extra
         per-query S3 stats (download time, number of partitions downloaded).

Configure the S3 settings below before running.
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
parser.add_argument("--s3-prefetch-initial", type=int, default=4,
                    help="Number of S3 partitions to prefetch in parallel at query start (default: 4).")
parser.add_argument("--s3-prefetch-lookahead", type=int, default=4,
                    help="Number of S3 partitions to prefetch in parallel per subsequent batch (default: 4).")
parser.add_argument("--block-size", type=int, default=0,
                    help="Enable block mode with this many vectors per block (0 = legacy partition mode).")
parser.add_argument("--memtable-flush-threshold", type=int, default=8192,
                    help="Vectors per partition memtable before flushing to blocks (default: 8192).")
args = parser.parse_args()

# ── S3 configuration ────────────────────────────────────────────────────────
S3_BUCKET   = os.environ.get("QUAKE_S3_BUCKET", "my-quake-bucket")
S3_PREFIX   = os.environ.get("QUAKE_S3_PREFIX", "spacev1m")
AWS_REGION  = os.environ.get("QUAKE_S3_REGION", "us-east-1")
S3_ENDPOINT = os.environ.get("QUAKE_S3_ENDPOINT", "")  # empty = AWS; set for MinIO
INDEX_DIR   = "quake_spacev10m.index"
# ────────────────────────────────────────────────────────────────────────────

print("Loading queries...")
fq = open('/users/yuhong/sdb/SPTAG/datasets/SPACEV1B/query.bin', 'rb')
q_count = struct.unpack('i', fq.read(4))[0]
q_dimension = struct.unpack('i', fq.read(4))[0]
queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

print("Loading truth...")
ftruth = open('/users/yuhong/sdb/quake/spacev10m_gt.bin', 'rb')
t_count = struct.unpack('i', ftruth.read(4))[0]
topk = struct.unpack('i', ftruth.read(4))[0]
truth_vids = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
truth_distances = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))

if not args.skip_build:
    print("Loading dataset...")
    fdataset = open('/users/yuhong/sdb/SPTAG/datasets/SPACEV1B/vectors.bin/vectors_merged.bin', 'rb')
    dataset_count = struct.unpack('i', fdataset.read(4))[0]
    dataset_count = min(dataset_count, 10000000)
    dataset_dimension = struct.unpack('i', fdataset.read(4))[0]
    dataset = np.frombuffer(fdataset.read(dataset_count * dataset_dimension), dtype=np.int8).reshape((dataset_count, dataset_dimension))

    vectors = torch.from_numpy(dataset.copy()).to(torch.float32)
    ids = torch.arange(dataset_count)

    # ── Phase 1: Build and save ──────────────────────────────────────────────────
    index = quake.QuakeIndex()
    build_params = quake.IndexBuildParams()
    build_params.nlist = 128
    build_params.metric = "l2"
    build_params.block_size = args.block_size
    build_params.memtable_flush_threshold = args.memtable_flush_threshold

    start_time = time.time()
    index.build(vectors, ids, build_params)
    end_time = time.time()
    print("Build time:", end_time - start_time)

    index.save(INDEX_DIR)
    print("Index saved to", INDEX_DIR)

    # ── Phase 2: Upload partitions to S3 ────────────────────────────────────────
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
print("S3 index loaded (partition data will be fetched from S3 on demand)")

# ── Phase 4: Query loop ───────────────────────────────────────────────────────
recall_target = 0.9

for top_K in [10, 30, 50, 100]:
    recall_list = []
    latency_list = []
    scanned_partitions_list = []
    s3_load_ms_list = []
    scan_ms_list = []
    n_s3_list = []

    for i in range(200):
        query = torch.from_numpy(queries[i].copy()).to(torch.float32).reshape(1, -1)
        search_params = quake.SearchParams()
        search_params.k = top_K
        search_params.recall_target = recall_target
        search_params.s3_prefetch_initial = args.s3_prefetch_initial
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

        print(f"{i + 1}/{q_count}\r", end="")
    print()

    print(f"Recall target: {recall_target}, top K: {top_K}")
    print("Recall: avg {:.2f}, p0 {:.2f}, p10 {:.2f}, p20 {:.2f}, p30 {:.2f}, p40 {:.2f}, "
          "p50 {:.2f}, p60 {:.2f}, p70 {:.2f}, p80 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
              np.mean(recall_list),
              np.percentile(recall_list, 0), np.percentile(recall_list, 10),
              np.percentile(recall_list, 20), np.percentile(recall_list, 30),
              np.percentile(recall_list, 40), np.percentile(recall_list, 50),
              np.percentile(recall_list, 60), np.percentile(recall_list, 70),
              np.percentile(recall_list, 80), np.percentile(recall_list, 90),
              np.percentile(recall_list, 99), np.percentile(recall_list, 100)))
    print("Latency: avg {:.2f} ms, p0 {:.2f} ms, p10 {:.2f} ms, p20 {:.2f} ms, p30 {:.2f} ms, "
          "p40 {:.2f} ms, p50 {:.2f} ms, p60 {:.2f} ms, p70 {:.2f} ms, p80 {:.2f} ms, p90 {:.2f} ms, p99 {:.2f} ms, p100 {:.2f} ms".format(
              np.mean(latency_list),
              np.percentile(latency_list, 0), np.percentile(latency_list, 10),
              np.percentile(latency_list, 20), np.percentile(latency_list, 30),
              np.percentile(latency_list, 40), np.percentile(latency_list, 50),
              np.percentile(latency_list, 60), np.percentile(latency_list, 70),
              np.percentile(latency_list, 80), np.percentile(latency_list, 90),
              np.percentile(latency_list, 99), np.percentile(latency_list, 100)))
    print("Scanned partitions: avg {:.2f}, p0 {:.2f}, p10 {:.2f}, p20 {:.2f}, p30 {:.2f}, p40 {:.2f}, "
          "p50 {:.2f}, p60 {:.2f}, p70 {:.2f}, p80 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
              np.mean(scanned_partitions_list),
              np.percentile(scanned_partitions_list, 0), np.percentile(scanned_partitions_list, 10),
              np.percentile(scanned_partitions_list, 20), np.percentile(scanned_partitions_list, 30),
              np.percentile(scanned_partitions_list, 40), np.percentile(scanned_partitions_list, 50),
              np.percentile(scanned_partitions_list, 60), np.percentile(scanned_partitions_list, 70),
              np.percentile(scanned_partitions_list, 80), np.percentile(scanned_partitions_list, 90),
              np.percentile(scanned_partitions_list, 99), np.percentile(scanned_partitions_list, 100)))
    print("S3 load time (ms): avg {:.2f}, p0 {:.2f}, p10 {:.2f}, p20 {:.2f}, p30 {:.2f}, p40 {:.2f}, "
          "p50 {:.2f}, p60 {:.2f}, p70 {:.2f}, p80 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
              np.mean(s3_load_ms_list),
              np.percentile(s3_load_ms_list, 0), np.percentile(s3_load_ms_list, 10),
              np.percentile(s3_load_ms_list, 20), np.percentile(s3_load_ms_list, 30),
              np.percentile(s3_load_ms_list, 40), np.percentile(s3_load_ms_list, 50),
              np.percentile(s3_load_ms_list, 60), np.percentile(s3_load_ms_list, 70),
              np.percentile(s3_load_ms_list, 80), np.percentile(s3_load_ms_list, 90),
              np.percentile(s3_load_ms_list, 99), np.percentile(s3_load_ms_list, 100)))
    print("Scan time (ms): avg {:.2f}, p0 {:.2f}, p10 {:.2f}, p20 {:.2f}, p30 {:.2f}, p40 {:.2f}, "
          "p50 {:.2f}, p60 {:.2f}, p70 {:.2f}, p80 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
              np.mean(scan_ms_list),
              np.percentile(scan_ms_list, 0), np.percentile(scan_ms_list, 10),
              np.percentile(scan_ms_list, 20), np.percentile(scan_ms_list, 30),
              np.percentile(scan_ms_list, 40), np.percentile(scan_ms_list, 50),
              np.percentile(scan_ms_list, 60), np.percentile(scan_ms_list, 70),
              np.percentile(scan_ms_list, 80), np.percentile(scan_ms_list, 90),
              np.percentile(scan_ms_list, 99), np.percentile(scan_ms_list, 100)))
    print("S3 downloads per query: avg {:.2f}, p0 {:.2f}, p10 {:.2f}, p20 {:.2f}, p30 {:.2f}, p40 {:.2f}, "
          "p50 {:.2f}, p60 {:.2f}, p70 {:.2f}, p80 {:.2f}, p90 {:.2f}, p99 {:.2f}, p100 {:.2f}".format(
              np.mean(n_s3_list),
              np.percentile(n_s3_list, 0), np.percentile(n_s3_list, 10),
              np.percentile(n_s3_list, 20), np.percentile(n_s3_list, 30),
              np.percentile(n_s3_list, 40), np.percentile(n_s3_list, 50),
              np.percentile(n_s3_list, 60), np.percentile(n_s3_list, 70),
              np.percentile(n_s3_list, 80), np.percentile(n_s3_list, 90),
              np.percentile(n_s3_list, 99), np.percentile(n_s3_list, 100)))
