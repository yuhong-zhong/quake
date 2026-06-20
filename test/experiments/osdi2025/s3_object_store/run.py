#!/usr/bin/env python3
"""
S3 object-store cost simulation for Quake inserts.
Measures the number of S3 API calls when inserting 100M vectors into a
Quake index in batches, assuming each partition = one S3 object.
"""
import sys
from pathlib import Path
import argparse
import json
import torch
import hashlib
import math
import os
import shutil
import struct
from datetime import datetime
import numpy as np
import time
import pandas as pd

try:
    import quake
except ModuleNotFoundError:
    # Fallback to locally built package path (e.g., build/lib.linux-*/quake).
    repo_root = Path(__file__).resolve().parents[4]
    build_libs = sorted(repo_root.glob("build/lib.*"))
    for build_lib in build_libs:
        if (build_lib / "quake").exists():
            sys.path.insert(0, str(build_lib))
            break
    import quake

# Config
DATASET_PATH = "/users/charlesx/SPTAG/datasets/SPACEV1B/vectors.bin/vectors_merged.bin"
MAX_VECTORS = 10_000_000
NLIST = 1024
METRIC = "l2"
BATCH_SIZE = 512 * 1024 * 1024   # 512 MiB worth of vectors per batch
PART_SIZE = 8 * 1024 * 1024      # 8 MiB S3 multi-part threshold
MAX_PARTITION_BYTES = 8 * 1024 * 1024 
MIN_PARTITION_BYTES = 1 * 1024 * 1024
OUTPUT_DIR = Path("test/experiments/osdi2025/s3_object_store/results/spacev1b_10m")
TEMP_DIR = OUTPUT_DIR / "_tmp_index"
TRACE_PHASES = os.environ.get("TRACE_PHASES", "0") == "1"
KEEP_FINAL_INDEX = os.environ.get("KEEP_FINAL_INDEX", "0") == "1"
HEADER_FMT = "<IIQQQ"
HEADER_BYTES = struct.calcsize(HEADER_FMT)
PUT_PRICE_PER_1K = 0.005     # per 1,000 PUT/POST/LIST requests
GET_PRICE_PER_1K = 0.0004    # per 1,000 GET/SELECT requests
STORAGE_PER_GB_MONTH = 0.023 # per GB per month
TRANSFER_PER_GB = 0.09       # data transfer out per GB (first 10 TB)


def s3_upload_calls(size_bytes):
    """
    Billable upload requests for an object.
    Uses simple thresholded multipart modeling:
      - < PART_SIZE: single PUT
      - >= PART_SIZE: CreateMultipart + UploadPart*N + CompleteMultipart
    """
    if size_bytes <= 0:
        return 0
    if size_bytes < PART_SIZE:
        return 1
    return int(math.ceil(size_bytes / PART_SIZE)) + 2


def s3_download_calls(size_bytes):
    """Billable GET requests for an object (single GET or ranged GETs by part)."""
    if size_bytes <= 0:
        return 0
    if size_bytes < PART_SIZE:
        return 1
    return int(math.ceil(size_bytes / PART_SIZE))


def hash_partitions(partitions_path):
    """
    For each partition, create (partition_id, size_bytes, sha256_hex) 
    for identifying changes between index snapshots.
    """
    with open(partitions_path, "rb") as f:
        raw = f.read(HEADER_BYTES)
        _, _, _, _, num_parts = struct.unpack(HEADER_FMT, raw)

        offsets = np.fromfile(f, dtype=np.uint64, count=num_parts + 1)
        part_ids = np.fromfile(f, dtype=np.uint64, count=num_parts)
        data_start = f.tell()

        for i in range(num_parts):
            pid = int(part_ids[i])
            chunk_start = int(offsets[i])
            chunk_end = int(offsets[i + 1])
            chunk_size = chunk_end - chunk_start

            f.seek(data_start + chunk_start)
            hasher = hashlib.sha256()
            remaining = chunk_size
            while remaining > 0:
                data = f.read(min(4 * 1024 * 1024, remaining))
                if not data:
                    raise ValueError(f"Unexpected EOF reading partition {pid}")
                hasher.update(data)
                remaining -= len(data)

            yield pid, chunk_size, hasher.hexdigest()


def read_partition_size_rows(partitions_path: Path, dataset_dimension: int):
    """Read partition ids/sizes from a saved partitions file without hashing payloads."""
    rows = []
    with open(partitions_path, "rb") as f:
        raw = f.read(HEADER_BYTES)
        _, _, _, _, num_parts = struct.unpack(HEADER_FMT, raw)
        offsets = np.fromfile(f, dtype=np.uint64, count=num_parts + 1)
        part_ids = np.fromfile(f, dtype=np.uint64, count=num_parts)

        bytes_per_vector_quake = (dataset_dimension * 4) + 8  # float32 vector + int64 id
        for i in range(num_parts):
            pid = int(part_ids[i])
            size_bytes = int(offsets[i + 1] - offsets[i])
            rows.append({
                "partition_id": pid,
                "size_bytes": size_bytes,
                "size_mib": round(size_bytes / (1024 ** 2), 6),
                "num_vectors": int(size_bytes // bytes_per_vector_quake),
            })
    rows.sort(key=lambda r: r["partition_id"])
    return rows


def build_partition_state_event(
    event_idx: int,
    batch_idx: int,
    op_type: str,
    stage: str,
    partition_rows: list[dict],
):
    total_size_bytes = int(sum(r["size_bytes"] for r in partition_rows))
    return {
        "event_idx": int(event_idx),
        "batch_idx": int(batch_idx),
        "op_type": op_type,
        "stage": stage,
        "partition_count": int(len(partition_rows)),
        "total_size_bytes": total_size_bytes,
        "total_size_mib": round(total_size_bytes / (1024 ** 2), 6),
        "partitions": partition_rows,
    }


def write_grouped_partition_json(
    partition_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    grouped_json_path: Path,
    partition_state_timeline: list[dict] | None = None,
):
    """Write grouped partition history JSON in requested stats/results schema."""
    part_df = partition_df.copy()

    # Map batch -> operation status ("build"/"insert")
    op_type_by_batch = {}
    if "batch_idx" in s3_df.columns and "op_type" in s3_df.columns:
        for _, r in s3_df.iterrows():
            op_type_by_batch[int(r["batch_idx"])] = str(r["op_type"])

    max_batch = int(part_df["batch_idx"].max()) if len(part_df) else -1
    last_batch_df = part_df[part_df["batch_idx"] == max_batch] if max_batch >= 0 else part_df

    total_batches = int(max_batch + 1) if max_batch >= 0 else 0
    total_partitions_final = int(last_batch_df["partition_id"].nunique()) if len(last_batch_df) else 0
    num_partitions_deleted = int(s3_df["partitions_deleted"].sum()) if "partitions_deleted" in s3_df.columns else 0
    num_partitions_changed = int(s3_df["partitions_changed"].sum()) if "partitions_changed" in s3_df.columns else 0
    max_size_mib = float(last_batch_df["size_mib"].max()) if len(last_batch_df) else 0.0
    min_size_mib = float(last_batch_df["size_mib"].min()) if len(last_batch_df) else 0.0

    payload = {
        "stats": {
            "total_batches": total_batches,
            "total_partitions": total_partitions_final,
            "num_partitions_deleted": num_partitions_deleted,
            "num_partitions_changed": num_partitions_changed,
            "max_size_per_partition": max_size_mib,
            "min_size_per_partition": min_size_mib,
        },
        "results": {
            "by_batch": s3_df.sort_values("batch_idx").to_dict(orient="records"),
            "partition_state_timeline": partition_state_timeline or [],
        },
    }
    with open(grouped_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def compute_s3_costs(df: pd.DataFrame, final_index_bytes: int):
    total_writes = int(df["s3_write_calls"].sum())
    total_reads = int(df["s3_read_calls"].sum())
    total_deletes = int(df["s3_delete_calls"].sum()) if "s3_delete_calls" in df.columns else 0
    total_bytes_written = float(df["bytes_written"].sum())
    total_bytes_read = float(df["bytes_read"].sum())
    put_cost = (total_writes / 1000) * PUT_PRICE_PER_1K
    get_cost = (total_reads / 1000) * GET_PRICE_PER_1K
    transfer_out_gb = total_bytes_read / 1e9
    transfer_cost = transfer_out_gb * TRANSFER_PER_GB
    storage_gb = final_index_bytes / 1e9 if len(df) > 0 else 0
    return {
        "total_writes": total_writes,
        "total_reads": total_reads,
        "total_deletes": total_deletes,
        "total_bytes_written": total_bytes_written,
        "total_bytes_read": total_bytes_read,
        "put_cost": put_cost,
        "get_cost": get_cost,
        "transfer_cost": transfer_cost,
        "storage_cost_month": storage_gb * STORAGE_PER_GB_MONTH,
        "total_excl_storage": put_cost + get_cost + transfer_cost,
    }


def plot_results(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR, show_plot: bool = True):
    try:
        import ctypes
        conda_libstdcpp = Path(sys.prefix) / "lib" / "libstdc++.so.6"
        if conda_libstdcpp.exists():
            ctypes.CDLL(str(conda_libstdcpp), mode=ctypes.RTLD_GLOBAL)

        import matplotlib.pyplot as plt

        df = df.copy()
        df["put_cost_usd"] = (df["s3_write_calls"] / 1000.0) * PUT_PRICE_PER_1K
        df["get_cost_usd"] = (df["s3_read_calls"] / 1000.0) * GET_PRICE_PER_1K
        df["transfer_cost_usd"] = (df["bytes_read"] / 1e9) * TRANSFER_PER_GB
        df["batch_total_cost_usd"] = df["put_cost_usd"] + df["get_cost_usd"] + df["transfer_cost_usd"]
        df["cum_cost_usd"] = df["batch_total_cost_usd"].cumsum()

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ax_lat, ax_parts = axs[0]
        ax_req, ax_cost = axs[1]

        ax_lat.plot(df["batch_idx"], df["op_latency_ms"], label="Op latency", marker="o")
        ax_lat.plot(df["batch_idx"], df["maintenance_latency_ms"], label="Maintenance latency", marker="o")
        ax_lat.set_title("Latency per Batch")
        ax_lat.set_xlabel("# Batch")
        ax_lat.set_ylabel("Time (ms)")
        ax_lat.set_yscale("log")
        ax_lat.legend()

        ax_parts.plot(df["batch_idx"], df["partitions_total"], label="Total", marker="o")
        ax_parts.plot(df["batch_idx"], df["partitions_new"], label="New", marker="o")
        ax_parts.plot(df["batch_idx"], df["partitions_changed"], label="Changed", marker="o")
        ax_parts.plot(df["batch_idx"], df["partitions_deleted"], label="Deleted", marker="o")
        ax_parts.set_title("Partitions per Batch")
        ax_parts.set_xlabel("# Batch")
        ax_parts.set_ylabel("Partitions")
        ax_parts.legend()

        ax_req.plot(df["batch_idx"], df["s3_write_calls"], label="Write req", marker="o")
        ax_req.plot(df["batch_idx"], df["s3_read_calls"], label="Read req", marker="o")
        ax_req.plot(df["batch_idx"], df["s3_delete_calls"], label="Delete req", marker="o")
        ax_req.set_title("S3 Requests per Batch")
        ax_req.set_xlabel("# Batch")
        ax_req.set_ylabel("Requests")
        ax_req.legend()

        ax_cost.plot(df["batch_idx"], df["cum_cost_usd"], marker="o")
        ax_cost.set_title("Cumulative S3 Cost (USD)")
        ax_cost.set_xlabel("# Batch")
        ax_cost.set_ylabel("USD")

        x_vals = df["batch_idx"].astype(int).to_numpy()
        if len(x_vals) > 0:
            x_min = int(x_vals.min()) - 0.5
            x_max = int(x_vals.max()) + 0.5
            for ax in (ax_lat, ax_parts, ax_req, ax_cost):
                ax.set_xlim(x_min, x_max)
                ax.set_xticks(x_vals)

        plt.tight_layout()
        plot_path = output_dir / "s3_experiment_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    except Exception as plot_exc:
        print(f"Skipping plot because matplotlib failed to load: {plot_exc}")


def load_batch_df_from_json(bench_json: Path) -> pd.DataFrame:
    with open(bench_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", {}).get("by_batch", [])
    if not rows:
        raise ValueError(f"No results.by_batch found in {bench_json}")
    return pd.DataFrame(rows)


def benchmark(run_plot: bool = True, show_plot: bool = True):
    # Load dataset
    print("Loading dataset...")
    with open(DATASET_PATH, "rb") as fdataset:
        total_vectors = struct.unpack("i", fdataset.read(4))[0]
        dataset_dimension = struct.unpack("i", fdataset.read(4))[0]
    dataset_count = min(total_vectors, MAX_VECTORS)
    print(f"Dataset: n={dataset_count}, d={dataset_dimension}")

    # Use memmap to avoid loading the full dataset into RAM for 100M runs.
    dataset = np.memmap(
        DATASET_PATH,
        dtype=np.int8,
        mode="r",
        offset=8,
        shape=(dataset_count, dataset_dimension),
    )

    # Batch sizing uses float32 payload (actual tensors passed into Quake).
    bytes_per_vector_batch = dataset_dimension * 4
    # Maintenance thresholds use approximate Quake storage bytes per vector.
    # Partition payload stores float32 vector values and vector ids.
    bytes_per_vector_quake = (dataset_dimension * 4) + 8

    batch_vectors = max(1, BATCH_SIZE // bytes_per_vector_batch)
    max_partition_vectors = max(1, MAX_PARTITION_BYTES // bytes_per_vector_quake)
    min_partition_vectors = max(1, MIN_PARTITION_BYTES // bytes_per_vector_quake)
    n_batches = math.ceil(dataset_count / batch_vectors)
    print(f"Batch size: {batch_vectors} vectors ({BATCH_SIZE / 1024 / 1024:.0f} MiB), {n_batches} batches total")
    print(f"Naive split policy: min_partition_size={min_partition_vectors} vectors "
          f"({MIN_PARTITION_BYTES / 1024 / 1024:.0f} MiB), "
          f"max_partition_size={max_partition_vectors} vectors "
          f"({MAX_PARTITION_BYTES / 1024 / 1024:.0f} MiB)")
    print(f"Byte model: batch_vector_bytes={bytes_per_vector_batch}, "
          f"quake_partition_vector_bytes={bytes_per_vector_quake}")

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_rows = []
    partition_rows = []
    partition_state_timeline = []
    event_idx = 0

    # Build & insert
    index = quake.QuakeIndex()
    prev_hashes = {}
    final_index_bytes = 0

    for batch_idx in range(n_batches):
        if TRACE_PHASES:
            print(f"[TRACE] batch={batch_idx} phase=batch_start")
        start = batch_idx * batch_vectors
        end = min(start + batch_vectors, dataset_count)
        n_vecs = end - start

        t_data = time.perf_counter()
        vectors = torch.from_numpy(dataset[start:end].copy()).to(torch.float32)
        ids = torch.arange(start, end, dtype=torch.int64)
        data_to_torch_ms = (time.perf_counter() - t_data) * 1000.0

        t0 = time.perf_counter()
        op_engine_ms = 0.0
        build_train_ms = 0.0
        build_assign_ms = 0.0
        add_find_partition_ms = 0.0
        if batch_idx == 0:
            build_params = quake.IndexBuildParams()
            build_params.nlist = NLIST
            build_params.metric = METRIC
            build_info = index.build(vectors, ids, build_params)
            op_engine_ms = build_info.total_time_us / 1000.0
            build_train_ms = build_info.train_time_us / 1000.0
            build_assign_ms = build_info.assign_time_us / 1000.0

            maintenance_params = quake.MaintenancePolicyParams()
            maintenance_params.min_partition_size = min_partition_vectors
            maintenance_params.max_partition_size = max_partition_vectors
            index.initialize_maintenance_policy(maintenance_params)
            op_type = "build"
        else:
            add_info = index.add(vectors, ids)
            op_engine_ms = add_info.modify_time_us / 1000.0
            add_find_partition_ms = add_info.find_partition_time_us / 1000.0
            op_type = "insert"
        op_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Snapshot immediately after build/add, before maintenance.
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(parents=True)
        index.save(str(TEMP_DIR))
        pre_maint_rows = read_partition_size_rows(TEMP_DIR / "partitions", dataset_dimension)
        partition_state_timeline.append(
            build_partition_state_event(
                event_idx=event_idx,
                batch_idx=batch_idx,
                op_type=op_type,
                stage=f"after_{op_type}",
                partition_rows=pre_maint_rows,
            )
        )
        event_idx += 1

        # Apply maintenance each batch so size-based split policy is enforced.
        maint_t0 = time.perf_counter()
        maint_info = index.maintenance()
        maintenance_wall_ms = (time.perf_counter() - maint_t0) * 1000.0
        maintenance_engine_ms = maint_info.total_time_us / 1000.0
        update_plus_maintenance_ms = op_latency_ms + maintenance_wall_ms

        # Save index to temp dir
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(parents=True)
        t_save = time.perf_counter()
        index.save(str(TEMP_DIR))
        save_time_ms = (time.perf_counter() - t_save) * 1000.0
        post_maint_rows = read_partition_size_rows(TEMP_DIR / "partitions", dataset_dimension)
        partition_state_timeline.append(
            build_partition_state_event(
                event_idx=event_idx,
                batch_idx=batch_idx,
                op_type=op_type,
                stage="after_maintenance",
                partition_rows=post_maint_rows,
            )
        )
        event_idx += 1

        # Hash partitions and compute S3 costs
        read_calls = 0
        write_calls = 0
        bytes_read = 0
        bytes_written = 0
        changed = 0
        created = 0
        deleted = 0
        total_parts = 0
        delete_calls = 0
        current_index_bytes = 0
        partition_size_bytes = []
        partitions_over_8mib = 0

        new_hashes = {}
        t_hash = time.perf_counter()
        for pid, size_bytes, digest in hash_partitions(TEMP_DIR / "partitions"):
            total_parts += 1
            current_index_bytes += size_bytes
            partition_size_bytes.append(size_bytes)
            new_hashes[pid] = digest
            over_8mib = int(size_bytes > MAX_PARTITION_BYTES)
            partitions_over_8mib += over_8mib
            partition_rows.append({
                "batch_idx": batch_idx,
                "partition_id": pid,
                "size_bytes": size_bytes,
                "size_mib": round(size_bytes / (1024 ** 2), 6),
                "estimated_vectors_int8": int(size_bytes // (dataset_dimension * 1)),
                "estimated_vectors_float32": int(size_bytes // (dataset_dimension * 4)),
                "estimated_vectors_quake": int(size_bytes // bytes_per_vector_quake),
                "over_8mib": over_8mib,
            })

            if pid not in prev_hashes:
                # New partition: only upload
                created += 1
                write_calls += s3_upload_calls(size_bytes)
                bytes_written += size_bytes
            elif prev_hashes[pid] != digest:
                # Changed partition: download old + upload new
                changed += 1
                read_calls += s3_download_calls(size_bytes)
                write_calls += s3_upload_calls(size_bytes)
                bytes_read += size_bytes
                bytes_written += size_bytes
        hash_time_ms = (time.perf_counter() - t_hash) * 1000.0

        # Deleted partitions (removed by maintenance); no user delete op required.
        for pid in prev_hashes:
            if pid not in new_hashes:
                deleted += 1
                delete_calls += 1

        prev_hashes = new_hashes
        final_index_bytes = current_index_bytes
        if partition_size_bytes:
            partition_max_bytes = int(np.max(partition_size_bytes))
            partition_avg_bytes = float(np.mean(partition_size_bytes))
            partition_p95_bytes = float(np.percentile(partition_size_bytes, 95))
        else:
            partition_max_bytes = 0
            partition_avg_bytes = 0.0
            partition_p95_bytes = 0.0

        # Clean up temp dir unless user wants to inspect final index.
        keep_this_snapshot = KEEP_FINAL_INDEX and (batch_idx == n_batches - 1)
        if not keep_this_snapshot:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)

        row = {
            "batch_idx": batch_idx,
            "op_type": op_type,
            "vectors_in_batch": n_vecs,
            "op_latency_ms": round(op_latency_ms, 2),
            "maintenance_latency_ms": round(maintenance_wall_ms, 2),
            "index_ntotal": end,
            "partitions_total": total_parts,
            "partitions_new": created,
            "partitions_changed": changed,
            "partitions_deleted": deleted,
            "partition_max_bytes": partition_max_bytes,
            "partition_avg_bytes": round(partition_avg_bytes, 2),
            "partition_p95_bytes": round(partition_p95_bytes, 2),
            "s3_read_calls": read_calls,
            "s3_write_calls": write_calls,
            "s3_delete_calls": delete_calls,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
        }
        results_rows.append(row)

        print(f"Batch {batch_idx}: {op_type} vectors={n_vecs} "
              f"op={op_latency_ms / 1000:.1f}s maint={maintenance_wall_ms / 1000:.1f}s "
              f"total={update_plus_maintenance_ms / 1000:.1f}s save={save_time_ms / 1000:.1f}s "
              f"hash={hash_time_ms / 1000:.1f}s "
              f"parts={total_parts} new={created} changed={changed} deleted={deleted} "
              f"max_part={partition_max_bytes / (1024 ** 2):.2f}MiB over8mib={partitions_over_8mib} "
              f"s3_reads={read_calls} s3_writes={write_calls} s3_deletes={delete_calls}")
        if TRACE_PHASES:
            print(f"[TRACE] batch={batch_idx} phase=batch_end op_ms={op_latency_ms:.2f} maint_ms={maintenance_wall_ms:.2f}")

    print("\nDone. Building in-memory results tables...")
    s3_df = pd.DataFrame(results_rows)
    partition_df = pd.DataFrame(partition_rows)

    timestamp = datetime.now().strftime("%d_%m_%H%M%S")
    bench_json = OUTPUT_DIR / f"s3_quake_bench_{timestamp}.json"
    payload = write_grouped_partition_json(
        partition_df,
        s3_df,
        bench_json,
        partition_state_timeline=partition_state_timeline,
    )
    print(f"Benchmark JSON saved to {bench_json}")
    if KEEP_FINAL_INDEX:
        print(f"Final index kept at {TEMP_DIR} (set KEEP_FINAL_INDEX=0 to auto-delete)")

    # S3 Pricing
    df = s3_df
    costs = compute_s3_costs(df, final_index_bytes)

    print(f"\n{'=' * 60}")
    print(f"S3 Cost Estimate (AWS us-east-1 Standard pricing)")
    print(f"{'=' * 60}")
    print(f"Total write requests:        {costs['total_writes']:>12,}")
    print(f"Total GET calls:             {costs['total_reads']:>12,}")
    print(f"Total DELETE calls (free):   {costs['total_deletes']:>12,}")
    print(f"Total data written:          {costs['total_bytes_written'] / 1e9:>12.2f} GB")
    print(f"Total data read:             {costs['total_bytes_read'] / 1e9:>12.2f} GB")
    print(f"")
    print(f"PUT request cost:            ${costs['put_cost']:>12.4f}")
    print(f"GET request cost:            ${costs['get_cost']:>12.4f}")
    print(f"Data transfer out cost:      ${costs['transfer_cost']:>12.4f}")
    print(f"Storage cost (per month):    ${costs['storage_cost_month']:>12.4f}")
    print(f"{'─' * 60}")
    print(f"Total (excl. storage):       ${costs['total_excl_storage']:>12.4f}")
    print(f"{'=' * 60}")

    # Append overall S3 call/cost rollups into the JSON stats section.
    if isinstance(payload, dict) and "stats" in payload:
        payload["stats"]["s3_read_calls_total"] = costs["total_reads"]
        payload["stats"]["s3_write_calls_total"] = costs["total_writes"]
        payload["stats"]["s3_delete_calls_total"] = costs["total_deletes"]
        payload["stats"]["bytes_read_total"] = int(costs["total_bytes_read"])
        payload["stats"]["bytes_written_total"] = int(costs["total_bytes_written"])
        payload["stats"]["put_cost_usd"] = round(float(costs["put_cost"]), 8)
        payload["stats"]["get_cost_usd"] = round(float(costs["get_cost"]), 8)
        payload["stats"]["transfer_cost_usd"] = round(float(costs["transfer_cost"]), 8)
        payload["stats"]["storage_cost_per_month_usd"] = round(float(costs["storage_cost_month"]), 8)
        payload["stats"]["total_cost_excl_storage_usd"] = round(float(costs["total_excl_storage"]), 8)
        with open(bench_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    if run_plot:
        plot_results(df, output_dir=OUTPUT_DIR, show_plot=show_plot)
    return bench_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S3 Quake benchmark and/or plotting.")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "plot", "both"],
        default="both",
        help="benchmark: run benchmark only, plot: plot from saved JSON only, both: run benchmark and then plot",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Path to benchmark JSON (required for --mode plot, optional for --mode both if you want re-plot).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plot image without opening a GUI window.",
    )
    args = parser.parse_args()

    if args.mode == "benchmark":
        benchmark(run_plot=False, show_plot=not args.no_show)
    elif args.mode == "plot":
        if not args.json:
            raise ValueError("--json is required when --mode plot")
        plot_df = load_batch_df_from_json(Path(args.json))
        plot_results(plot_df, output_dir=OUTPUT_DIR, show_plot=not args.no_show)
    else:
        bench_json = benchmark(run_plot=False, show_plot=not args.no_show)
        target_json = Path(args.json) if args.json else Path(bench_json)
        plot_df = load_batch_df_from_json(target_json)
        plot_results(plot_df, output_dir=OUTPUT_DIR, show_plot=not args.no_show)
