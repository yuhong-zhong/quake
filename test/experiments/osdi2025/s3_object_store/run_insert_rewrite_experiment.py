#!/usr/bin/env python3
"""
Build 10M base vectors, then run one incremental insert (100K/200K/500K/1M)
per fresh index to quantify partition rewrites.
"""

import hashlib
import json
import math
import os
import shutil
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import quake
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[4]
    build_libs = sorted(repo_root.glob("build/lib.*"))
    for build_lib in build_libs:
        if (build_lib / "quake").exists():
            sys.path.insert(0, str(build_lib))
            break
    import quake


DATASET_PATH = "/users/charlesx/SPTAG/datasets/SPACEV1B/vectors.bin/vectors_merged.bin"
OUTPUT_DIR = Path("test/experiments/osdi2025/s3_object_store/results/spacev1b_10m")
TEMP_DIR = OUTPUT_DIR / "_tmp_index_rewrite"

BASE_VECTORS = 10_000_000
INSERT_SIZES = [100_000, 200_000, 500_000, 1_000_000]
NLIST = 1024
METRIC = "l2"
PART_SIZE = 8 * 1024 * 1024
MAX_PARTITION_BYTES = 8 * 1024 * 1024
MIN_PARTITION_BYTES = 1 * 1024 * 1024

HEADER_FMT = "<IIQQQ"
HEADER_BYTES = struct.calcsize(HEADER_FMT)

PUT_PRICE_PER_1K = 0.005
GET_PRICE_PER_1K = 0.0004
TRANSFER_PER_GB = 0.09
MAX_MAINTENANCE_ROUNDS = 32


def s3_upload_calls(size_bytes: int) -> int:
    if size_bytes <= 0:
        return 0
    if size_bytes < PART_SIZE:
        return 1
    return int(math.ceil(size_bytes / PART_SIZE)) + 2


def s3_download_calls(size_bytes: int) -> int:
    if size_bytes <= 0:
        return 0
    if size_bytes < PART_SIZE:
        return 1
    return int(math.ceil(size_bytes / PART_SIZE))


def parse_partitions_with_hash(partitions_path: Path, d: int):
    """Return dict[pid] -> {size_bytes, size_mib, num_vectors, hash}."""
    out = {}
    bytes_per_vector_quake = (d * 4) + 8
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
            size_bytes = chunk_end - chunk_start

            f.seek(data_start + chunk_start)
            hasher = hashlib.sha256()
            remaining = size_bytes
            while remaining > 0:
                data = f.read(min(4 * 1024 * 1024, remaining))
                if not data:
                    raise ValueError(f"Unexpected EOF while hashing partition {pid}")
                hasher.update(data)
                remaining -= len(data)

            out[pid] = {
                "size_bytes": int(size_bytes),
                "size_mib": round(size_bytes / (1024 ** 2), 6),
                "num_vectors": int(size_bytes // bytes_per_vector_quake),
                "hash": hasher.hexdigest(),
            }
    return out


def read_max_partition_size_bytes(partitions_path: Path) -> int:
    """Read only max partition chunk size from serialized partitions file."""
    with open(partitions_path, "rb") as f:
        raw = f.read(HEADER_BYTES)
        _, _, _, _, num_parts = struct.unpack(HEADER_FMT, raw)
        offsets = np.fromfile(f, dtype=np.uint64, count=num_parts + 1)
        if num_parts == 0:
            return 0
        sizes = offsets[1:] - offsets[:-1]
        return int(sizes.max())


def maintenance_until_partition_cap(index, max_partition_bytes: int):
    """
    Run maintenance repeatedly until all partitions are <= max_partition_bytes.
    Returns (total_maintenance_latency_ms, rounds_run).
    """
    total_ms = 0.0
    rounds = 0

    for _ in range(MAX_MAINTENANCE_ROUNDS):
        t = time.perf_counter()
        index.maintenance()
        total_ms += (time.perf_counter() - t) * 1000.0
        rounds += 1

        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        index.save(str(TEMP_DIR))
        max_size = read_max_partition_size_bytes(TEMP_DIR / "partitions")
        if max_size <= max_partition_bytes:
            return total_ms, rounds

    raise RuntimeError(
        f"Maintenance did not reach <= {max_partition_bytes} bytes per partition "
        f"after {MAX_MAINTENANCE_ROUNDS} rounds."
    )


def diff_snapshots(prev_parts: dict, curr_parts: dict):
    created = 0
    changed = 0
    deleted = 0
    read_calls = 0
    write_calls = 0
    delete_calls = 0
    bytes_read = 0
    bytes_written = 0

    for pid, curr in curr_parts.items():
        curr_size = curr["size_bytes"]
        if pid not in prev_parts:
            created += 1
            write_calls += s3_upload_calls(curr_size)
            bytes_written += curr_size
        elif prev_parts[pid]["hash"] != curr["hash"]:
            changed += 1
            read_calls += s3_download_calls(curr_size)
            write_calls += s3_upload_calls(curr_size)
            bytes_read += curr_size
            bytes_written += curr_size

    for pid in prev_parts:
        if pid not in curr_parts:
            deleted += 1
            delete_calls += 1

    return {
        "partitions_created": created,
        "partitions_changed": changed,
        "partitions_deleted": deleted,
        "s3_read_calls": read_calls,
        "s3_write_calls": write_calls,
        "s3_delete_calls": delete_calls,
        "bytes_read": int(bytes_read),
        "bytes_written": int(bytes_written),
    }


def costs_from_diff(diff: dict):
    put_cost = (diff["s3_write_calls"] / 1000.0) * PUT_PRICE_PER_1K
    get_cost = (diff["s3_read_calls"] / 1000.0) * GET_PRICE_PER_1K
    transfer_cost = (diff["bytes_read"] / 1e9) * TRANSFER_PER_GB
    return {
        "put_request_cost_usd": round(float(put_cost), 8),
        "get_request_cost_usd": round(float(get_cost), 8),
        "data_transfer_cost_usd": round(float(transfer_cost), 8),
        "total_cost_usd": round(float(put_cost + get_cost + transfer_cost), 8),
    }


def stage_stats(op_latency_ms: float, maintenance_latency_ms: float, parts: dict, diff: dict):
    if parts:
        sizes = np.array([v["size_bytes"] for v in parts.values()], dtype=np.int64)
        max_size_mib = float(sizes.max()) / (1024 ** 2)
        min_size_mib = float(sizes.min()) / (1024 ** 2)
        p95_size_mib = float(np.percentile(sizes, 95)) / (1024 ** 2)
    else:
        max_size_mib = 0.0
        min_size_mib = 0.0
        p95_size_mib = 0.0

    stats = {
        "partition_created": int(diff["partitions_created"]),
        "partition_changed": int(diff["partitions_changed"]),
        "partition_deleted": int(diff["partitions_deleted"]),
        "final_total_partition": int(len(parts)),
        "op_latency_ms": round(float(op_latency_ms), 2),
        "maintenance_latency_ms": round(float(maintenance_latency_ms), 2),
        "max_size_per_partition_mib": round(max_size_mib, 6),
        "min_size_per_partition_mib": round(min_size_mib, 6),
        "p95_size_per_partition_mib": round(p95_size_mib, 6),
        "s3_read_calls": int(diff["s3_read_calls"]),
        "s3_write_calls": int(diff["s3_write_calls"]),
        "s3_delete_calls": int(diff["s3_delete_calls"]),
        "bytes_read": int(diff["bytes_read"]),
        "bytes_written": int(diff["bytes_written"]),
    }
    stats.update(costs_from_diff(diff))
    return stats


def parts_to_result_list(parts: dict):
    rows = []
    for pid in sorted(parts.keys()):
        p = parts[pid]
        rows.append({
            "partition_id": int(pid),
            "size_bytes": int(p["size_bytes"]),
            "size_mib": float(p["size_mib"]),
            "num_vectors": int(p["num_vectors"]),
        })
    return rows


def vector_tensor(dataset, start: int, end: int):
    x = torch.from_numpy(dataset[start:end].copy()).to(torch.float32)
    ids = torch.arange(start, end, dtype=torch.int64)
    return x, ids


def run_one_insert(dataset, d: int, insert_size: int, min_part_vecs: int, max_part_vecs: int):
    index = quake.QuakeIndex()

    x_build, ids_build = vector_tensor(dataset, 0, BASE_VECTORS)
    t0 = time.perf_counter()
    build_params = quake.IndexBuildParams()
    build_params.nlist = NLIST
    build_params.metric = METRIC
    index.build(x_build, ids_build, build_params)
    op_build_ms = (time.perf_counter() - t0) * 1000.0

    maintenance_params = quake.MaintenancePolicyParams()
    maintenance_params.min_partition_size = min_part_vecs
    maintenance_params.max_partition_size = max_part_vecs
    index.initialize_maintenance_policy(maintenance_params)

    maint_build_ms, maint_build_rounds = maintenance_until_partition_cap(index, MAX_PARTITION_BYTES)

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    index.save(str(TEMP_DIR))
    build_parts = parse_partitions_with_hash(TEMP_DIR / "partitions", d)
    build_diff = diff_snapshots({}, build_parts)

    ins_start = BASE_VECTORS
    ins_end = BASE_VECTORS + insert_size
    x_add, ids_add = vector_tensor(dataset, ins_start, ins_end)

    t2 = time.perf_counter()
    index.add(x_add, ids_add)
    op_add_ms = (time.perf_counter() - t2) * 1000.0

    maint_add_ms, maint_add_rounds = maintenance_until_partition_cap(index, MAX_PARTITION_BYTES)

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    index.save(str(TEMP_DIR))
    insert_parts = parse_partitions_with_hash(TEMP_DIR / "partitions", d)
    insert_diff = diff_snapshots(build_parts, insert_parts)

    return index, {
        "build_stats": stage_stats(op_build_ms, maint_build_ms, build_parts, build_diff),
        "insert_stats": stage_stats(op_add_ms, maint_add_ms, insert_parts, insert_diff),
        "build_maintenance_rounds": int(maint_build_rounds),
        "insert_maintenance_rounds": int(maint_add_rounds),
        "build_parts": parts_to_result_list(build_parts),
        "insert_parts": parts_to_result_list(insert_parts),
    }


def write_json_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "rb") as f:
        total_vectors = struct.unpack("i", f.read(4))[0]
        d = struct.unpack("i", f.read(4))[0]

    need = BASE_VECTORS + max(INSERT_SIZES)
    if total_vectors < need:
        raise ValueError(f"Dataset has {total_vectors} vectors, need at least {need}")

    dataset = np.memmap(
        DATASET_PATH,
        dtype=np.int8,
        mode="r",
        offset=8,
        shape=(need, d),
    )

    bytes_per_vector_quake = (d * 4) + 8
    min_part_vecs = max(1, MIN_PARTITION_BYTES // bytes_per_vector_quake)
    max_part_vecs = max(1, MAX_PARTITION_BYTES // bytes_per_vector_quake)

    payload = {
        "stats": {},
        "results": {},
    }
    ts = datetime.now().strftime("%d_%m_%H%M%S")
    out = OUTPUT_DIR / f"s3_quake_rewrite_sensitivity_{ts}.json"
    payload["meta"] = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_file": str(out),
    }
    write_json_atomic(out, payload)

    print(f"Dataset: n={total_vectors}, d={d}")
    print(f"Build vectors: {BASE_VECTORS:,}")
    print(f"Insert sizes: {INSERT_SIZES}")
    print(f"Maintenance policy: min={min_part_vecs} vectors, max={max_part_vecs} vectors")

    for insert_size in INSERT_SIZES:
        insert_key = f"{insert_size // 1000}K_insert" if insert_size < 1_000_000 else "1M_insert"
        print(f"\n=== Run: build 10M + insert {insert_size:,} ===")
        index, run = run_one_insert(dataset, d, insert_size, min_part_vecs, max_part_vecs)

        # build stats/partitions should be same shape each run; keep first run's build snapshot.
        if "10M_build" not in payload["stats"]:
            payload["stats"]["10M_build"] = run["build_stats"]
            payload["stats"]["10M_build"]["maintenance_rounds"] = run["build_maintenance_rounds"]
            payload["results"]["10M_build"] = run["build_parts"]

        payload["stats"][insert_key] = run["insert_stats"]
        payload["stats"][insert_key]["maintenance_rounds"] = run["insert_maintenance_rounds"]
        payload["results"][insert_key] = run["insert_parts"]
        payload["meta"]["last_completed_stage"] = insert_key
        write_json_atomic(out, payload)

    payload["meta"]["status"] = "completed"
    payload["meta"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(out, payload)

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
