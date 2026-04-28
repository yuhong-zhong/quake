"""
Utility to upload a saved Quake index to S3.

Each partition is uploaded as a separate object with key:
    {prefix}/partition_{pid}          (legacy mode)
or as blocks:
    {prefix}/block_{block_id}         (block mode, when block_size > 0)

The object body is raw binary: codes (nv * code_size bytes) followed by
ids (nv * 8 bytes), matching what DynamicInvertedLists expects on the C++ side.

The metadata.txt and parent/ directory files are uploaded as-is so that
QuakeIndex::load() can still read centroid data from the local index directory
(or from a separately mirrored copy).
"""

import math
import os
import struct

import boto3


def _parse_partitions_manifest(partitions_path):
    """
    Read the header and manifest of the binary partitions file.

    Returns:
        code_size (int): bytes per vector
        pid_list  (list[int]): partition IDs in file order
        offsets   (list[int]): chunk offsets array (len == num_partitions + 1)
        data_start (int): byte offset in file where chunk data begins
    """
    with open(partitions_path, "rb") as f:
        # 32-byte header: magic(4) version(4) nlist(8) code_size(8) num_partitions(8)
        magic, version = struct.unpack("<II", f.read(8))
        nlist, code_size, num_partitions = struct.unpack("<QQQ", f.read(24))

        # offsets array: (num_partitions + 1) * uint64
        offsets = list(struct.unpack(f"<{num_partitions + 1}Q",
                                     f.read((num_partitions + 1) * 8)))

        # partition ID array: num_partitions * uint64
        pid_list = list(struct.unpack(f"<{num_partitions}Q",
                                      f.read(num_partitions * 8)))

        data_start = f.tell()

    return code_size, pid_list, offsets, data_start


def upload_index_to_s3(index_dir, bucket, prefix,
                       region="us-east-1", endpoint_url=None,
                       block_size=0):
    """
    Upload a saved Quake index directory to S3.

    Args:
        index_dir    (str): Path to the saved index directory (output of index.save()).
        bucket       (str): S3 bucket name.
        prefix       (str): S3 key prefix (no trailing slash).
        region       (str): AWS region.
        endpoint_url (str|None): Custom endpoint URL (e.g. for MinIO).
        block_size   (int): If > 0, upload as blocks instead of partitions.
    """
    s3 = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)

    partitions_path = os.path.join(index_dir, "partitions")
    code_size, pid_list, offsets, data_start = _parse_partitions_manifest(partitions_path)

    id_size = 8  # sizeof(int64_t)
    record_size = code_size + id_size
    dim = code_size // 4  # float32

    if block_size > 0:
        blocks_file = os.path.join(index_dir, "partitions.blocks")
        _upload_blocks_from_manifest(s3, blocks_file, bucket, prefix)
    else:
        print(f"Uploading {len(pid_list)} partitions to s3://{bucket}/{prefix}/")
        with open(partitions_path, "rb") as f:
            for i, pid in enumerate(pid_list):
                chunk_start = data_start + offsets[i]
                chunk_size = offsets[i + 1] - offsets[i]
                f.seek(chunk_start)
                data = f.read(chunk_size)
                key = f"{prefix}/partition_{pid}"
                s3.put_object(Bucket=bucket, Key=key, Body=data)
                if (i + 1) % 100 == 0 or (i + 1) == len(pid_list):
                    print(f"  {i + 1}/{len(pid_list)} partitions uploaded\r", end="")
        print()

    # Upload metadata.txt
    metadata_path = os.path.join(index_dir, "metadata.txt")
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            s3.put_object(Bucket=bucket, Key=f"{prefix}/metadata.txt", Body=f.read())
        print(f"Uploaded metadata.txt")

    # Recursively upload parent/ directory
    parent_dir = os.path.join(index_dir, "parent")
    if os.path.isdir(parent_dir):
        _upload_directory(s3, parent_dir, bucket, f"{prefix}/parent")
        print(f"Uploaded parent/ directory")

    print(f"Done. Index uploaded to s3://{bucket}/{prefix}/")


_MAGIC = 0x44494E4C
_ID_SIZE = 8


def _upload_blocks_from_manifest(s3, blocks_file, bucket, prefix):
    """Read pre-made blocks from .blocks file (created by C++ build) and upload to S3.

    The .blocks file layout (written by DynamicInvertedLists::save):
      [manifest: header, block_num_vectors, refcounts, tombstones, partition→blocks, memtables]
      [block data section: n_data_blocks, then (block_id, nv, codes, ids) for each]
    """
    if not os.path.exists(blocks_file):
        raise FileNotFoundError(
            f"Block manifest not found: {blocks_file}\n"
            "Did you set block_size in IndexBuildParams before calling build()?")

    with open(blocks_file, "rb") as f:
        # Read header
        magic, version = struct.unpack("<II", f.read(8))
        if magic != _MAGIC:
            raise ValueError(f"Bad magic in blocks file: {magic:#x}")
        curr_block_id = struct.unpack("<Q", f.read(8))[0]
        block_size = struct.unpack("<Q", f.read(8))[0]
        memtable_flush = struct.unpack("<Q", f.read(8))[0]
        metric_val = struct.unpack("<Q", f.read(8))[0]
        code_size = struct.unpack("<Q", f.read(8))[0]
        nlist = struct.unpack("<Q", f.read(8))[0]

        # Skip block_num_vectors
        n_blocks = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_blocks):
            f.read(16)  # bid, nv

        # Skip block refcounts
        n_refcounts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_refcounts):
            f.read(16)  # bid, rc

        # Skip tombstones
        n_ts_blocks = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_ts_blocks):
            _bid = struct.unpack("<Q", f.read(8))[0]
            n_ts = struct.unpack("<Q", f.read(8))[0]
            f.read(n_ts * _ID_SIZE)

        # Skip partition→blocks mapping + memtable data
        n_parts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_parts):
            _pid = struct.unpack("<Q", f.read(8))[0]
            nb = struct.unpack("<Q", f.read(8))[0]
            f.read(nb * 8)  # block_ids
            mt_nv = struct.unpack("<Q", f.read(8))[0]
            if mt_nv > 0:
                f.read(mt_nv * code_size + mt_nv * _ID_SIZE)

        # Read block data section
        n_data_blocks = struct.unpack("<Q", f.read(8))[0]
        print(f"Uploading {n_data_blocks} blocks to s3://{bucket}/{prefix}/")

        for i in range(n_data_blocks):
            bid, nv = struct.unpack("<QQ", f.read(16))
            data_size = nv * code_size + nv * _ID_SIZE
            data = f.read(data_size) if nv > 0 else b""
            key = f"{prefix}/block_{bid}"
            s3.put_object(Bucket=bucket, Key=key, Body=data)

            if (i + 1) % 100 == 0 or (i + 1) == n_data_blocks:
                print(f"  {i + 1}/{n_data_blocks} blocks uploaded\r", end="")

    print()


def _upload_directory(s3_client, local_dir, bucket, s3_prefix):
    """Recursively upload all files in local_dir to S3 under s3_prefix."""
    for entry in os.scandir(local_dir):
        if entry.is_file():
            key = f"{s3_prefix}/{entry.name}"
            with open(entry.path, "rb") as f:
                s3_client.put_object(Bucket=bucket, Key=key, Body=f.read())
        elif entry.is_dir():
            _upload_directory(s3_client, entry.path, bucket,
                              f"{s3_prefix}/{entry.name}")
