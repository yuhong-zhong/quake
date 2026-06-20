#!/usr/bin/env python3
"""Plot partition rewrite sensitivity from s3_quake_rewrite_sensitivity JSON."""

import argparse
import json
import sys
from pathlib import Path


def latest_result_json(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("s3_quake_rewrite_sensitivity_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No rewrite sensitivity JSON found in {results_dir}")
    return files[0]


def load_series(stats: dict):
    order = ["10M_build", "100K_insert", "200K_insert", "500K_insert", "1M_insert"]
    labels = []
    created = []
    changed = []
    deleted = []
    for key in order:
        if key not in stats:
            continue
        if key == "10M_build":
            labels.append("10M base")
        else:
            labels.append(key.replace("_insert", ""))
        created.append(int(stats[key].get("partition_created", 0)))
        changed.append(int(stats[key].get("partition_changed", 0)))
        deleted.append(int(stats[key].get("partition_deleted", 0)))
    return labels, created, changed, deleted


def main():
    parser = argparse.ArgumentParser(description="Plot rewrite counts from rewrite sensitivity JSON.")
    parser.add_argument("--json", default=None, help="Path to s3_quake_rewrite_sensitivity_*.json")
    parser.add_argument("--no-show", action="store_true", help="Save plot only, do not open GUI window.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_results_dir = script_dir / "results" / "spacev1b_10m"
    json_path = Path(args.json) if args.json else latest_result_json(default_results_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload.get("stats", {})
    labels, created, changed, deleted = load_series(stats)
    if not labels:
        raise ValueError(f"No insert stats found in {json_path}")

    # Try to avoid libstdc++ mismatch in some envs.
    try:
        import ctypes
        conda_libstdcpp = Path(sys.prefix) / "lib" / "libstdc++.so.6"
        if conda_libstdcpp.exists():
            ctypes.CDLL(str(conda_libstdcpp), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    w = 0.24

    fig, ax1 = plt.subplots(1, 1, figsize=(10.5, 5.4))

    bars_created = ax1.bar(x - w, created, width=w, label="Created")
    bars_changed = ax1.bar(x, changed, width=w, label="Changed")
    bars_deleted = ax1.bar(x + w, deleted, width=w, label="Deleted")
    ax1.set_title("Partition Rewrites Under Incremental Inserts")
    ax1.set_xlabel("Insert Size")
    ax1.set_ylabel("Partition Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for bars in (bars_created, bars_changed, bars_deleted):
        for b in bars:
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout(rect=(0, 0, 0.84, 1))
    out = json_path.with_name(json_path.stem + "_rewrite_counts.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved plot: {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
