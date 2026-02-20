#!/usr/bin/env python3
"""Merge ECLARE HDF5 shards into the main eclare.h5 file.

After sharded ECLARE generation, each shard writes to
``eclare_shard_{i}.h5``. This script copies all datasets from shards into
the main ``eclare.h5``, skipping keys that already exist. Shard files are
deleted after successful merge.

Usage:
    python scripts/merge_eclare_shards.py \
        --experts-dir /path/to/experts \
        --num-shards 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def merge_shards(experts_dir: Path, num_shards: int, *, keep: bool = False) -> None:
    """Merge shard HDF5 files into the main eclare.h5.

    Args:
        experts_dir: Directory containing eclare.h5 and shard files.
        num_shards: Number of shards to merge.
        keep: If True, do not delete shard files after merge.
    """
    main_h5 = experts_dir / "eclare.h5"
    copied = 0
    skipped = 0

    for shard_id in range(num_shards):
        shard_path = experts_dir / f"eclare_shard_{shard_id}.h5"
        if not shard_path.exists():
            print(f"[shard {shard_id}] Not found: {shard_path} — skipping")
            continue

        with h5py.File(shard_path, "r") as src, h5py.File(main_h5, "a") as dst:
            n_src = _count_datasets(src)
            stats = _merge_stats(src, dst)
            print(
                f"[shard {shard_id}] {n_src} datasets in {shard_path.name}"
                f" → copy {stats['copied']}, skip {stats['skipped']}"
            )
            _copy_recursive(src, dst)

        print(f"  → merged into {main_h5.name}")

        if not keep:
            shard_path.unlink()
            print(f"  → deleted {shard_path.name}")

    # Final count
    if main_h5.exists():
        with h5py.File(main_h5, "r") as f:
            total = _count_datasets(f)
            print(f"\n[done] Total datasets in eclare.h5: {total}")
    else:
        print("\n[warn] eclare.h5 does not exist after merge")


def _count_datasets(grp: h5py.Group) -> int:
    """Count all datasets recursively."""
    n = 0
    for key in grp:
        if isinstance(grp[key], h5py.Dataset):
            n += 1
        else:
            n += _count_datasets(grp[key])
    return n


def _copy_recursive(
    src: h5py.Group,
    dst: h5py.File,
    stats: dict | None = None,
    prefix: str = "",
) -> None:
    """Recursively copy datasets from src into dst, skipping existing keys."""
    for key in src:
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(src[key], h5py.Dataset):
            if full_key not in dst:
                # Ensure parent groups exist
                parent = "/".join(full_key.split("/")[:-1])
                if parent and parent not in dst:
                    dst.require_group(parent)
                src.copy(src[key], dst, name=full_key)
                if stats is not None:
                    stats["copied"] += 1
            else:
                if stats is not None:
                    stats["skipped"] += 1
        else:
            _copy_recursive(src[key], dst, stats, full_key)


def _merge_stats(src: h5py.Group, dst: h5py.File, prefix: str = "") -> dict:
    """Count how many datasets would be copied vs skipped."""
    stats = {"copied": 0, "skipped": 0}
    for key in src:
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(src[key], h5py.Dataset):
            if full_key not in dst:
                stats["copied"] += 1
            else:
                stats["skipped"] += 1
        else:
            sub = _merge_stats(src[key], dst, full_key)
            stats["copied"] += sub["copied"]
            stats["skipped"] += sub["skipped"]
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Merge ECLARE HDF5 shards")
    parser.add_argument(
        "--experts-dir", type=Path, required=True, help="Directory with shard files"
    )
    parser.add_argument(
        "--num-shards", type=int, required=True, help="Number of shards"
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep shard files after merge"
    )
    args = parser.parse_args()

    merge_shards(args.experts_dir, args.num_shards, keep=args.keep)


if __name__ == "__main__":
    main()
