#!/usr/bin/env python3
"""
Memory diagnostics for PaCS-SR parallel processing.
Run this on a SLURM node to estimate safe worker count.
"""

import sys
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil")
    sys.exit(1)

def diagnose_memory():
    """Diagnose memory and recommend worker count."""

    mem = psutil.virtual_memory()

    print("=" * 60)
    print("PaCS-SR Memory Diagnostics")
    print("=" * 60)
    print(f"\nSystem Memory:")
    print(f"  Total:     {mem.total / (1024**3):.2f} GB")
    print(f"  Available: {mem.available / (1024**3):.2f} GB")
    print(f"  Used:      {mem.used / (1024**3):.2f} GB")
    print(f"  Percent:   {mem.percent:.1f}%")

    # Estimate memory per worker
    print(f"\nMemory Per Worker Estimates:")
    print(f"  Without registration: ~1.5 GB")
    print(f"  With ANTs registration: ~2.5 GB")

    # Calculate safe worker counts
    available_gb = mem.available / (1024**3)
    usable_gb = available_gb * 0.5  # Use 50% of available

    workers_no_reg = max(1, int(usable_gb / 1.5))
    workers_with_reg = max(1, int(usable_gb / 2.5))

    print(f"\nRecommended Worker Counts:")
    print(f"  Usable memory (50% of available): {usable_gb:.2f} GB")
    print(f"  Without registration: {workers_no_reg} workers")
    print(f"  With ANTs registration: {workers_with_reg} workers")

    # Check if enough memory for minimum parallelism
    if workers_no_reg < 4:
        print(f"\n⚠️  WARNING: Low available memory!")
        print(f"    Current available: {available_gb:.2f} GB")
        print(f"    Recommended minimum: 6 GB for 4 workers")
        print(f"    Consider:")
        print(f"      - Requesting more memory in SLURM (#SBATCH --mem=32G)")
        print(f"      - Reducing concurrent jobs on this node")
    else:
        print(f"\n✓ Memory looks good for parallel processing")

    # Environment variable suggestion
    print(f"\nTo manually limit workers, set environment variable:")
    print(f"  export PACS_SR_MAX_WORKERS={workers_no_reg}")
    print(f"  # or in SLURM script:")
    print(f"  #SBATCH --export=ALL,PACS_SR_MAX_WORKERS={workers_no_reg}")

    print("=" * 60)


if __name__ == "__main__":
    diagnose_memory()
