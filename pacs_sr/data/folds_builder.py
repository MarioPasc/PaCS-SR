"""Build K-fold patient-level splits and a simplified JSON manifest.

With HDF5 storage, paths are deterministic from config. The manifest just stores
patient IDs per fold:

    {"folds": [{"train": ["PAT001", ...], "test": ["PAT002", ...]}, ...]}

HDF5 keys are constructed at runtime via helpers in hdf5_io.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

from pacs_sr.data.hdf5_io import (
    expert_h5_path,
    expert_key,
    has_key,
    hr_key,
    list_groups,
)

# --------------------------- configuration dataclasses ---------------------------


@dataclass(frozen=True)
class BuilderConfig:
    source_h5: Path  # path to source_data.h5
    experts_dir: Path  # directory containing {model}.h5 files
    spacings: Tuple[str, ...]
    pulses: Tuple[str, ...]
    models: Tuple[str, ...]
    k_folds: int
    seed: int


# --------------------------- logging ---------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


# --------------------------- HDF5 scanning ---------------------------


def list_patients_with_complete_coverage(cfg: BuilderConfig) -> List[str]:
    """Scan HDF5 files and return patient IDs present with complete coverage.

    Requires:
    - HR volume for every pulse in source_data.h5
    - Expert output for every (model, spacing, pulse) in {model}.h5

    Args:
        cfg: Builder configuration with HDF5 paths.

    Returns:
        Sorted list of patient IDs with complete data coverage.
    """
    logging.info("Scanning HDF5 files for candidate patients")

    # Candidate set: patients present under high_resolution in source_data.h5
    candidates: Set[str] = set(list_groups(cfg.source_h5, "high_resolution"))
    logging.info("Initial HR candidates: %d", len(candidates))

    keep: Set[str] = set()
    drop: Set[str] = set()

    for patient in sorted(candidates):
        ok = True

        # HR presence: check every pulse
        for pulse in cfg.pulses:
            if not has_key(cfg.source_h5, hr_key(patient, pulse)):
                ok = False
                logging.warning("Missing HR for %s %s", patient, pulse)
                break
        if not ok:
            drop.add(patient)
            continue

        # Expert outputs: check every (model, spacing, pulse)
        for model in cfg.models:
            model_h5 = expert_h5_path(cfg.experts_dir, model)
            for spacing in cfg.spacings:
                for pulse in cfg.pulses:
                    if not has_key(model_h5, expert_key(spacing, patient, pulse)):
                        ok = False
                        logging.warning(
                            "Missing SR for %s | %s %s %s",
                            patient,
                            model,
                            spacing,
                            pulse,
                        )
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            keep.add(patient)
        else:
            drop.add(patient)

    kept = sorted(list(keep))
    logging.info("Patients with complete coverage: %d", len(kept))
    if drop:
        logging.info("Excluded patients due to missing files: %d", len(drop))
    return kept


# --------------------------- CV split ---------------------------


def kfold_split(
    patients: List[str], k: int, seed: int
) -> List[Tuple[List[str], List[str]]]:
    """Produce K folds as (train_list, test_list).

    Args:
        patients: Sorted list of patient IDs.
        k: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of (train, test) tuples of patient ID lists.
    """
    rng = random.Random(seed)
    pts = patients.copy()
    rng.shuffle(pts)
    n = len(pts)
    folds: List[Tuple[List[str], List[str]]] = []
    fold_sizes = [(n + i) // k for i in range(k)]
    start = 0
    for size in fold_sizes:
        test = pts[start : start + size]
        train = [p for p in pts if p not in test]
        folds.append((train, test))
        start += size
    return folds


# --------------------------- main ---------------------------


def build_manifest(cfg: BuilderConfig) -> Dict:
    """Assemble the simplified JSON manifest across K folds.

    Args:
        cfg: Builder configuration.

    Returns:
        Dict with structure: {"folds": [{"train": [...], "test": [...]}, ...]}
    """
    patients = list_patients_with_complete_coverage(cfg)
    assert len(patients) >= cfg.k_folds, "Not enough patients for requested K folds"
    folds = kfold_split(patients, cfg.k_folds, cfg.seed)
    folds_list = []
    for i, (train, test) in enumerate(folds, start=1):
        folds_list.append({"train": train, "test": test})
        logging.info("Fold %d | train=%d test=%d", i, len(train), len(test))
    return {"folds": folds_list}


def build_kfold_manifest(
    source_h5: Path,
    experts_dir: Path,
    spacings: List[str],
    pulses: List[str],
    models: List[str],
    kfolds: int,
    seed: int,
) -> Dict:
    """Convenience wrapper for building K-fold manifest.

    Args:
        source_h5: Path to source_data.h5.
        experts_dir: Directory containing {model}.h5 files.
        spacings: List of spacing values.
        pulses: List of pulse sequences.
        models: List of expert model names.
        kfolds: Number of cross-validation folds.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing the K-fold manifest structure.
    """
    setup_logging()

    cfg = BuilderConfig(
        source_h5=Path(source_h5).expanduser().resolve(),
        experts_dir=Path(experts_dir).expanduser().resolve(),
        spacings=tuple(spacings),
        pulses=tuple(pulses),
        models=tuple(models),
        k_folds=kfolds,
        seed=seed,
    )

    return build_manifest(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build K-fold JSON manifest for SR MoE."
    )
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--experts-dir", type=Path, required=True)
    parser.add_argument("--spacings", type=str, default="3mm,5mm,7mm")
    parser.add_argument("--pulses", type=str, default="t1c,t2w,t2f")
    parser.add_argument("--models", type=str, default="BSPLINE,ECLARE")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = BuilderConfig(
        source_h5=args.source_h5.expanduser().resolve(),
        experts_dir=args.experts_dir.expanduser().resolve(),
        spacings=tuple(s.strip() for s in args.spacings.split(",")),
        pulses=tuple(s.strip() for s in args.pulses.split(",")),
        models=tuple(s.strip() for s in args.models.split(",")),
        k_folds=args.kfolds,
        seed=args.seed,
    )
    manifest = build_manifest(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logging.info("Wrote manifest to %s", args.out)


if __name__ == "__main__":
    main()
