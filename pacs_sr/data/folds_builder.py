"""
Build K-fold patient-level splits and a nested JSON manifest for SR MoE experiments.

The manifest groups, for each fold ∈ {1..K}, a train/ test split.
Inside each split, patients are keys. Each patient maps to paths for:
  - Per-model SR outputs (BSPLINE, ECLARE, SMORE, UNIRES) at spacings {3mm,5mm,7mm} and pulses {t1c,t1n,t2w,t2f}
  - Low-resolution inputs (LR) at spacings {3mm,5mm,7mm}
  - High-resolution targets (HR) (duplicated under each spacing key for convenience)
The script enforces patient-level CV: the same patient never appears in both train and test.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import random

# --------------------------- configuration dataclasses ---------------------------

@dataclass(frozen=True)
class BuilderConfig:
    models_root: Path         # /media/.../results/models
    hr_root: Path             # /media/.../high_resolution
    lr_root: Path             # ~/research/.../low_res
    spacings: Tuple[str, ...] # e.g., ("3mm","5mm","7mm")
    pulses: Tuple[str, ...]   # e.g., ("t1c","t1n","t2w","t2f")
    models: Tuple[str, ...]   # e.g., ("BSPLINE","ECLARE","SMORE","UNIRES")
    k_folds: int
    seed: int

# --------------------------- logging ---------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

# --------------------------- filesystem helpers ---------------------------

def list_patients_with_complete_coverage(cfg: BuilderConfig) -> List[str]:
    """
    Scan the filesystem and return patient IDs present with complete coverage:
    - For every model, spacing, pulse: SR output exists at
      {models_root}/{MODEL}/{spacing}/output_volumes/{patient}-{pulse}.nii.gz
    - For HR: {hr_root}/{patient}/{patient}-{pulse}.nii.gz
    - For LR (optional): {lr_root}/{spacing}/{patient}/{patient}-{pulse}.nii.gz
    Patients missing any required file (HR or SR) are excluded. LR is optional.
    """
    logging.info("Scanning filesystem to assemble candidate patients")
    # Candidate set: patients present in HR root
    candidates: Set[str] = set(
        p.name for p in cfg.hr_root.iterdir() if p.is_dir()
    )
    logging.info("Initial HR candidates: %d", len(candidates))

    def sr_path_ok(model: str, spacing: str, patient: str, pulse: str) -> bool:
        p = cfg.models_root / model / spacing / "output_volumes" / f"{patient}-{pulse}.nii.gz"
        return p.is_file()

    def lr_path_ok(spacing: str, patient: str, pulse: str) -> bool:
        p = cfg.lr_root / spacing / patient / f"{patient}-{pulse}.nii.gz"
        return p.is_file()

    def hr_path_ok(patient: str, pulse: str) -> bool:
        p = cfg.hr_root / patient / f"{patient}-{pulse}.nii.gz"
        return p.is_file()

    keep: Set[str] = set()
    drop: Set[str] = set()
    for patient in sorted(candidates):
        ok = True
        # HR presence (independent of spacing; we check once per pulse)
        for pulse in cfg.pulses:
            if not hr_path_ok(patient, pulse):
                ok = False
                logging.warning("Missing HR for %s %s", patient, pulse)
                break
        if not ok:
            drop.add(patient)
            continue
        # LR per spacing and pulse (optional - log if missing but don't exclude)
        has_lr = True
        for spacing in cfg.spacings:
            for pulse in cfg.pulses:
                if not lr_path_ok(spacing, patient, pulse):
                    has_lr = False
                    logging.debug("Missing LR for %s %s %s (non-fatal)", patient, spacing, pulse)
                    break
            if not has_lr:
                break
        # SR per model, spacing, pulse
        for model in cfg.models:
            for spacing in cfg.spacings:
                for pulse in cfg.pulses:
                    if not sr_path_ok(model, spacing, patient, pulse):
                        ok = False
                        logging.warning("Missing SR for %s | %s %s %s", patient, model, spacing, pulse)
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

def make_patient_entry(cfg: BuilderConfig, patient: str) -> Dict:
    """
    Create the nested dict for one patient with the exact structure requested.
    LR entries are only included if all required LR files exist.
    """
    entry: Dict[str, Dict] = {}
    # Per-model SR
    for model in cfg.models:
        model_block: Dict[str, Dict[str, str]] = {}
        for spacing in cfg.spacings:
            spacing_block: Dict[str, str] = {}
            for pulse in cfg.pulses:
                spacing_block[pulse] = str(
                    cfg.models_root / model / spacing / "output_volumes" / f"{patient}-{pulse}.nii.gz"
                )
            model_block[spacing] = spacing_block
        entry[model] = model_block
    # LR (optional - only include if all files exist)
    lr_block: Dict[str, Dict[str, str]] = {}
    has_all_lr = True
    for spacing in cfg.spacings:
        spacing_block = {}
        for pulse in cfg.pulses:
            lr_path = cfg.lr_root / spacing / patient / f"{patient}-{pulse}.nii.gz"
            if lr_path.is_file():
                spacing_block[pulse] = str(lr_path)
            else:
                has_all_lr = False
                break
        if not has_all_lr:
            break
        lr_block[spacing] = spacing_block
    if has_all_lr:
        entry["LR"] = lr_block
    # HR (duplicated under each spacing key for convenience)
    hr_block: Dict[str, Dict[str, str]] = {}
    for spacing in cfg.spacings:
        spacing_block = {}
        for pulse in cfg.pulses:
            spacing_block[pulse] = str(
                cfg.hr_root / patient / f"{patient}-{pulse}.nii.gz"
            )
        hr_block[spacing] = spacing_block
    entry["HR"] = hr_block
    return entry

# --------------------------- CV split ---------------------------

def kfold_split(patients: List[str], k: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    """
    Produce K folds as (train_list, test_list). Random shuffle then partition.
    """
    rng = random.Random(seed)
    pts = patients.copy()
    rng.shuffle(pts)
    n = len(pts)
    folds: List[Tuple[List[str], List[str]]] = []
    fold_sizes = [(n + i) // k for i in range(k)]
    start = 0
    for size in fold_sizes:
        test = pts[start:start+size]
        train = [p for p in pts if p not in test]
        folds.append((train, test))
        start += size
    return folds

# --------------------------- main ---------------------------

def build_manifest(cfg: BuilderConfig) -> Dict:
    """
    Assemble the full nested JSON manifest across K folds.
    """
    patients = list_patients_with_complete_coverage(cfg)
    assert len(patients) >= cfg.k_folds, "Not enough patients for requested K folds"
    folds = kfold_split(patients, cfg.k_folds, cfg.seed)
    folds_list = []
    for i, (train, test) in enumerate(folds, start=1):
        fold_data = {"train": {}, "test": {}}
        for patient in train:
            fold_data["train"][patient] = make_patient_entry(cfg, patient)
        for patient in test:
            fold_data["test"][patient] = make_patient_entry(cfg, patient)
        folds_list.append(fold_data)
        logging.info("Fold %d | train=%d test=%d", i, len(train), len(test))
    return {"folds": folds_list}


def build_kfold_manifest(
    models_root: Path,
    hr_root: Path,
    lr_root: Optional[Path],
    spacings: List[str],
    pulses: List[str],
    models: List[str],
    kfolds: int,
    seed: int,
) -> Dict:
    """
    Convenience wrapper for building K-fold manifest.

    This function provides a simpler interface for the pipeline orchestrator.

    Args:
        models_root: Root directory containing SR model outputs
        hr_root: Root directory containing HR ground truth
        lr_root: Root directory containing LR inputs (optional)
        spacings: List of spacing values (e.g., ["3mm", "5mm", "7mm"])
        pulses: List of pulse sequences (e.g., ["t1c", "t1n", "t2w", "t2f"])
        models: List of expert model names (e.g., ["BSPLINE", "ECLARE", "SMORE"])
        kfolds: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing the K-fold manifest structure
    """
    setup_logging()

    cfg = BuilderConfig(
        models_root=Path(models_root).expanduser().resolve(),
        hr_root=Path(hr_root).expanduser().resolve(),
        lr_root=Path(lr_root).expanduser().resolve() if lr_root else Path("/tmp"),
        spacings=tuple(spacings),
        pulses=tuple(pulses),
        models=tuple(models),
        k_folds=kfolds,
        seed=seed,
    )

    return build_manifest(cfg)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build K-fold JSON manifest for SR MoE.")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--hr-root", type=Path, required=True)
    parser.add_argument("--lr-root", type=Path, required=True)
    parser.add_argument("--spacings", type=str, default="3mm,5mm,7mm")
    parser.add_argument("--pulses", type=str, default="t1c,t1n,t2w,t2f")
    parser.add_argument("--models", type=str, default="BSPLINE,ECLARE,SMORE,UNIRES")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()

    """
    python folds_builder.py \
        --models-root /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/models \
        --hr-root /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution \
        --lr-root /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/low_res \
        --spacings 3mm,5mm,7mm \
        --pulses t1c,t1n,t2w,t2f \
        --models BSPLINE,ECLARE,SMORE,UNIRES \
        --kfolds 5 \
        --seed 42 \
        --out /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/kfolds_manifest.json
    """

def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = BuilderConfig(
        models_root=args.models_root.expanduser().resolve(),
        hr_root=args.hr_root.expanduser().resolve(),
        lr_root=args.lr_root.expanduser().resolve(),
        spacings=tuple(s.strip() for s in args.spacings.split(",")),
        pulses=tuple(s.strip() for s in args.pulses.split(",")),
        models=tuple(s.strip() for s in args.models.split(",")),
        k_folds=args.kfolds,
        seed=args.seed
    )
    manifest = build_manifest(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logging.info("Wrote manifest to %s", args.out)

if __name__ == "__main__":
    main()
