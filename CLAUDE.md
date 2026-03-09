# CLAUDE.md - PaCS-SR Project Documentation

## Executive Summary

**PaCS-SR** (Patchwise Convex Stacking for Super-Resolution) is an explainable AI approach for enhancing the resolution of clinical MRI scans acquired with anisotropic voxel spacing. Instead of relying on a single super-resolution (SR) method, PaCS-SR intelligently combines multiple expert SR models (BSPLINE, ECLARE, SMORE, UNIRES) by learning **region-specific weights** that indicate which expert performs best in different parts of the brain. This provides both improved image quality and clinical interpretability: radiologists can see exactly which method the system trusts most in each anatomical region.

---

## 1. Clinical Motivation

### The Problem: Anisotropic Clinical MRI

Clinical MRI acquisitions often have **anisotropic voxel spacing** - high in-plane resolution (e.g., 1mm) but thick slices (e.g., 3-7mm) to reduce scan time. This creates:

- Blurry reconstructions in the slice direction
- Reduced diagnostic quality for volumetric analysis
- Challenges for downstream tasks (segmentation, registration)

### Current Approaches Fall Short

Existing SR methods each have strengths and weaknesses:

| Method | Approach | Typical Strength |
|--------|----------|------------------|
| **BSPLINE** | Interpolation | Smooth, artifact-free |
| **ECLARE** | Deep learning | Detail enhancement |

No single method excels everywhere - one might be better for white matter, another for cortical regions.

### PaCS-SR Advantage

PaCS-SR asks: *"Why choose one expert when we can consult all of them?"*

Like a clinical consensus where multiple specialists contribute their expertise, PaCS-SR learns which expert to trust in which brain region, providing:

1. **Better image quality** - leverages each expert's strengths
2. **Explainability** - weight maps show the decision rationale
3. **Robustness** - ensemble reduces single-method failures

---

## 2. Methodology

### 2.1 Core Concept: Convex Stacking of Experts

PaCS-SR combines expert predictions using a **weighted average** where weights are constrained to be:

- **Non-negative**: `w_m >= 0` (each expert contributes positively)
- **Sum to one**: `sum(w_m) = 1` (weights represent proportions)

This is called a **convex combination** because the blended result lies within the "convex hull" of expert predictions.

### 2.2 Key Equations

#### Blending Equation

For each voxel, the super-resolved value is:

```
y_blend = w_BSPLINE × y_BSPLINE + w_ECLARE × y_ECLARE + w_SMORE × y_SMORE + w_UNIRES × y_UNIRES
```

Or more compactly for M experts:

```
y_blend = Σ (w_m × y_m)   where Σw_m = 1 and w_m >= 0
```

**Clinical interpretation**: If a region has weights `[0.1, 0.6, 0.2, 0.1]`, it means 60% trust in ECLARE, 20% in SMORE, etc.

#### Optimization Objective

For each 3D region `r`, PaCS-SR solves:

```
minimize    ||Σw_m × X_m - Y_HR||²  +  λ × ||w||²
    w

subject to  w_m >= 0  for all m
            Σw_m = 1
```

Where:
- `X_m` = expert m's prediction in region r
- `Y_HR` = ground-truth high-resolution target
- `λ` = ridge regularization (default: 1e-4)

This is a **constrained quadratic program** solved efficiently per region.

### 2.3 Patchwise Architecture

The volume is divided into overlapping 3D **tiles** (patches):

```
Volume (Z × Y × X)
    ↓
Divide into 32³ tiles with 50% overlap (stride=16)
    ↓
Learn independent weights per tile
    ↓
Blend with Hann windowing for smooth transitions
```

**Why patchwise?**
- Different brain regions have different optimal expert combinations
- Enables spatial analysis of model preferences
- Reduces memory: process one tile at a time

### 2.4 Regularization Techniques

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `lambda_ridge` | 1e-4 | Stabilize weight optimization |
| `laplacian_tau` | 0.05 | Smooth weights across adjacent tiles |
| `lambda_edge` | 0.25 | Emphasize edge voxels during training |
| `lambda_grad` | 0.2 | Match gradient (edge) information |

---

## 3. Interpretability Features

### 3.1 Weight Maps

For each patient, PaCS-SR outputs a **4D weight volume** of shape `(Z, Y, X, M)` where M=4 experts. Each voxel contains the weights showing expert contributions.

**How to interpret:**
- Visualize as heatmaps per expert
- Identify regions where specific experts dominate
- Compare across patients to find consistent patterns

### 3.2 Entropy Maps

The **entropy** at each voxel measures blending confidence:

```
H = -Σ w_m × log(w_m)
```

| Entropy | Interpretation |
|---------|---------------|
| Low (→0) | High confidence: one expert dominates |
| High (→log(M)) | Uncertainty: weights are uniform |

**Clinical use**: Low entropy regions show where the model is confident about which expert to use.

### 3.3 Dominant Model Maps

Shows which expert has the **highest weight** at each voxel:

```
dominant_model[x,y,z] = argmax(w[x,y,z,:])
```

This creates a segmentation-like visualization showing expert "territories" in the brain.

---

## 4. Usage Guide

### 4.1 Quick Start

```bash
# Train on one fold, spacing, and pulse
python run_demo.py --mode train --fold 1 --spacing 3mm --pulse t1c

# Run analysis (regional specialization + clinical validation)
python run_demo.py --mode analyze --results-dir ./results

# Generate visualizations
python run_demo.py --mode visualize --patient-id BraTS-XXX

# Complete demo pipeline
python run_demo.py --mode quick-start
```

### 4.2 Full Training Pipeline

```bash
# Step 1: Build K-fold manifest
pacs-sr-build-manifest --config configs/config.yaml

# Step 2: Train model
pacs-sr-train --config configs/config.yaml --fold 1 --spacing 3mm --pulse t1c

# Step 3: Apply to new data
pacs-sr-predict --config configs/config.yaml --weights path/to/weights.json
```

### 4.3 Configuration Parameters

Key parameters in `configs/config.yaml`:

```yaml
pacs_sr:
  patch_size: 32          # Tile size (8, 16, 24, or 32 voxels)
  stride: 16              # Overlap control (patch_size/2 = 50% overlap)
  simplex: true           # Enforce convex weights
  lambda_ridge: 1.0e-4    # Regularization strength
  use_registration: true  # Atlas-based alignment
  save_weight_volumes: true  # Save interpretability maps
```

---

## 5. Understanding Results

### 5.1 Output Directory Structure

```
results/{experiment_name}/{spacing}/
├── output_volumes/
│   └── {patient_id}-{pulse}.nii.gz          # Blended SR prediction
└── model_data/fold_{N}/{pulse}/
    ├── metrics.json                          # PSNR, SSIM scores
    ├── weights.json                          # Per-region weight vectors
    └── {patient_id}_weights_test.npz         # 4D weight maps
```

### 5.2 Evaluation Metrics

| Metric | Range | Interpretation | Use Case |
|--------|-------|----------------|----------|
| **PSNR** | Higher=better | Pixel-level fidelity | Technical quality |
| **SSIM** | 0-1 | Structural similarity | Perceptual quality |
| **LPIPS** | Lower=better | Deep perceptual distance | Visual realism |
| **KID** | Lower=better | Distributional similarity | Generation quality |

### 5.3 Reading Weight Maps

Load and visualize weight maps:

```python
import numpy as np

# Load weight maps
data = np.load("patient_weights_test.npz")
weight_maps = data["weight_maps"]  # Shape: (Z, Y, X, 4)
model_names = data["model_names"]  # ['BSPLINE', 'ECLARE', 'SMORE', 'UNIRES']

# Get dominant model per voxel
dominant = np.argmax(weight_maps, axis=-1)

# Compute entropy
entropy = -np.sum(weight_maps * np.log(weight_maps + 1e-8), axis=-1)
```

---

## 6. Scientific Experiments

### 6.1 Regional Specialization Analysis

**Question**: Which experts excel in which brain regions?

```bash
python -m pacs_sr.experiments.regional_specialization \
    --npz path/to/weights.npz \
    --seg path/to/segmentation.nii.gz \
    --output results/regional_analysis/
```

**Output**: Heatmap showing average expert weights per anatomical ROI.

### 6.2 Clinical Validation

**Question**: How does performance vary across tumor regions?

```bash
python -m pacs_sr.experiments.clinical_validation \
    --results-dir ./results \
    --methods BSPLINE ECLARE SMORE UNIRES PACS_SR \
    --output results/clinical_validation/
```

**Output**: Per-ROI metrics (tumor core, edema, surrounding tissue).

### 6.3 Cross-Resolution Generalization

**Question**: Do weights learned at one resolution transfer to others?

```bash
python -m pacs_sr.experiments.cross_resolution \
    --config configs/config.yaml \
    --train-spacing 3mm \
    --test-spacings 5mm 7mm \
    --output results/generalization/
```

**Output**: Transfer performance matrix showing generalization ability.

---

## Scientific Development Protocol

### 1. Evidence-Grounded Changes
- Every non-trivial decision must cite: a paper, a mathematical justification, or empirical data.
- "I think this is better" is not valid. "This reduces variance because [formula/reference]" is.
- When proposing architectural or methodological changes, state the expected effect and why.
- If no evidence exists, flag it explicitly as a hypothesis and propose a way to test it.

### 2. Research Workflow: Plan → Test → Analyze → Fix
**Planning phase:**
- Break the task into checkable items in `tasks/todo.md`.
- For each item, annotate: objective, success metric, and relevant references.
- Proactively flag: "Based on [paper/method], we could also try X — want me to include it?"
- Write specs before code. Ambiguity in spec = ambiguity in results.

**Testing phase:**
- Define quantitative success criteria before running anything.
- Log all hyperparameters, seeds, and environment details (reproducibility is non-negotiable).
- Use controlled comparisons: change one variable at a time unless explicitly doing ablations.

**Analysis phase:**
- Be proactive: if results reveal an anomaly or improvement opportunity, report it with evidence.
- Propose fixes or enhancements with: (a) what you found, (b) why it matters, (c) what to do.
- Always compute and report: mean, std, confidence intervals or statistical tests where applicable.
- Distinguish between statistically significant and practically significant differences.
- If a metric degrades, investigate root cause before proposing a fix.

**Fixing phase:**
- Fixes must reference what the analysis revealed. No blind patches.
- After fixing, re-run the relevant test to confirm the fix and check for regressions.
- Update `docs/tasks/lessons.md` or task-specific file with the failure mode and the corrective pattern.

### 3. Interdisciplinary Rigor (CS × AI × Biomedicine)
- Code changes: justify with computational complexity, memory, or convergence arguments.
- Model changes: justify with loss landscape, gradient dynamics, or information-theoretic reasoning.
- Clinical/biomedical changes: justify with domain constraints (e.g., anatomical priors, acquisition physics, class imbalance in rare pathologies).
- When in doubt about clinical validity, flag it — do not assume.

### 4. Proactive Scientific Agent Behavior
- During planning and analysis: if you identify a method, paper, or trick that could improve the current approach, **propose it immediately** with a one-line rationale.
- Suggest ablations or controls the user may not have considered.
- If a result contradicts expectations, form a hypothesis and propose a diagnostic experiment.
- Never silently ignore warnings, NaNs, or unexpected distributions — investigate and report.

### 5. Code & Experiment Standards
- All functions: typed, documented (docstring, no usage examples), brief inline comments.
- Prefer libraries over custom implementations. Cite the library and version.
- Logging over print. Use `logging` module with appropriate levels.
- Atomic functions, low cyclomatic complexity, OOP with dataclasses where appropriate.
- Experiment configs: use YAML/JSON, never hardcode hyperparameters in scripts.
- Random seeds must be set and logged. Results must be reproducible.

### 6. Communication Standards
- When reporting results: tables > prose. Include units, dataset split, and N.
- When proposing changes: state the current state, the proposed change, and the expected delta.
- When uncertain: quantify uncertainty. "This might work" → "This has ~X% chance based on [reasoning]."
- Use LaTeX notation for any mathematical expression in documentation or comments.

### 7. Verification & Self-Correction
- Never mark a task done without quantitative evidence it works.
- After any correction from the user: update `docs/tasks/lessons.md` or task-specific file with the pattern.
- Challenge your own proposals before presenting them. Ask: "What could go wrong?"
- If a subagent is used, verify its output — trust but verify.

---

## 7. Technical Reference

### 7.1 Expert Models

| Expert | Type | Description |
|--------|------|-------------|
| **BSPLINE** | Classical | B-spline interpolation |
| **ECLARE** | Deep Learning | Explicit Contrastive Learning for SR |
### 7.2 Algorithm Pseudocode

```
Input: Expert predictions {X_1, ..., X_M}, HR target Y
Output: Blended prediction, Weight maps

1. Divide volume into 3D tiles with overlap
2. For each tile r:
   a. Extract expert patches {X_1^r, ..., X_M^r} and target Y^r
   b. Build quadratic form: Q = Σ X_m × X_m^T, B = Σ X_m × Y
   c. Solve: min_w  w^T Q w - 2 B^T w + λ||w||²
             s.t.  w >= 0, Σw = 1
   d. Store weights W[r] = w
3. Optional: Apply Laplacian smoothing across tiles
4. Blend: For each voxel, y_blend = Σ W[r(voxel)] × X_m
5. Return blended volume and weight maps
```

### 7.3 Memory and Compute

- **Memory**: ~1.5 GB per worker (2.5 GB with registration)
- **Time**: ~5-10 min per patient (CPU, 6 workers)
- **Storage**: ~50 MB per patient (blends + weight maps)

---

## 8. Reproducibility

### 8.1 Required Dependencies

```bash
pip install pacs-sr[torch]  # For LPIPS/KID metrics
```

Core dependencies: numpy, scipy, nibabel, scikit-image, joblib

### 8.2 Recommended Settings

For clinical validation:
- `patch_size: 32` with `stride: 16` (50% overlap)
- `simplex: true` (interpretable weights)
- `use_registration: true` (atlas alignment)
- `kfolds: 5` (robust cross-validation)

For quick experiments:
- `patch_size: 16` with `stride: 16` (no overlap)
- `use_registration: false` (faster)
- `num_workers: 2` (laptop-friendly)

---

## 9. Citation

If you use PaCS-SR in your research, please cite:

```bibtex
@article{pascual2025pacsssr,
  title={PaCS-SR: Patchwise Convex Stacking for 3D MRI Super-Resolution under Clinical Anisotropy},
  author={Pascual, Mario and ...},
  journal={...},
  year={2025}
}
```

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `pacs_sr/model/model.py` | Core PatchwiseConvexStacker class |
| `pacs_sr/cli/train.py` | Training entry point |
| `pacs_sr/cli/predict.py` | Inference entry point |
| `pacs_sr/utils/weight_maps.py` | Weight analysis utilities |
| `pacs_sr/utils/visualize_pacs_sr.py` | Visualization functions |
| `pacs_sr/experiments/` | Scientific experiment scripts |
| `configs/config.yaml` | Full configuration template |
| `configs/demo_config.yaml` | Minimal demo configuration |
| `run_demo.py` | Simplified demo entry point |
