#!/usr/bin/env bash
# Compaction recovery hook — re-injects critical context after /compact
# Called by Claude Code on SessionStart[compact] events

cat <<'CONTEXT'

=== NEUROMF POST-COMPACTION CONTEXT ===

PROJECT: NeuroMF — Latent MeanFlow for 3D Brain MRI Synthesis
  Train MeanFlow in frozen MAISI VAE latent space (4×32³) for 1-NFE generation of 128³ brain MRI.

CRITICAL CONSTANTS:
  scale_factor = 0.96240234375  (from diffusion checkpoint, NOT VAE)
  VAE checkpoint wrapped in "unet_state_dict" key — must unwrap before load_state_dict
  Latent shape: (B, 4, 32, 32, 32) for 128³ input
  GPU: RTX 4060 Laptop, 8GB VRAM, max batch_size=1 for VAE

ENVIRONMENT:
  Python:  /home/mpascual/.conda/envs/neuromf/bin/python
  pytest:  ~/.conda/envs/neuromf/bin/python -m pytest tests/ -v --tb=short
  Config:  /home/mpascual/research/code/neuromf/configs/base.yaml
  Results: /media/mpascual/Sandisk2TB/research/neuromf/results/

KEY PATHS:
  Project:     /home/mpascual/research/code/neuromf/
  Core code:   src/neuromf/
  Tests:       tests/ (fixtures in conftest.py)
  VAE weights: /media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt
  IXI data:    /media/mpascual/Sandisk2TB/research/neuromf/datasets/IXI/IXI-T1/ (581 T1 volumes)
  Phases:      docs/splits/phase_{N}.md (read before implementing)

FORBIDDEN:
  - DO NOT modify src/external/ or docs/main/
  - DO NOT use diffusers (2D-only) or torchcfm (time convention mismatch)
  - DO NOT retrain the MAISI VAE

COMMANDS: /implement-phase N, /run-tests N, /check-gate N, /review-external
DEPENDENCIES: If you need a new package, add it to pyproject.toml and run:
  ~/.conda/envs/neuromf/bin/pip install -e "/home/mpascual/research/code/neuromf"

Re-read CLAUDE.md for full context: /home/mpascual/research/code/neuromf/CLAUDE.md

=== END COMPACTION RECOVERY ===
CONTEXT
