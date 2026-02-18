---
name: explore
description: Deep codebase exploration in isolated context
context: fork
agent: Explore
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
  - WebFetch
  - WebSearch
---

# Codebase Explorer

Thoroughly explore the neuromf codebase to answer: $ARGUMENTS

## Focus Areas
- File structure and module dependencies
- Configuration flow (Hydra YAML -> OmegaConf -> module)
- How the reference MeanFlow code in `src/external/` is wrapped
- Data flow from NIfTI loading through preprocessing to model input
- Time convention usage (t=0 data, t=1 noise)

## Key Locations
- Source: `src/neuromf/`
- External refs: `src/external/MeanFlow/`, `src/external/MeanFlow-PyTorch/`, `src/external/pmf/`, `src/external/MOTFM/`, `src/external/NV-Generate-CTMR/`
- Configs: `configs/`
- Tests: `tests/`
- Specs: `docs/splits/`
