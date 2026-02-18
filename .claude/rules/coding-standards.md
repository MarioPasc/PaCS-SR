---
description: General coding standards for the neuromf project
---

# Coding Standards

1. **Type hints** on ALL function signatures and return types.
2. **Google-style docstrings** on all public functions and classes. No usage examples needed.
3. **Brief inline comments** on non-obvious code only. Do not comment obvious lines.
4. **Logging:** Python `logging` module with `rich` handler. INFO for training events, DEBUG for shapes/values.
5. **No magic numbers.** All hyperparameters from YAML configs via OmegaConf/Hydra.
6. **Prefer library functions.** MONAI transforms over custom preprocessing. `einops.rearrange` over manual reshapes. `F.scaled_dot_product_attention` over manual QKV matmuls.
7. **Tests use pytest.** Each test file runnable independently: `pytest tests/test_xxx.py -v`.
8. **Keep functions atomic.** One conceptual task per function.
9. **No magic numbers** — all hyperparams from YAML configs.
10. **Leverage reference codebases.** Start from PyTorch MeanFlow reference, do not reimplement tested patterns.
