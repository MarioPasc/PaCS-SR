---
name: test
description: Run pytest tests for the neuromf project
allowed-tools:
  - Bash
argument-hint: "[test-path or pattern, e.g. 'tests/test_meanflow_loss.py' or '-k test_forward']"
---

# Test Runner

Run pytest for the neuromf project using the neuromf conda environment.

## Instructions

1. Run the tests specified by `$ARGUMENTS`. If no arguments given, run all tests in `tests/`.
2. Command format:
   ```bash
   ~/.conda/envs/neuromf/bin/python -m pytest $ARGUMENTS -v --tb=short
   ```
3. If tests fail, analyze the failures and suggest fixes.
4. If all tests pass, report the summary.
