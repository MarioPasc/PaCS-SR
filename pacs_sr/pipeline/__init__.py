"""
PaCS-SR End-to-End Pipeline Module
===================================

This module provides a complete pipeline for running PaCS-SR experiments
with a single command, including:

- K-fold cross-validation training
- Checkpoint/resume support
- Metrics aggregation and statistical analysis
- Publication-ready figure generation

Usage:
    pacs-sr-run --config configs/pipeline_config.yaml

Components:
    - PipelineOrchestrator: Main orchestration class
    - CheckpointManager: Fault-tolerant state management
    - PipelineContext: Shared execution context
    - Pipeline stages: setup, manifest, training, analysis, visualization
"""

from .checkpoint import CheckpointManager
from .context import PipelineContext
from .orchestrator import PipelineOrchestrator

__all__ = [
    "CheckpointManager",
    "PipelineContext",
    "PipelineOrchestrator",
]
