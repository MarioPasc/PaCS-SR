"""
Pipeline Stages
===============

Individual stages that make up the PaCS-SR pipeline.
"""

from .base import PipelineStage, StageResult
from .setup import SetupStage
from .manifest import ManifestStage
from .training import TrainingStage
from .analysis import AnalysisStage
from .visualization import VisualizationStage
from .metrics_computation import MetricsComputationStage
from .pareto_visualization import ParetoVisualizationStage

__all__ = [
    "PipelineStage",
    "StageResult",
    "SetupStage",
    "ManifestStage",
    "TrainingStage",
    "AnalysisStage",
    "VisualizationStage",
    "MetricsComputationStage",
    "ParetoVisualizationStage",
]
