"""
Pipeline Stage Base Class
=========================

Abstract base class for all pipeline stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from pacs_sr.pipeline.checkpoint import CheckpointManager
    from pacs_sr.pipeline.context import PipelineContext


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, message: str = "", data: Optional[Dict] = None) -> "StageResult":
        """Create a successful result."""
        return cls(success=True, message=message, data=data)

    @classmethod
    def fail(cls, message: str, data: Optional[Dict] = None) -> "StageResult":
        """Create a failed result."""
        return cls(success=False, message=message, data=data)


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage implements:
    - name: Unique identifier for the stage
    - run(): Execute the stage
    - is_completed(): Check if stage is already done
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this stage."""
        pass

    @abstractmethod
    def run(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> StageResult:
        """
        Execute the stage.

        Args:
            context: Pipeline execution context
            checkpoint: Checkpoint manager for state persistence

        Returns:
            StageResult indicating success or failure
        """
        pass

    def is_completed(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> bool:
        """
        Check if this stage is already completed.

        Args:
            context: Pipeline execution context
            checkpoint: Checkpoint manager

        Returns:
            True if stage is completed
        """
        return checkpoint.is_stage_completed(self.name)

    def should_skip(
        self,
        context: "PipelineContext",
        checkpoint: "CheckpointManager",
    ) -> bool:
        """
        Check if this stage should be skipped.

        Override in subclasses for custom skip logic.

        Args:
            context: Pipeline execution context
            checkpoint: Checkpoint manager

        Returns:
            True if stage should be skipped
        """
        return False

    def log(self, context: "PipelineContext", message: str, level: str = "info") -> None:
        """Log a message through the context logger."""
        context.log(f"[{self.name}] {message}", level)
