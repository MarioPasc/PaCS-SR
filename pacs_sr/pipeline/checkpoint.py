"""
Checkpoint Manager
==================

Manages pipeline state persistence for fault-tolerant execution.
Enables resume from interruption without losing progress.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    """Status of a pipeline task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointManager:
    """
    Manages pipeline state persistence and recovery.

    Saves state to JSON file after each task completion,
    enabling resume from interruption.
    """

    VERSION = "1.0"

    def __init__(self, state_path: Path):
        """
        Initialize checkpoint manager.

        Args:
            state_path: Path to checkpoint JSON file
        """
        self.state_path = state_path
        self._state: Dict[str, Any] = self._default_state()
        self._dirty = False

    def _default_state(self) -> Dict[str, Any]:
        """Create default empty state."""
        return {
            "version": self.VERSION,
            "experiment_name": "",
            "started_at": None,
            "last_updated_at": None,
            "config_hash": None,
            "stages": {},
            "training_tasks": {},
            "errors": [],
        }

    def load(self) -> bool:
        """
        Load checkpoint from disk.

        Returns:
            True if checkpoint was loaded, False if starting fresh
        """
        if not self.state_path.exists():
            return False

        try:
            with open(self.state_path, "r") as f:
                self._state = json.load(f)
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return False

    def save(self) -> None:
        """Persist current state to disk."""
        self._state["last_updated_at"] = datetime.now().isoformat()

        # Ensure parent directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically via temp file
        temp_path = self.state_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self._state, f, indent=2, default=str)
        temp_path.replace(self.state_path)

        self._dirty = False

    def initialize(
        self,
        experiment_name: str,
        config_hash: str,
        training_tasks: List[str],
    ) -> None:
        """
        Initialize checkpoint for a new experiment.

        Args:
            experiment_name: Name of the experiment
            config_hash: SHA256 hash of config for change detection
            training_tasks: List of task IDs to track
        """
        self._state["experiment_name"] = experiment_name
        self._state["started_at"] = datetime.now().isoformat()
        self._state["config_hash"] = config_hash

        # Initialize stages
        for stage in ["setup", "manifest", "training", "analysis", "metrics_computation",
                      "visualization", "pareto_visualization", "report"]:
            self._state["stages"][stage] = {
                "status": TaskStatus.PENDING.value,
                "started_at": None,
                "completed_at": None,
            }

        # Initialize training tasks
        for task_id in training_tasks:
            self._state["training_tasks"][task_id] = {
                "status": TaskStatus.PENDING.value,
                "started_at": None,
                "completed_at": None,
                "metrics": None,
                "outputs": None,
            }

        self._dirty = True
        self.save()

    @staticmethod
    def compute_config_hash(config_dict: Dict) -> str:
        """Compute SHA256 hash of config for change detection."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_config_hash(self) -> Optional[str]:
        """Get stored config hash."""
        return self._state.get("config_hash")

    # Stage management

    def mark_stage_started(self, stage: str) -> None:
        """Mark a stage as in-progress."""
        self._state["stages"][stage]["status"] = TaskStatus.IN_PROGRESS.value
        self._state["stages"][stage]["started_at"] = datetime.now().isoformat()
        self.save()

    def mark_stage_completed(self, stage: str) -> None:
        """Mark a stage as completed."""
        self._state["stages"][stage]["status"] = TaskStatus.COMPLETED.value
        self._state["stages"][stage]["completed_at"] = datetime.now().isoformat()
        self.save()

    def mark_stage_failed(self, stage: str, error: str) -> None:
        """Mark a stage as failed."""
        self._state["stages"][stage]["status"] = TaskStatus.FAILED.value
        self._state["stages"][stage]["error"] = error
        self._state["errors"].append({
            "stage": stage,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def is_stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed."""
        stage_data = self._state["stages"].get(stage, {})
        return stage_data.get("status") == TaskStatus.COMPLETED.value

    def get_stage_status(self, stage: str) -> TaskStatus:
        """Get current status of a stage."""
        stage_data = self._state["stages"].get(stage, {})
        status_str = stage_data.get("status", TaskStatus.PENDING.value)
        return TaskStatus(status_str)

    # Training task management

    def mark_task_started(self, task_id: str) -> None:
        """Mark a training task as in-progress."""
        if task_id not in self._state["training_tasks"]:
            self._state["training_tasks"][task_id] = {}

        self._state["training_tasks"][task_id]["status"] = TaskStatus.IN_PROGRESS.value
        self._state["training_tasks"][task_id]["started_at"] = datetime.now().isoformat()
        self.save()

    def mark_task_completed(
        self,
        task_id: str,
        metrics: Optional[Dict] = None,
        outputs: Optional[Dict] = None,
    ) -> None:
        """
        Mark a training task as completed.

        Args:
            task_id: Unique task identifier
            metrics: Optional metrics dict (psnr, ssim, etc.)
            outputs: Optional dict of output file paths
        """
        self._state["training_tasks"][task_id]["status"] = TaskStatus.COMPLETED.value
        self._state["training_tasks"][task_id]["completed_at"] = datetime.now().isoformat()

        if metrics:
            self._state["training_tasks"][task_id]["metrics"] = metrics
        if outputs:
            self._state["training_tasks"][task_id]["outputs"] = outputs

        self.save()

    def mark_task_failed(self, task_id: str, error: str, retry_count: int = 0) -> None:
        """Mark a training task as failed."""
        self._state["training_tasks"][task_id]["status"] = TaskStatus.FAILED.value
        self._state["training_tasks"][task_id]["error"] = error
        self._state["training_tasks"][task_id]["retries"] = retry_count

        self._state["errors"].append({
            "task": task_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "retries": retry_count,
        })
        self.save()

    def is_task_completed(self, task_id: str) -> bool:
        """Check if a training task is completed."""
        task_data = self._state["training_tasks"].get(task_id, {})
        return task_data.get("status") == TaskStatus.COMPLETED.value

    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get current status of a training task."""
        task_data = self._state["training_tasks"].get(task_id, {})
        status_str = task_data.get("status", TaskStatus.PENDING.value)
        return TaskStatus(status_str)

    def get_pending_tasks(self) -> List[str]:
        """Get list of pending training task IDs."""
        return [
            task_id
            for task_id, data in self._state["training_tasks"].items()
            if data.get("status") in (TaskStatus.PENDING.value, TaskStatus.FAILED.value)
        ]

    def get_completed_tasks(self) -> List[str]:
        """Get list of completed training task IDs."""
        return [
            task_id
            for task_id, data in self._state["training_tasks"].items()
            if data.get("status") == TaskStatus.COMPLETED.value
        ]

    def get_task_metrics(self, task_id: str) -> Optional[Dict]:
        """Get metrics for a completed task."""
        task_data = self._state["training_tasks"].get(task_id, {})
        return task_data.get("metrics")

    # Progress tracking

    def get_training_progress(self) -> Dict[str, int]:
        """
        Get training stage progress.

        Returns:
            Dict with completed, total, and pending counts
        """
        tasks = self._state["training_tasks"]
        completed = sum(1 for t in tasks.values() if t.get("status") == TaskStatus.COMPLETED.value)
        failed = sum(1 for t in tasks.values() if t.get("status") == TaskStatus.FAILED.value)
        pending = sum(1 for t in tasks.values() if t.get("status") == TaskStatus.PENDING.value)

        return {
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "total": len(tasks),
        }

    def get_all_task_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all completed tasks."""
        return {
            task_id: data.get("metrics", {})
            for task_id, data in self._state["training_tasks"].items()
            if data.get("status") == TaskStatus.COMPLETED.value and data.get("metrics")
        }

    def get_errors(self) -> List[Dict]:
        """Get list of all errors."""
        return self._state.get("errors", [])

    def clear_errors(self) -> None:
        """Clear error history."""
        self._state["errors"] = []
        self.save()
