"""
Parallel backend management for PaCS-SR.

Provides memory-safe parallel execution with support for both threading and loky backends.
Loky is preferred for SLURM environments as it avoids GIL contention and memory fragmentation.
"""

from __future__ import annotations
import os
import logging
from contextlib import contextmanager
from typing import Callable, Iterable, Any, Optional, Iterator
from joblib import Parallel, delayed
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class ParallelBackend:
    """
    Manages parallel execution with configurable backends.

    Supports:
    - "threading": Traditional thread-based parallelism (shared memory, GIL-limited)
    - "loky": Process-based parallelism with reusable worker pool (SLURM-friendly)

    Features:
    - Memory-safe execution with proper resource cleanup
    - Automatic temporary directory management for loky
    - Batch size optimization to reduce memory fragmentation
    - Graceful fallback on errors
    """

    def __init__(
        self,
        backend: str = "loky",
        n_jobs: int = 1,
        temp_folder: Optional[Path] = None,
        verbose: int = 0
    ):
        """
        Initialize parallel backend.

        Args:
            backend: "threading" or "loky"
            n_jobs: Number of parallel workers (-1 for all CPUs)
            temp_folder: Custom temp folder for loky (uses JOBLIB_TEMP_FOLDER or system temp)
            verbose: Verbosity level (0=silent, 10=debug)
        """
        self.backend = backend.lower()
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Validate backend
        if self.backend not in ["threading", "loky"]:
            logger.warning(f"Unknown backend '{backend}', falling back to 'loky'")
            self.backend = "loky"

        # Setup temp folder for loky
        if self.backend == "loky":
            if temp_folder is not None:
                self.temp_folder = Path(temp_folder)
            elif "JOBLIB_TEMP_FOLDER" in os.environ:
                self.temp_folder = Path(os.environ["JOBLIB_TEMP_FOLDER"])
            else:
                # Use system temp with unique subfolder
                self.temp_folder = Path(tempfile.gettempdir()) / "pacs_sr_joblib"

            # Create temp folder
            self.temp_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Loky temp folder: {self.temp_folder}")
        else:
            self.temp_folder = None

        # Configure backend-specific parameters
        self._configure_backend()

    def _configure_backend(self):
        """Configure backend-specific optimizations."""
        if self.backend == "loky":
            # Loky configuration for memory efficiency
            # Set environment variables before joblib import to avoid issues
            if "LOKY_MAX_CPU_COUNT" not in os.environ:
                # Limit to actual requested workers to avoid spawning extra processes
                os.environ["LOKY_MAX_CPU_COUNT"] = str(self.n_jobs if self.n_jobs > 0 else os.cpu_count())

            # Reduce worker timeout to release memory faster
            if "LOKY_WORKER_TIMEOUT" not in os.environ:
                os.environ["LOKY_WORKER_TIMEOUT"] = "300"  # 5 minutes (default is 10 min)

            logger.info(f"Loky backend configured: n_jobs={self.n_jobs}, temp={self.temp_folder}")
        else:
            logger.info(f"Threading backend configured: n_jobs={self.n_jobs}")

    @contextmanager
    def parallel_context(self, batch_size: Optional[int] = None) -> Iterator[Parallel]:
        """
        Context manager for parallel execution with automatic cleanup.

        Args:
            batch_size: Number of tasks per batch (helps reduce memory with loky)

        Yields:
            Configured Parallel object

        Example:
            >>> backend = ParallelBackend("loky", n_jobs=4)
            >>> with backend.parallel_context() as parallel:
            ...     results = parallel(delayed(func)(x) for x in data)
        """
        # Configure Parallel based on backend
        if self.backend == "loky":
            # Loky: process-based, good for CPU-bound tasks and avoiding GIL
            parallel = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
                verbose=self.verbose,
                temp_folder=str(self.temp_folder),
                batch_size=batch_size or "auto",  # Auto-batch for memory efficiency
                pre_dispatch="2*n_jobs",  # Limit pre-dispatched tasks to reduce memory
                max_nbytes="100M",  # Threshold for memmapping arrays (reduces copies)
            )
        else:
            # Threading: thread-based, good for I/O-bound tasks, shared memory
            parallel = Parallel(
                n_jobs=self.n_jobs,
                backend="threading",
                verbose=self.verbose,
                batch_size=batch_size or "auto",
            )

        try:
            yield parallel
        finally:
            # Explicit cleanup for loky
            if self.backend == "loky":
                # Force garbage collection to release worker memory
                import gc
                gc.collect()

    def map(
        self,
        func: Callable,
        iterable: Iterable[Any],
        batch_size: Optional[int] = None,
        desc: Optional[str] = None
    ) -> list[Any]:
        """
        Map function over iterable with parallel execution.

        Args:
            func: Function to apply to each element
            iterable: Iterable of inputs
            batch_size: Optional batch size for chunking
            desc: Optional description for logging

        Returns:
            List of results

        Example:
            >>> backend = ParallelBackend("loky", n_jobs=4)
            >>> results = backend.map(process_patient, patient_list)
        """
        if desc:
            logger.info(f"{desc} (backend={self.backend}, n_jobs={self.n_jobs})")

        with self.parallel_context(batch_size=batch_size) as parallel:
            results = parallel(delayed(func)(item) for item in iterable)

        return results

    def starmap(
        self,
        func: Callable,
        iterable: Iterable[tuple],
        batch_size: Optional[int] = None,
        desc: Optional[str] = None
    ) -> list[Any]:
        """
        Map function over iterable of argument tuples (like itertools.starmap).

        Args:
            func: Function to apply
            iterable: Iterable of argument tuples
            batch_size: Optional batch size
            desc: Optional description

        Returns:
            List of results

        Example:
            >>> backend = ParallelBackend("loky", n_jobs=4)
            >>> results = backend.starmap(process_patient, [(p, cfg) for p in patients])
        """
        if desc:
            logger.info(f"{desc} (backend={self.backend}, n_jobs={self.n_jobs})")

        with self.parallel_context(batch_size=batch_size) as parallel:
            results = parallel(delayed(func)(*args) for args in iterable)

        return results

    def cleanup(self):
        """Cleanup temporary files (for loky backend)."""
        if self.backend == "loky" and self.temp_folder and self.temp_folder.exists():
            try:
                import shutil
                # Only clean if we created it (not JOBLIB_TEMP_FOLDER from env)
                if "JOBLIB_TEMP_FOLDER" not in os.environ:
                    shutil.rmtree(self.temp_folder, ignore_errors=True)
                    logger.info(f"Cleaned up temp folder: {self.temp_folder}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp folder: {e}")


def create_backend(
    backend: str = "loky",
    n_jobs: int = 1,
    temp_folder: Optional[Path] = None,
    verbose: int = 0
) -> ParallelBackend:
    """
    Factory function to create a parallel backend.

    Args:
        backend: "threading" or "loky"
        n_jobs: Number of workers
        temp_folder: Optional temp folder for loky
        verbose: Verbosity level

    Returns:
        Configured ParallelBackend instance
    """
    return ParallelBackend(
        backend=backend,
        n_jobs=n_jobs,
        temp_folder=temp_folder,
        verbose=verbose
    )


# Convenience function for backward compatibility
def parallel_map(
    func: Callable,
    iterable: Iterable[Any],
    backend: str = "loky",
    n_jobs: int = 1,
    verbose: int = 0,
    desc: Optional[str] = None
) -> list[Any]:
    """
    Convenience function for parallel map with automatic backend selection.

    Args:
        func: Function to apply
        iterable: Iterable of inputs
        backend: "threading" or "loky"
        n_jobs: Number of workers
        verbose: Verbosity level
        desc: Optional description

    Returns:
        List of results
    """
    pb = create_backend(backend=backend, n_jobs=n_jobs, verbose=verbose)
    try:
        return pb.map(func, iterable, desc=desc)
    finally:
        pb.cleanup()
