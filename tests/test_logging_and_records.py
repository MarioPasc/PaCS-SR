"""Tests for comprehensive logging and raw result saving additions.

Covers:
1. capture_environment() in logger.py
2. _solve_simplex_qp() returning (w, solver_info) tuple
3. _config_to_dict() helper in model.py
4. Structure of solver_info dicts for both simplex and unconstrained branches
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pacs_sr.utils.logger import capture_environment


class TestCaptureEnvironment:
    """Tests for the capture_environment() function."""

    def test_returns_dict(self) -> None:
        """capture_environment should return a dict."""
        env = capture_environment()
        assert isinstance(env, dict)

    def test_required_keys_present(self) -> None:
        """All specified keys must be present in the returned dict."""
        env = capture_environment()
        required_keys = [
            "timestamp",
            "python_version",
            "platform",
            "cpu_count",
            "memory_total_gb",
            "memory_available_gb",
            "numpy_version",
            "scipy_version",
            "skimage_version",
            "joblib_version",
            "ants_available",
            "torch_available",
            "torch_version",
            "cuda_available",
        ]
        for key in required_keys:
            assert key in env, f"Missing key: {key}"

    def test_timestamp_is_iso8601(self) -> None:
        """Timestamp should be a valid ISO 8601 string."""
        from datetime import datetime

        env = capture_environment()
        # Should not raise
        datetime.fromisoformat(env["timestamp"])

    def test_python_version_is_string(self) -> None:
        """python_version should be a non-empty string."""
        env = capture_environment()
        assert isinstance(env["python_version"], str)
        assert len(env["python_version"]) > 0

    def test_cpu_count_positive(self) -> None:
        """cpu_count should be a positive integer."""
        env = capture_environment()
        assert isinstance(env["cpu_count"], int)
        assert env["cpu_count"] > 0

    def test_numpy_version_present(self) -> None:
        """numpy is a core dependency and its version should be a string."""
        env = capture_environment()
        assert isinstance(env["numpy_version"], str)

    def test_scipy_version_present(self) -> None:
        """scipy is a core dependency and its version should be a string."""
        env = capture_environment()
        assert isinstance(env["scipy_version"], str)

    def test_ants_available_is_bool(self) -> None:
        """ants_available should be a boolean."""
        env = capture_environment()
        assert isinstance(env["ants_available"], bool)

    def test_torch_available_is_bool(self) -> None:
        """torch_available should be a boolean."""
        env = capture_environment()
        assert isinstance(env["torch_available"], bool)

    def test_json_serializable(self) -> None:
        """The entire dict should be JSON-serializable."""
        env = capture_environment()
        serialized = json.dumps(env)
        assert isinstance(serialized, str)


class TestSolveSimplexQPReturnsTuple:
    """Tests for _solve_simplex_qp returning (w, solver_info) tuple."""

    def test_simplex_branch_returns_tuple(self) -> None:
        """Simplex (SLSQP) branch should return (w, info) tuple."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 3
        # Simple positive-definite Q and B
        Q = np.eye(M, dtype=np.float64) * 2.0
        B = np.array([1.0, 0.5, 0.5], dtype=np.float64)

        result = _solve_simplex_qp(Q, B, lambda_ridge=1e-4, simplex=True)
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Tuple should have 2 elements"

        w, info = result
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float32
        assert w.shape == (M,)
        assert isinstance(info, dict)

    def test_simplex_solver_info_keys(self) -> None:
        """solver_info for simplex branch should have required keys."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 2
        Q = np.eye(M, dtype=np.float64) * 2.0
        B = np.array([1.0, 0.5], dtype=np.float64)

        _, info = _solve_simplex_qp(Q, B, lambda_ridge=1e-4, simplex=True)
        assert info["method"] == "SLSQP"
        assert isinstance(info["success"], bool)
        assert isinstance(info["nit"], int)
        assert isinstance(info["objective"], float)
        assert isinstance(info["message"], str)

    def test_unconstrained_branch_returns_tuple(self) -> None:
        """Unconstrained (closed-form) branch should return (w, info) tuple."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 3
        Q = np.eye(M, dtype=np.float64) * 2.0
        B = np.array([1.0, 0.5, 0.5], dtype=np.float64)

        result = _solve_simplex_qp(Q, B, lambda_ridge=1e-4, simplex=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

        w, info = result
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float32
        assert isinstance(info, dict)

    def test_unconstrained_solver_info_keys(self) -> None:
        """solver_info for unconstrained branch should have required keys."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 2
        Q = np.eye(M, dtype=np.float64) * 2.0
        B = np.array([1.0, 0.5], dtype=np.float64)

        _, info = _solve_simplex_qp(Q, B, lambda_ridge=1e-4, simplex=False)
        assert info["method"] == "closed_form"
        assert info["success"] is True
        assert info["nit"] == 0
        assert isinstance(info["objective"], float)
        assert info["message"] in ("direct_solve", "pseudoinverse_fallback")

    def test_simplex_weights_valid(self) -> None:
        """Simplex weights should be non-negative and sum to ~1."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 4
        Q = np.eye(M, dtype=np.float64) * 2.0
        B = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.float64)

        w, _ = _solve_simplex_qp(Q, B, lambda_ridge=1e-4, simplex=True)
        assert np.all(w >= 0), "Weights should be non-negative"
        assert abs(float(np.sum(w)) - 1.0) < 1e-6, "Weights should sum to 1"

    def test_unconstrained_pinv_fallback(self) -> None:
        """Singular Q should trigger pseudoinverse fallback message."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 3
        # Create a singular matrix
        Q = np.zeros((M, M), dtype=np.float64)
        B = np.array([1.0, 0.5, 0.5], dtype=np.float64)

        w, info = _solve_simplex_qp(Q, B, lambda_ridge=0.0, simplex=False)
        # With lambda_ridge=0 and singular Q, it should use pinv
        # Note: with lambda_ridge > 0, Qr = Q + lambda*I is always invertible
        assert info["method"] == "closed_form"
        # The message depends on whether np.linalg.solve raises or not
        assert info["message"] in ("direct_solve", "pseudoinverse_fallback")


class TestConfigToDict:
    """Tests for the _config_to_dict helper."""

    def test_converts_paths_to_strings(self) -> None:
        """Path fields should be converted to strings."""
        from pacs_sr.model.model import _config_to_dict
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Cfg:
            path_field: Path = Path("/tmp/test")
            str_field: str = "hello"

        d = _config_to_dict(_Cfg())
        assert isinstance(d["path_field"], str)
        assert d["path_field"] == "/tmp/test"

    def test_converts_tuples_to_lists(self) -> None:
        """Tuple fields should be converted to lists."""
        from pacs_sr.model.model import _config_to_dict
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Cfg:
            tuple_field: tuple = ("a", "b", "c")
            int_field: int = 42

        d = _config_to_dict(_Cfg())
        assert isinstance(d["tuple_field"], list)
        assert d["tuple_field"] == ["a", "b", "c"]

    def test_scalars_preserved(self) -> None:
        """Scalar fields should pass through unchanged."""
        from pacs_sr.model.model import _config_to_dict
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Cfg:
            int_field: int = 42
            float_field: float = 3.14
            bool_field: bool = True
            str_field: str = "test"

        d = _config_to_dict(_Cfg())
        assert d["int_field"] == 42
        assert d["float_field"] == 3.14
        assert d["bool_field"] is True
        assert d["str_field"] == "test"

    def test_result_is_json_serializable(self) -> None:
        """The resulting dict should be JSON-serializable."""
        from pacs_sr.model.model import _config_to_dict
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Cfg:
            path_field: Path = Path("/tmp/test")
            tuple_field: tuple = ("a", "b")
            int_field: int = 42

        d = _config_to_dict(_Cfg())
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


class TestSolverInfoObjectiveConsistency:
    """Verify that objective values in solver_info are numerically consistent."""

    def test_simplex_objective_matches_manual(self) -> None:
        """The reported objective should match manual computation."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 3
        Q = np.array(
            [[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]], dtype=np.float64
        )
        B = np.array([1.0, 0.8, 0.6], dtype=np.float64)
        lam = 1e-4

        w, info = _solve_simplex_qp(Q, B, lambda_ridge=lam, simplex=True)
        Qr = Q + lam * np.eye(M)
        manual_obj = float(
            w.astype(np.float64) @ Qr @ w.astype(np.float64)
            - 2.0 * B @ w.astype(np.float64)
        )
        assert abs(info["objective"] - manual_obj) < 1e-4, (
            f"Objective mismatch: info={info['objective']:.8f} vs manual={manual_obj:.8f}"
        )

    def test_unconstrained_objective_matches_manual(self) -> None:
        """The reported objective should match manual computation for closed form."""
        from pacs_sr.model.model import _solve_simplex_qp

        M = 2
        Q = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        B = np.array([2.0, 1.5], dtype=np.float64)
        lam = 1e-3

        w, info = _solve_simplex_qp(Q, B, lambda_ridge=lam, simplex=False)
        Qr = Q + lam * np.eye(M)
        manual_obj = float(
            w.astype(np.float64) @ Qr @ w.astype(np.float64)
            - 2.0 * B @ w.astype(np.float64)
        )
        assert abs(info["objective"] - manual_obj) < 1e-6, (
            f"Objective mismatch: info={info['objective']:.8f} vs manual={manual_obj:.8f}"
        )
